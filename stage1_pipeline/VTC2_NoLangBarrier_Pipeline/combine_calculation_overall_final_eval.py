"""
Combined Transcription Evaluation Report Generator
====================================================
Reads all *.combined.txt files from subfolders structured as:
    recording_name/recording_name_clipname/*.combined.txt

Outputs one combined report per recording:
    recording_name/output.txt

Usage:
    python combine_reports.py --folder /path/to/recordings
"""

import os
import re
import argparse
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
# 1. PARSING
# ─────────────────────────────────────────────────────────────────

def parse_report(filepath):
    """Parse a single *.combined.txt file and return a dict of all values."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    data = {}

    # ── Detect speakers dynamically from column header line ────────
    # Looks for the header line like: Metric   MAL   FEM   KCHI   ADULT   OVERALL
    header_match = re.search(r"^Metric\s+((?:[A-Z]+\s+)+)", text, re.MULTILINE)
    if not header_match:
        raise ValueError(f"Could not find speaker header in {filepath}")

    detected_speakers = header_match.group(1).split()
    data["speakers"] = detected_speakers

    # ── WER metrics table ──────────────────────────────────────────
    wer_metrics = ["WER", "MER", "WIL", "WIP", "CER"]
    float_groups = r"\s+".join([r"([\d.]+)"] * len(detected_speakers))
    for metric in wer_metrics:
        data[metric] = {}
        pattern = rf"^{metric}\s+{float_groups}"
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            for i, s in enumerate(detected_speakers):
                try:
                    data[metric][s] = float(m.group(i + 1))
                except (ValueError, IndexError):
                    data[metric][s] = 0.0

    # ── Word counts table ──────────────────────────────────────────
    int_groups = r"\s+".join([r"([\d]+)"] * len(detected_speakers))
    count_fields = {
        "ref_words":     rf"ref_words_count\s+{int_groups}",
        "hits":          rf"hits\s+{int_groups}",
        "substitutions": rf"substitutions\s+{int_groups}",
        "insertions":    rf"insertions\s+{int_groups}",
        "deletions":     rf"deletions\s+{int_groups}",
    }
    for field, pattern in count_fields.items():
        data[field] = {}
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            for i, s in enumerate(detected_speakers):
                try:
                    data[field][s] = int(m.group(i + 1))
                except (ValueError, IndexError):
                    data[field][s] = 0

    # ── Language distribution – REFERENCE ─────────────────────────
    data["ref_lang"] = {}
    for spk in detected_speakers:
        pattern = rf"^{spk}\s+([\d]+)\s+([\d]+)\s+([\d]+)\s+([\d.]+)\s*%"
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            m = matches[0]
            data["ref_lang"][spk] = {
                "spanish": int(m.group(1)),
                "english": int(m.group(2)),
                "total":   int(m.group(3)),
                "pct":     float(m.group(4)),
            }
        else:
            data["ref_lang"][spk] = {"spanish": 0, "english": 0, "total": 0, "pct": 0.0}

    # ── Language distribution – HYPOTHESIS ────────────────────────
    data["hyp_lang"] = {}
    hyp_split = re.split(r"HYPOTHESIS", text, maxsplit=1)
    hyp_section = hyp_split[1] if len(hyp_split) > 1 else ""
    for spk in detected_speakers:
        pattern = rf"^{spk}\s+([\d]+)\s+([\d]+)\s+([\d]+)\s+([\d.]+)\s*%"
        m = re.search(pattern, hyp_section, re.MULTILINE)
        if m:
            data["hyp_lang"][spk] = {
                "spanish": int(m.group(1)),
                "english": int(m.group(2)),
                "total":   int(m.group(3)),
                "pct":     float(m.group(4)),
            }
        else:
            data["hyp_lang"][spk] = {"spanish": 0, "english": 0, "total": 0, "pct": 0.0}

    return data


# ─────────────────────────────────────────────────────────────────
# 2. AGGREGATION  (sum raw counts, recompute metrics)
# ─────────────────────────────────────────────────────────────────

def aggregate(all_data):
    """Sum raw word counts across all clips, then recompute all metrics."""

    # Collect all unique speakers across all files (preserve order)
    seen = set()
    all_speakers = []
    for d in all_data:
        for s in d["speakers"]:
            if s not in seen:
                seen.add(s)
                all_speakers.append(s)

    agg = {
        "ref_words":     {s: 0 for s in all_speakers},
        "hits":          {s: 0 for s in all_speakers},
        "substitutions": {s: 0 for s in all_speakers},
        "insertions":    {s: 0 for s in all_speakers},
        "deletions":     {s: 0 for s in all_speakers},
        "ref_lang":      {s: {"spanish": 0, "english": 0} for s in all_speakers},
        "hyp_lang":      {s: {"spanish": 0, "english": 0} for s in all_speakers},
    }

    for d in all_data:
        for s in d["speakers"]:
            agg["ref_words"][s]     += d["ref_words"].get(s, 0)
            agg["hits"][s]          += d["hits"].get(s, 0)
            agg["substitutions"][s] += d["substitutions"].get(s, 0)
            agg["insertions"][s]    += d["insertions"].get(s, 0)
            agg["deletions"][s]     += d["deletions"].get(s, 0)

            for lang in ("spanish", "english"):
                agg["ref_lang"][s][lang] += d["ref_lang"].get(s, {}).get(lang, 0)
                agg["hyp_lang"][s][lang] += d["hyp_lang"].get(s, {}).get(lang, 0)

    # Recompute derived metrics from aggregated counts
    metrics = {m: {} for m in ["WER", "MER", "WIL", "WIP", "CER", "accuracy"]}

    for s in all_speakers:
        ref  = agg["ref_words"][s]
        hits = agg["hits"][s]
        subs = agg["substitutions"][s]
        ins  = agg["insertions"][s]
        dels = agg["deletions"][s]

        WER = (subs + ins + dels) / ref            if ref > 0 else 0.0
        MER = (subs + dels) / (hits + subs + dels) if (hits + subs + dels) > 0 else 0.0
        WIP = hits / ref                            if ref > 0 else 0.0
        WIL = 1 - WIP
        acc = 1 - WER
        CER = WER * 0.75  # approximation; replace if char-level counts are available

        metrics["WER"][s]      = WER
        metrics["MER"][s]      = MER
        metrics["WIL"][s]      = WIL
        metrics["WIP"][s]      = WIP
        metrics["CER"][s]      = CER
        metrics["accuracy"][s] = acc

        for source in ("ref_lang", "hyp_lang"):
            sp  = agg[source][s]["spanish"]
            en  = agg[source][s]["english"]
            tot = sp + en
            pct = (sp / tot * 100) if tot > 0 else 0.0
            agg[source][s]["total"] = tot
            agg[source][s]["pct"]   = pct

    return agg, metrics, all_speakers


# ─────────────────────────────────────────────────────────────────
# 3. REPORT WRITING
# ─────────────────────────────────────────────────────────────────

SEP  = "=" * 120
SEP2 = "-" * 120
COL  = 19  # column width


def row(label, values, fmt=".4f"):
    cells = "".join(f"{v:{fmt}}{'':<{COL - len(f'{v:{fmt}}')}} " for v in values)
    return f"{label:<31}{cells}\n"


def write_report(agg, metrics, all_speakers, out_path, n_clips, recording_name, all_data, clip_names):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    A = all_speakers

    def hdr(title):
        lines.append(SEP + "\n")
        lines.append(title.center(120) + "\n")
        lines.append(SEP + "\n")

    def sub(title):
        lines.append(SEP2 + "\n")
        lines.append(title + "\n")
        lines.append(SEP2 + "\n")

    def col_hdr():
        h = f"{'Metric':<31}" + "".join(f"{s:<{COL}} " for s in A)
        lines.append(h + "\n")
        lines.append(SEP2 + "\n")

    # ── Title ──────────────────────────────────────────────────────
    hdr(f"COMBINED TRANSCRIPTION EVALUATION REPORT — {recording_name} ({n_clips} CLIPS)")
    lines.append(f"Generated on: {now}\n\n")

    # ── WER Metrics ────────────────────────────────────────────────
    sub("SUMMARY COMPARISON - WER METRICS")
    col_hdr()
    for metric in ["WER", "MER", "WIL", "WIP", "CER"]:
        vals = [metrics[metric].get(s, 0.0) for s in A]
        lines.append(row(metric, vals))
    lines.append("\n")

    # ── Word Counts ────────────────────────────────────────────────
    sub("WORD COUNTS AND ACCURACY")
    col_hdr()
    int_fields = [
        ("ref_words_count", "ref_words"),
        ("hits",            "hits"),
        ("substitutions",   "substitutions"),
        ("insertions",      "insertions"),
        ("deletions",       "deletions"),
    ]
    for label, key in int_fields:
        vals = [agg[key].get(s, 0) for s in A]
        lines.append(row(label, vals, fmt="d"))

    acc_vals = [metrics["accuracy"].get(s, 0.0) for s in A]
    acc_line = f"{'Accuracy %':<31}" + "".join(
        f"{v*100:.2f}{'%':<{COL - len(f'{v*100:.2f}')}} " for v in acc_vals
    )
    lines.append(acc_line + "\n\n")

    # ── Language Distribution – Reference ─────────────────────────
    hdr("LANGUAGE DISTRIBUTION ANALYSIS - REFERENCE (GROUND TRUTH)")
    lang_hdr = (f"{'Speaker':<31}{'Spanish Words':<{COL}} {'English Words':<{COL}} "
                f"{'Total':<{COL}} {'Spanish %':<{COL}}\n")
    lines.append(lang_hdr)
    lines.append(SEP2 + "\n")
    for s in A:
        d = agg["ref_lang"][s]
        lines.append(
            f"{s:<31}{d['spanish']:<{COL}} {d['english']:<{COL}} "
            f"{d['total']:<{COL}} {d['pct']:.2f}{'%':<{COL}}\n"
        )
    lines.append("\n")

    # ── Language Distribution – Hypothesis ────────────────────────
    hdr("LANGUAGE DISTRIBUTION ANALYSIS - HYPOTHESIS (TRAINEE)")
    lines.append(lang_hdr)
    lines.append(SEP2 + "\n")
    for s in A:
        d = agg["hyp_lang"][s]
        lines.append(
            f"{s:<31}{d['spanish']:<{COL}} {d['english']:<{COL}} "
            f"{d['total']:<{COL}} {d['pct']:.2f}{'%':<{COL}}\n"
        )
    lines.append("\n")

    # ── Calculation Breakdown ─────────────────────────────────────
    hdr("CALCULATION BREAKDOWN (FOR VERIFICATION)")
    lines.append("Formulas used:\n")
    lines.append("  WER      = (substitutions + insertions + deletions) / ref_words\n")
    lines.append("  MER      = (substitutions + deletions) / (hits + substitutions + deletions)\n")
    lines.append("  WIP      = hits / ref_words\n")
    lines.append("  WIL      = 1 - WIP\n")
    lines.append("  CER      = WER * 0.75  (approximation)\n")
    lines.append("  Accuracy = 1 - WER\n")
    lines.append("\n")

    # Column widths for per-clip table
    CW = 18  # count column width
    NW = 55  # filename column width

    for s in A:
        lines.append(SEP2 + "\n")
        lines.append(f"Speaker: {s}\n")
        lines.append(SEP2 + "\n")

        # Per-clip table header
        lines.append(f"  {'Clip File':<{NW}} {'ref_words':>{CW}} {'hits':>{CW}} {'subs':>{CW}} {'ins':>{CW}} {'dels':>{CW}}\n")
        lines.append(f"  {'-'*NW} {'-'*CW} {'-'*CW} {'-'*CW} {'-'*CW} {'-'*CW}\n")

        # Per-clip rows
        for clip_name, clip_data in zip(clip_names, all_data):
            # Only show if speaker exists in this clip
            r = clip_data["ref_words"].get(s, 0)
            h = clip_data["hits"].get(s, 0)
            su = clip_data["substitutions"].get(s, 0)
            i = clip_data["insertions"].get(s, 0)
            d = clip_data["deletions"].get(s, 0)
            lines.append(f"  {clip_name:<{NW}} {r:>{CW}} {h:>{CW}} {su:>{CW}} {i:>{CW}} {d:>{CW}}\n")

        # Totals row
        ref  = agg["ref_words"][s]
        hits = agg["hits"][s]
        subs = agg["substitutions"][s]
        ins  = agg["insertions"][s]
        dels = agg["deletions"][s]

        lines.append(f"  {'-'*NW} {'-'*CW} {'-'*CW} {'-'*CW} {'-'*CW} {'-'*CW}\n")
        lines.append(f"  {'TOTAL':<{NW}} {ref:>{CW}} {hits:>{CW}} {subs:>{CW}} {ins:>{CW}} {dels:>{CW}}\n")
        lines.append("\n")

        # Calculation from totals
        wer = metrics["WER"][s]
        mer = metrics["MER"][s]
        wip = metrics["WIP"][s]
        wil = metrics["WIL"][s]
        cer = metrics["CER"][s]
        acc = metrics["accuracy"][s]

        lines.append(f"  WER      = ({subs} + {ins} + {dels}) / {ref} = {subs + ins + dels} / {ref} = {wer:.4f}\n")
        lines.append(f"  MER      = ({subs} + {dels}) / ({hits} + {subs} + {dels}) = {subs + dels} / {hits + subs + dels} = {mer:.4f}\n")
        lines.append(f"  WIP      = {hits} / {ref} = {wip:.4f}\n")
        lines.append(f"  WIL      = 1 - {wip:.4f} = {wil:.4f}\n")
        lines.append(f"  CER      = {wer:.4f} * 0.75 = {cer:.4f}\n")
        lines.append(f"  Accuracy = 1 - {wer:.4f} = {acc:.4f} ({acc*100:.2f}%)\n")
        lines.append("\n")

    # ── Best Performer (exclude OVERALL) ──────────────────────────
    main_spks = [s for s in A if s != "OVERALL"]
    if main_spks:
        best = min(main_spks, key=lambda s: metrics["WER"].get(s, float("inf")))
        lines.append(SEP + "\n")
        lines.append(f"BEST PERFORMER: {best}\n")
        lines.append(f"WER: {metrics['WER'][best]:.4f}\n")
        lines.append(SEP + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"     Report written to : {out_path}")
    print(f"       Clips processed   : {n_clips}")
    if "OVERALL" in metrics["WER"]:
        print(f"       Combined WER      : {metrics['WER']['OVERALL']:.4f}")
        print(f"       Combined CER      : {metrics['CER']['OVERALL']:.4f}")
    if main_spks:
        print(f"       Best speaker      : {best} (WER={metrics['WER'][best]:.4f})")


# ─────────────────────────────────────────────────────────────────
# 4. DISCOVER FILES — group by recording_name
# ─────────────────────────────────────────────────────────────────

def discover_files(root_folder):
    """
    Nested folder structure:
        root_folder/
            P08_pre/                                <- recording_name
                P08_pre_001/                        <- clip folder
                    P08_pre_001_combined_evaluation.txt
                P08_pre_002/
                    P08_pre_002_combined_evaluation.txt
            P00_pre/
                P00_pre/
                    P00_pre.txt

    Returns a dict:
        { recording_name: {"files": [...], "recording_path": "..."} }
    """
    recording_map = defaultdict(lambda: {"files": [], "recording_path": ""})

    for recording_name in sorted(os.listdir(root_folder)):
        recording_path = os.path.join(root_folder, recording_name)
        if not os.path.isdir(recording_path):
            continue

        # Walk clip subfolders inside the recording folder
        for clip_folder in sorted(os.listdir(recording_path)):
            clip_path = os.path.join(recording_path, clip_folder)
            if not os.path.isdir(clip_path):
                continue

            for fname in sorted(os.listdir(clip_path)):
                if fname.endswith("_combined_evaluation.txt"):
                    recording_map[recording_name]["files"].append(
                        os.path.join(clip_path, fname)
                    )
                    recording_map[recording_name]["recording_path"] = recording_path

    return recording_map


# ─────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combine clip evaluation reports per recording.")
    parser.add_argument("--folder", required=True,
                        help="Root folder containing recording subfolders")
    args = parser.parse_args()

    root_folder = args.folder
    recording_map = discover_files(root_folder)

    if not recording_map:
        print(f"ERROR: No *.combined.txt files found under: {root_folder}")
        return

    print(f"  Found {len(recording_map)} recording(s) in: {root_folder}\n")

    for recording_name, info in recording_map.items():
        files = info["files"]
        recording_path = info["recording_path"]

        print(f"  Processing: {recording_name}  ({len(files)} clip file(s))")

        all_data = []
        clip_names = []
        for fp in files:
            print(f"   Parsing: {os.path.basename(fp)}")
            try:
                all_data.append(parse_report(fp))
                clip_names.append(os.path.basename(fp))
            except Exception as e:
                print(f"   WARNING: Skipped {fp}: {e}")

        if not all_data:
            print(f"   ERROR: No valid reports parsed for {recording_name}\n")
            continue

        agg, metrics, all_speakers = aggregate(all_data)

        # Output goes into the recording_name folder
        out_path = os.path.join(recording_path, "output.txt")
        write_report(agg, metrics, all_speakers, out_path,
                     n_clips=len(all_data), recording_name=recording_name,
                     all_data=all_data, clip_names=clip_names)
        print()


if __name__ == "__main__":
    main()


# Example usage:
# python combine_reports.py --folder /path/to/root_folder
#
# Expected folder structure:
#   root_folder/
#     P08_pre/
#       P08_pre_001/
#         P08_pre_001_combined_evaluation.txt
#       P08_pre_002/
#         P08_pre_002_combined_evaluation.txt
#     P00_pre/
#       P00_pre/
#         P00_pre.txt
#
# Output (one file per recording, inside recording folder):
#   root_folder/P08_pre/output.txt
#   root_folder/P00_pre/output.txt
