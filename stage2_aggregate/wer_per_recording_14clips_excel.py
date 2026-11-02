"""
pipeline_comparison.py
======================
Reads per-clip *_evaluation.txt files produced by WhisperX pipelines,
aggregates metrics correctly from raw counts, and writes a formatted Excel
workbook.

OVERALL semantics
-----------------
OVERALL is the *session-level* ASR WER for the recording: it is pooled across
clips from each clip's `*_overall_evaluation.txt` (which the upstream
evaluator computes by aligning the full hypothesis against the full reference
in one pass). This is the standard ASR WER.

We do NOT compute OVERALL by summing counts across per-speaker files, because
that double-counts diarization errors (a correct word attributed to the wrong
speaker becomes one deletion in the correct speaker's file AND one insertion
in the wrong speaker's file — 2 errors instead of 0).
"""

import re
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
_PROJ = Path(__file__).parent
_OUT  = _PROJ / "output_14clips"

PIPELINES = [
    ("Human-TS + Whisper",     str(_OUT / "human_timestamps"    / "eval_report")),
    ("VTC1 + Whisper",         str(_OUT / "vtc1"                / "eval_report")),
    ("VTC2 + Whisper",         str(_OUT / "vtc2"                / "eval_report")),
    ("VTC2-NoLang + Whisper",  str(_OUT / "vtc2_nolang_barrier" / "eval_report")),
    ("Only Whisper",           str(_OUT / "whisperx_only"       / "eval_report")),
]
OUTPUT_FILE = str(_PROJ / "wer_per_recording_14clips.xlsx")

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
SPEAKER_MAP = {
    "fem": "FEM", "mal": "MAL", "kchi": "KCHI", "och": "OCH",
    "adult": "ADULT", "children": "CHILDREN",
    "overall": "OVERALL", "combined": "COMBINED",
}

# Special "speaker" labels that are not real speakers but session-level
# aggregates produced by the upstream evaluator. We handle them separately
# from per-speaker rows.
OVERALL_LABEL  = "OVERALL"   # session-level ASR WER (full hyp vs full ref)
COMBINED_LABEL = "COMBINED"  # alternate session aggregate; ignored here

SPEAKER_ORDER = ["MAL", "FEM", "KCHI", "OCH", "ADULT", "CHILDREN", "OVERALL"]

INT_KEYS = ["ref_words_count", "hits", "substitutions", "deletions", "insertions"]

# All metrics shown on per-speaker sheets
METRICS = [
    ("WER",             "WER",             True),
    ("CER†",            "CER",             True),
    ("Accuracy",        "ACC_PCT",         False),
    ("Insertions %",    "INS_PCT",         False),
    ("Deletions %",     "DEL_PCT",         False),
    ("Substitutions %", "SUB_PCT",         False),
    ("MER",             "MER",             True),
    ("WIL",             "WIL",             True),
    ("WIP",             "WIP",             True),
    ("Ref Words",       "ref_words_count", False),
    ("Hits",            "hits",            False),
    ("Substitutions",   "substitutions",   False),
    ("Deletions",       "deletions",       False),
    ("Insertions",      "insertions",      False),
]

# Key metrics shown on Summary sheet
KEY_METRICS = [
    ("WER",             "WER",     True),
    ("Accuracy",        "ACC_PCT", False),
    ("Insertions %",    "INS_PCT", False),
    ("Deletions %",     "DEL_PCT", False),
    ("Substitutions %", "SUB_PCT", False),
    ("CER†",            "CER",     True),
]

# ─── Palette ──────────────────────────────────────────────────
HDR_DARK  = "1F3864"
HDR_MED   = "2E75B6"
HDR_LIGHT = "D6E4F0"
PIPE_COLS = ["D9EAD3", "FFF2CC", "FCE5CD", "EAD1DC"]
WHITE     = "FFFFFF"
ALT_ROW   = "F5F8FF"
SPK_COLS  = ["1F3864", "2E75B6", "2F9C6E", "7B3F7A", "C0392B", "6E4B2E", "555555"]

FOOTNOTE = (
    "† CER is averaged across clips (character counts unavailable); all other rates recomputed from summed raw counts. "
    "OVERALL = session-level ASR WER pooled from per-clip *_overall_evaluation.txt files (full hypothesis vs full reference), "
    "not the sum across per-speaker files (which would double-count diarization errors)."
)


# ──────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ──────────────────────────────────────────────────────────────
def recompute_rates(agg: dict) -> dict:
    """
    Given a dict with summed integer counts, recompute all derived rates in-place.

    Formulas
    --------
    WER = (S + D + I) / N
    MER = (S + D + I) / (H + S + D + I)
    WIP = H^2 / ((H + S + D) * (H + I))
    WIL = 1 - WIP
    ACC = (1 - WER) * 100
    INS/DEL/SUB % = count / N * 100

    CER is NOT recomputed (no character counts available).
    """
    H = agg.get("hits",            0)
    S = agg.get("substitutions",   0)
    D = agg.get("deletions",       0)
    I = agg.get("insertions",      0)
    N = agg.get("ref_words_count", 0)

    errors    = S + D + I
    total_hyp = H + S + D + I
    denom_wip = (H + S + D) * (H + I)

    agg["WER"] = errors / N          if N          > 0 else 0.0
    agg["MER"] = errors / total_hyp  if total_hyp  > 0 else 0.0
    agg["WIP"] = (H * H) / denom_wip if denom_wip  > 0 else 0.0
    agg["WIL"] = 1.0 - agg["WIP"]

    agg["ACC_PCT"] = (1.0 - agg["WER"]) * 100
    agg["INS_PCT"] = (I / N * 100) if N > 0 else 0.0
    agg["DEL_PCT"] = (D / N * 100) if N > 0 else 0.0
    agg["SUB_PCT"] = (S / N * 100) if N > 0 else 0.0

    return agg


def aggregate_clips(clip_list: list[dict]) -> dict:
    """
    Merge a list of per-clip dicts into one aggregated dict.
    - Integer counts are summed.
    - Rates (WER, MER, WIP, WIL, percentages) are RECOMPUTED from summed counts.
    - CER is averaged (unavoidable approximation — no char counts in source files).
    """
    if len(clip_list) == 1:
        return clip_list[0].copy()

    agg = {k: 0 for k in INT_KEYS}
    for clip in clip_list:
        for k in INT_KEYS:
            agg[k] += clip.get(k, 0)

    cer_vals = [c["CER"] for c in clip_list if "CER" in c]
    if cer_vals:
        agg["CER"] = sum(cer_vals) / len(cer_vals)

    return recompute_rates(agg)


# ──────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────
FLOAT_PATTERNS = {
    "WER": r"Word Error Rate \(WER\):\s+([\d.]+)",
    "MER": r"Match Error Rate \(MER\):\s+([\d.]+)",
    "WIL": r"Word Information Lost \(WIL\):\s+([\d.]+)",
    "WIP": r"Word Information Preserved \(WIP\):\s+([\d.]+)",
    "CER": r"Character Error Rate \(CER\):\s+([\d.]+)",
}
INT_PATTERNS = {
    "ref_words_count": r"Total Reference Words:\s+(\d+)",
    "hits":            r"Hits \(Correct\):\s+(\d+)",
    "substitutions":   r"Substitutions.*?Total:\s+(\d+)",
    "deletions":       r"Deletions.*?Total:\s+(\d+)",
    "insertions":      r"Insertions.*?Total:\s+(\d+)",
}


def parse_evaluation_file(filepath: str) -> dict | None:
    """Parse one *_evaluation.txt and return raw counts + CER."""
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [WARN] Could not read {filepath}: {e}")
        return None

    data = {}
    for key, pat in FLOAT_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            data[key] = float(m.group(1))
    for key, pat in INT_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            data[key] = int(m.group(1))

    if not data:
        return None

    # Derive percentage fields from raw counts so single-clip path
    # is consistent with multi-clip path.
    N   = data.get("ref_words_count", 0)
    wer = data.get("WER", 0.0)
    data["ACC_PCT"] = (1.0 - wer) * 100
    data["INS_PCT"] = (data.get("insertions",    0) / N * 100) if N > 0 else 0.0
    data["DEL_PCT"] = (data.get("deletions",     0) / N * 100) if N > 0 else 0.0
    data["SUB_PCT"] = (data.get("substitutions", 0) / N * 100) if N > 0 else 0.0
    return data


# ──────────────────────────────────────────────────────────────
# DATA COLLECTION
# ──────────────────────────────────────────────────────────────
def collect_all_data(pipelines: list[tuple]) -> tuple:
    """
    Walk each pipeline directory, parse all evaluation files, aggregate
    clips per (recording, pipeline, speaker).

    OVERALL is built from each clip's `*_overall_evaluation.txt` (session-level
    ASR WER), pooled across clips by summing raw counts. Per-speaker rows are
    aggregated independently from their own files. COMBINED files are ignored.

    Returns
    -------
    all_recordings : sorted list of recording names
    all_speakers   : ordered list including OVERALL
    results        : results[recording][pipeline][speaker] = aggregated dict
    clip_counts    : clip_counts[recording][pipeline][speaker] = int
    """
    raw: dict          = {}  # raw[rec][pipeline][speaker] = [clip_dict, ...]
    overall_raw: dict  = {}  # overall_raw[rec][pipeline]  = [clip_dict, ...]
    unique_clips: dict = {}  # unique_clips[rec][pipeline] = set of clip_ids

    for pipeline_name, base_path in pipelines:
        base = Path(base_path)
        if not base.exists():
            print(f"  [WARN] Directory not found: {base}")
            continue

        for recording_dir in sorted(base.iterdir()):
            if not recording_dir.is_dir():
                continue

            speaker_files = sorted(recording_dir.rglob("*_evaluation.txt"))
            if not speaker_files:
                print(f"  [WARN] No evaluation files under {recording_dir}")
                continue

            rec = recording_dir.name

            for spk_file in speaker_files:
                parts   = spk_file.stem.split("_")
                spk_raw = parts[-2].lower()
                speaker = SPEAKER_MAP.get(spk_raw, spk_raw.upper())

                # Everything before the last two parts (_<speaker>_evaluation)
                # identifies the unique audio segment.
                clip_id = "_".join(parts[:-2])

                parsed = parse_evaluation_file(str(spk_file))
                if not parsed:
                    continue

                if speaker == COMBINED_LABEL:
                    # Ignore COMBINED — OVERALL is our session-level metric.
                    continue

                if speaker == OVERALL_LABEL:
                    # Session-level ASR WER for this clip — pooled later.
                    (overall_raw
                     .setdefault(rec, {})
                     .setdefault(pipeline_name, [])
                     .append(parsed))
                    (unique_clips
                     .setdefault(rec, {})
                     .setdefault(pipeline_name, set())
                     .add(clip_id))
                    print(f"   {pipeline_name} | {rec} | OVERALL | {spk_file.name}")
                    continue

                # Per-speaker file (FEM, MAL, KCHI, OCH, ADULT, CHILDREN, ...).
                (raw
                 .setdefault(rec, {})
                 .setdefault(pipeline_name, {})
                 .setdefault(speaker, [])
                 .append(parsed))
                (unique_clips
                 .setdefault(rec, {})
                 .setdefault(pipeline_name, set())
                 .add(clip_id))
                print(f"   {pipeline_name} | {rec} | {speaker} | {spk_file.name}")

    # ── Aggregate per-speaker clips ───────────────────────────
    results:      dict = {}
    clip_counts:  dict = {}
    all_speakers: set  = set()

    for rec, pipes in raw.items():
        for pipe, speakers in pipes.items():
            for spk, clip_list in speakers.items():
                agg = aggregate_clips(clip_list)
                results.setdefault(rec, {}).setdefault(pipe, {})[spk] = agg
                clip_counts.setdefault(rec, {}).setdefault(pipe, {})[spk] = len(clip_list)
                all_speakers.add(spk)

    # ── Pool OVERALL across clips from *_overall_evaluation.txt ──
    for rec, pipes in overall_raw.items():
        for pipe, clip_list in pipes.items():
            agg = aggregate_clips(clip_list)
            results.setdefault(rec, {}).setdefault(pipe, {})[OVERALL_LABEL] = agg
            clip_counts.setdefault(rec, {}).setdefault(pipe, {})[OVERALL_LABEL] = len(
                unique_clips.get(rec, {}).get(pipe, set())
            )
            all_speakers.add(OVERALL_LABEL)

    # ── Order speakers ─────────────────────────────────────────
    ordered = [s for s in SPEAKER_ORDER if s in all_speakers]
    for s in sorted(all_speakers):
        if s not in ordered:
            ordered.append(s)

    all_recs = sorted(set(raw.keys()) | set(overall_raw.keys()))
    return all_recs, ordered, results, clip_counts


# ──────────────────────────────────────────────────────────────
# STYLE HELPERS
# ──────────────────────────────────────────────────────────────
def _font(color="FFFFFF", sz=11, bold=True):
    return Font(name="Arial", bold=bold, color=color, size=sz)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _center(wrap=True):
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)

def _left():
    return Alignment(horizontal="left", vertical="center")

def _thin_border():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def _sep_border():
    return Border(
        left=Side(style="thin",    color="AAAAAA"),
        right=Side(style="medium", color="444444"),
        top=Side(style="thin",     color="AAAAAA"),
        bottom=Side(style="thin",  color="AAAAAA"),
    )


def write_value(cell, value, is_rate: bool, m_key: str):
    """Write a metric value with correct number format."""
    if value is None:
        cell.value = "N/A"
        cell.font  = Font(name="Arial", color="999999", size=10)
        return
    if is_rate:
        cell.value         = value        # stored as 0–1 fraction
        cell.number_format = "0.00%"
    elif m_key in ("INS_PCT", "DEL_PCT", "SUB_PCT", "ACC_PCT"):
        cell.value         = value / 100  # convert to 0–1 for % format
        cell.number_format = "0.00%"
    else:
        cell.value         = value
        cell.number_format = "#,##0"
    cell.font = Font(name="Arial", size=10)


# ──────────────────────────────────────────────────────────────
# EXCEL BUILDER
# ──────────────────────────────────────────────────────────────
def _write_three_row_header(ws, pipeline_names, metrics, row_offset=1,
                             col_offset=2, label_cols=None):
    """
    Write the 3-row header block (title, pipeline groups, metric sub-headers)
    used by both per-speaker sheets and the Summary sheet.

    label_cols : list of (col_index, label) for fixed label columns before metrics
    """
    pass  # inlined below for clarity


def build_per_speaker_sheet(ws, speaker, all_recordings, pipeline_names,
                             results, clip_counts):
    n_m   = len(METRICS)
    n_p   = len(pipeline_names)
    total = 2 + n_p * n_m   # Recording + Clips + metric columns

    # Row 1 – title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total)
    c = ws.cell(1, 1, f"Transcription Pipeline Comparison — Speaker: {speaker}")
    c.font = _font(sz=13); c.fill = _fill(HDR_DARK); c.alignment = _center()

    # Row 2 – fixed headers + pipeline group headers
    for col, lbl in [(1, "Recording"), (2, "Clips")]:
        c = ws.cell(2, col, lbl)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _center()

    for p_idx, p_name in enumerate(pipeline_names):
        cs = 3 + p_idx * n_m
        ws.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=cs + n_m - 1)
        c = ws.cell(2, cs, p_name)
        c.font = _font()
        c.fill = _fill(HDR_DARK if p_idx % 2 == 0 else HDR_MED)
        c.alignment = _center()

    # Row 3 – metric sub-headers
    for col, lbl in [(1, "Recording"), (2, "Clips")]:
        c = ws.cell(3, col, lbl)
        c.font = _font("000000", sz=9); c.fill = _fill(HDR_LIGHT)
        c.alignment = _center(); c.border = _sep_border()

    for p_idx in range(n_p):
        for m_idx, (m_label, _, _) in enumerate(METRICS):
            col = 3 + p_idx * n_m + m_idx
            c = ws.cell(3, col, m_label)
            c.font = Font(name="Arial", bold=True, size=9)
            c.fill = _fill(PIPE_COLS[p_idx % len(PIPE_COLS)])
            c.alignment = _center(); c.border = _thin_border()

    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 20
    ws.row_dimensions[3].height = 32

    # Data rows
    for r_idx, rec in enumerate(all_recordings):
        row = 4 + r_idx
        bg  = WHITE if r_idx % 2 == 0 else ALT_ROW

        c = ws.cell(row, 1, rec)
        c.font = Font(name="Arial", bold=True, size=10)
        c.fill = _fill("E8EEF7"); c.alignment = _left(); c.border = _sep_border()

        # Use the OVERALL clip count (total clips for this recording + pipeline)
        first_pipe = pipeline_names[0]
        n_clips = clip_counts.get(rec, {}).get(first_pipe, {}).get(speaker, "?")
        c = ws.cell(row, 2, n_clips)
        c.font = Font(name="Arial", size=10, color="555555")
        c.fill = _fill("EEF2FF"); c.alignment = _center(wrap=False); c.border = _thin_border()

        for p_idx, p_name in enumerate(pipeline_names):
            spk_data = results.get(rec, {}).get(p_name, {}).get(speaker, {})
            for m_idx, (_, m_key, is_rate) in enumerate(METRICS):
                col = 3 + p_idx * n_m + m_idx
                c   = ws.cell(row, col)
                c.fill = _fill(bg); c.alignment = _center(wrap=False); c.border = _thin_border()
                write_value(c, spk_data.get(m_key), is_rate, m_key)

    # Footnote row
    fn_row = 4 + len(all_recordings) + 1
    ws.merge_cells(start_row=fn_row, start_column=1, end_row=fn_row, end_column=total)
    c = ws.cell(fn_row, 1, FOOTNOTE)
    c.font = Font(name="Arial", italic=True, size=8, color="666666")
    c.alignment = _left()

    # Column widths
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 8
    for p_idx in range(n_p):
        for m_idx in range(n_m):
            ws.column_dimensions[get_column_letter(3 + p_idx * n_m + m_idx)].width = 13


def build_summary_sheet(ws, all_recordings, all_speakers, pipeline_names,
                         results, clip_counts):
    n_k    = len(KEY_METRICS)
    n_s    = len(all_speakers)
    total  = 3 + n_s * n_k   # Recording + Clips + Pipeline + speaker metric columns

    # Row 1 – title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total)
    c = ws.cell(1, 1, "Pipeline Comparison Summary — All Speakers")
    c.font = _font(sz=12); c.fill = _fill(HDR_DARK); c.alignment = _center()

    # Row 2 – fixed headers + speaker group headers
    for col, lbl in [(1, "Recording"), (2, "Clips"), (3, "Pipeline")]:
        c = ws.cell(2, col, lbl)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _center()

    for s_idx, spk in enumerate(all_speakers):
        cs = 4 + s_idx * n_k
        ws.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=cs + n_k - 1)
        c = ws.cell(2, cs, spk)
        c.font = _font(); c.fill = _fill(SPK_COLS[s_idx % len(SPK_COLS)])
        c.alignment = _center()

    # Row 3 – metric sub-headers
    for col, lbl in [(1, "Recording"), (2, "Clips"), (3, "Pipeline")]:
        c = ws.cell(3, col, lbl)
        c.font = _font("000000", sz=9); c.fill = _fill(HDR_LIGHT)
        c.alignment = _center(); c.border = _sep_border()

    for s_idx in range(n_s):
        for m_idx, (m_label, _, _) in enumerate(KEY_METRICS):
            col = 4 + s_idx * n_k + m_idx
            c = ws.cell(3, col, m_label)
            c.font = Font(name="Arial", bold=True, size=9)
            c.fill = _fill(PIPE_COLS[s_idx % len(PIPE_COLS)])
            c.alignment = _center(); c.border = _thin_border()

    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 20
    ws.row_dimensions[3].height = 32

    row = 4
    for rec in all_recordings:
        for p_idx, p_name in enumerate(pipeline_names):
            bg = "F0F4FF" if p_idx % 2 == 0 else WHITE

            # Recording cell (show only on first pipeline row)
            ca = ws.cell(row, 1, rec if p_idx == 0 else "")
            ca.font = Font(name="Arial", bold=(p_idx == 0), size=10)
            ca.fill = _fill("E8EEF7" if p_idx == 0 else "F5F5F5")
            ca.alignment = _left(); ca.border = _sep_border()

            # Clip count for OVERALL for this pipeline
            n_clips = clip_counts.get(rec, {}).get(p_name, {}).get("OVERALL", "?")
            cb = ws.cell(row, 2, n_clips if p_idx == 0 else "")
            cb.font = Font(name="Arial", size=10, color="555555")
            cb.fill = _fill("EEF2FF" if p_idx == 0 else "F5F5F5")
            cb.alignment = _center(wrap=False); cb.border = _thin_border()

            cc = ws.cell(row, 3, p_name)
            cc.font = Font(name="Arial", size=10)
            cc.fill = _fill(bg); cc.alignment = _left(); cc.border = _sep_border()

            for s_idx, spk in enumerate(all_speakers):
                spk_data = results.get(rec, {}).get(p_name, {}).get(spk, {})
                for m_idx, (_, m_key, is_rate) in enumerate(KEY_METRICS):
                    col = 4 + s_idx * n_k + m_idx
                    c   = ws.cell(row, col)
                    c.fill = _fill(bg); c.alignment = _center(wrap=False); c.border = _thin_border()
                    write_value(c, spk_data.get(m_key), is_rate, m_key)

            row += 1
        row += 1  # blank separator row between recordings

    # Footnote
    fn_row = row + 1
    ws.merge_cells(start_row=fn_row, start_column=1, end_row=fn_row, end_column=total)
    c = ws.cell(fn_row, 1, FOOTNOTE)
    c.font = Font(name="Arial", italic=True, size=8, color="666666")
    c.alignment = _left()

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 26
    for s_idx in range(n_s):
        for m_idx in range(n_k):
            ws.column_dimensions[get_column_letter(4 + s_idx * n_k + m_idx)].width = 13


def build_excel(all_recordings, all_speakers, results, clip_counts,
                pipelines, output_file):
    wb = Workbook()
    pipeline_names = [p[0] for p in pipelines]

    # Summary sheet first (index=0)
    ws_summary = wb.active
    ws_summary.title = "Summary"
    build_summary_sheet(ws_summary, all_recordings, all_speakers,
                        pipeline_names, results, clip_counts)

    # One sheet per speaker
    for speaker in all_speakers:
        ws = wb.create_sheet(title=speaker)
        build_per_speaker_sheet(ws, speaker, all_recordings, pipeline_names,
                                results, clip_counts)

    wb.save(output_file)
    print(f"\n Excel saved → {output_file}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
def main():
    print("Collecting evaluation data...")
    print("=" * 60)
    all_recordings, all_speakers, results, clip_counts = collect_all_data(PIPELINES)

    if not results:
        print("\nERROR: No data found. Check your PIPELINES paths in the CONFIG section.")
        return

    print(f"\nFound {len(all_recordings)} recording(s): {sorted(all_recordings)}")
    print(f"Speakers detected: {all_speakers}")

    print("\nBuilding Excel workbook...")
    build_excel(all_recordings, all_speakers, results, clip_counts, PIPELINES, OUTPUT_FILE)


if __name__ == "__main__":
    main()
