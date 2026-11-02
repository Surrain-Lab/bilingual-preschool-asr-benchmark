#!/usr/bin/env python3
"""
Macro-averaged WER per pipeline (key child, adult, overall) from a stage-1
output folder, i.e. data/derived/supp_wer_by_pipeline.csv.

Errors and reference words are summed within each recording, and the 14
recording-level rates are averaged.

    python stage2_aggregate/recompute_supp_wer.py <stage1_output_dir> [--out FILE]

<stage1_output_dir> holds one folder per pipeline, each with an eval_report/
tree. For the two post-hoc pipelines, use the time-ordered evaluation
(see posthoc_vtc2/README.md), otherwise the overall column comes out too high.
"""
import argparse, collections, csv, glob, os, re

PIPES = [("human_timestamps",      "Human-TS + WhisperX (control)"),
         ("vtc1",                  "VTC1 + WhisperX"),
         ("vtc2",                  "VTC2 + WhisperX"),
         ("vtc2_nolang_barrier",   "VTC2-NoLang + WhisperX"),
         ("whisperx_only",         "WhisperX-VAD + VTC1 (post-hoc)"),
         ("whisperx_vtc2_posthoc", "WhisperX-VAD + VTC2 (post-hoc)")]
STRATA = ("kchi", "adult", "overall")
HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED = os.path.join(HERE, "..", "data", "derived", "supp_wer_by_pipeline.csv")


def count(text, key):
    m = re.search(key + r":\s*(\d+)", text)
    return int(m.group(1)) if m else 0


def macro_wer(eval_dir, stratum):
    rec = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(os.path.join(eval_dir, "*", "*", f"*_{stratum}_evaluation.txt")):
        s = open(f, encoding="utf-8", errors="replace").read()
        n = count(s, "Total Reference Words")
        e = count(s, "Substitutions") + count(s, "Deletions") + count(s, "Insertions")
        r = os.path.relpath(f, eval_dir).split(os.sep)[0]
        rec[r][0] += e
        rec[r][1] += n
    rates = [e / n for e, n in rec.values() if n > 0]
    return sum(rates) / len(rates) if rates else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="supp_wer_by_pipeline.csv")
    a = ap.parse_args()

    shipped = {r["pipeline"]: r for r in csv.DictReader(open(SHIPPED, encoding="utf-8"))}
    rows, all_ok = [], True
    print("%-34s%10s%10s%10s" % ("pipeline", *STRATA))
    for d, label in PIPES:
        ev = os.path.join(a.root, d, "eval_report")
        if not os.path.isdir(ev):
            print("%-34s  (missing %s)" % (label, ev))
            continue
        got = [macro_wer(ev, s) for s in STRATA]
        ref = shipped.get(label)
        cells = []
        for s, g in zip(STRATA, got):
            ok = ref is None or abs(g - float(ref[f"{s}_macro_wer"])) <= 0.0005
            all_ok &= ok
            cells.append("%.4f%s" % (g, "" if ok else "*"))
        print("%-34s%10s%10s%10s" % (label, *cells))
        rows.append([label] + ["%.4f" % g for g in got])

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipeline"] + [f"{s}_macro_wer" for s in STRATA])
        w.writerows(rows)
    print("\nwrote", a.out)
    print("all values match data/derived/supp_wer_by_pipeline.csv" if all_ok
          else "values marked * differ from data/derived/supp_wer_by_pipeline.csv")


if __name__ == "__main__":
    main()
