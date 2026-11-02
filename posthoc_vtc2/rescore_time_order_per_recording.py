#!/usr/bin/env python3
"""
Pool overall-WER evaluation reports per recording, for one or more eval_report folders.

    python rescore_time_order_per_recording.py LABEL=EVAL_REPORT_DIR [LABEL=DIR ...] > pooled.csv

Prints one CSV row per recording: clips, reference words (N), hits, substitutions,
deletions, insertions, hypothesis words (P), and the mean CER / WIL / WIP / MER.
"""
import collections, glob, os, re, sys


def g(s, k):
    m = re.search(k + r":\s*([\d.]+)", s)
    return float(m.group(1)) if m else None


if len(sys.argv) < 2:
    sys.exit(__doc__)
C = dict(a.split("=", 1) for a in sys.argv[1:])

print("cond,rec,clips,N,H,S,D,I,P,cer_mean,wil_mean,wip_mean,mer_mean")
for lab, d in C.items():
    rec = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, []])
    for f in glob.glob(os.path.join(d, "*", "*", "*_overall_evaluation.txt")):
        s = open(f, encoding="utf-8", errors="replace").read()
        N = g(s, "Total Reference Words")
        if not N:
            continue
        a = rec[os.path.relpath(f, d).split(os.sep)[0].replace("_pre_1", "_pre")]
        a[0] += 1; a[1] += N; a[2] += g(s, r"Hits \(Correct\)")
        a[3] += g(s, "Substitutions"); a[4] += g(s, "Deletions"); a[5] += g(s, "Insertions")
        a[6] += g(s, "Total Hypothesis Words")
        a[7].append((g(s, r"Character Error Rate \(CER\)"), g(s, r"Word Information Lost \(WIL\)"),
                     g(s, r"Word Information Preserved \(WIP\)"), g(s, r"Match Error Rate \(MER\)")))
    for r, a in sorted(rec.items()):
        print(lab, r, *[int(x) for x in a[:7]],
              *[sum(t[i] for t in a[7]) / len(a[7]) for i in range(4)], sep=",")
