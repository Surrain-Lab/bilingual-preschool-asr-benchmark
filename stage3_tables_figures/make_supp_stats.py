#!/usr/bin/env python3
"""
Supplementary inferential statistics: bootstrap CIs (tab:supp-wer) and the
pairwise bootstrap tests (tab:supp-pairwise-*).

Procedure:
  * CIs          - numpy default_rng(42), 5,000 resamples of the 14 recording
                   WERs, percentile interval.
  * pairwise     - numpy default_rng(42) re-seeded per pair, 10,000 resamples,
                   deltas centred on the observed difference, two-tailed.
Resampling is over recordings, not clips (Methods, sec:methods-stats).

Writes data/derived/supp_bootstrap.csv and supp_pairwise.csv.
"""
import os, re, csv, argparse
import numpy as np, openpyxl

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
XL   = os.path.join(ROOT, "data", "derived", "wer_per_recording_14clips.xlsx")
OUT  = os.path.join(ROOT, "data", "derived")

PIPES = ["Human-TS + Whisper", "VTC1 + Whisper", "VTC2 + Whisper",
         "VTC2-NoLang + Whisper", "Only Whisper", "WhisperX-VAD + VTC2"]
SHORT = {"Human-TS + Whisper": "HumanTS", "VTC1 + Whisper": "VTC1",
         "VTC2 + Whisper": "VTC2", "VTC2-NoLang + Whisper": "VTC2-NL",
         "Only Whisper": "WhisperX", "WhisperX-VAD + VTC2": "WhisperX-VTC2"}
SHEETS = ("KCHI", "ADULT", "OVERALL")

def load(sheet):
    rows = list(openpyxl.load_workbook(XL, read_only=True, data_only=True)[sheet]
                .iter_rows(values_only=True))
    blocks = {c: i for i, c in enumerate(rows[1]) if c in PIPES}
    hdr = rows[2]; out = {}
    for p, c0 in blocks.items():
        wc = next(j for j in range(c0, c0 + 14) if hdr[j] == "WER")
        v = {}
        for r in rows[3:]:
            # a recording row has an id in col 0 and a numeric WER; matching on an
            # id prefix would break once identifiers are anonymised
            if not r or not isinstance(r[0], str) or not r[0].strip(): continue
            x = r[wc]
            if isinstance(x, (int, float)): v[r[0]] = float(x)
            elif isinstance(x, str) and re.match(r"^[\d.]+$", x): v[r[0]] = float(x)
        if v: out[p] = v
    return out

def bootstrap_ci(values, n_boot=5000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    v = np.array(values, float); v = v[~np.isnan(v)]
    if len(v) == 0: return np.nan, np.nan, np.nan
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return (float(v.mean()), float(np.percentile(means, (100-ci)/2)),
            float(np.percentile(means, 100-(100-ci)/2)))

def paired_p(a, b, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array(a, float); b = np.array(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]; n = len(a)
    if n < 3: return np.nan, np.nan, n
    d0 = a.mean() - b.mean()
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ds[i] = a[idx].mean() - b[idx].mean() - d0
    return float(d0), float(np.mean(np.abs(ds) >= abs(d0))), n

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quiet", action="store_true"); a = ap.parse_args()
    ci_rows, pw_rows = [], []
    for sheet in SHEETS:
        d = load(sheet)
        for p in PIPES:
            m, lo, hi = bootstrap_ci(list(d[p].values()))
            ci_rows.append([sheet, SHORT[p], f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(d[p])])
        for i, A in enumerate(PIPES):
            for B in PIPES[i+1:]:
                recs = sorted(set(d[A]) | set(d[B]))
                delta, pv, n = paired_p([d[A].get(r, np.nan) for r in recs],
                                        [d[B].get(r, np.nan) for r in recs])
                pw_rows.append([sheet, SHORT[A], SHORT[B], f"{delta:.4f}", f"{pv:.4f}", stars(pv), n])
    for fn, hdr, rows in (("supp_bootstrap.csv", ["stratum","pipeline","mean","ci_lo","ci_hi","n"], ci_rows),
                          ("supp_pairwise.csv", ["stratum","pipe_a","pipe_b","delta","p","stars","n"], pw_rows)):
        p = os.path.join(OUT, fn)
        w = csv.writer(open(p, "w", newline="", encoding="utf-8")); w.writerow(hdr); w.writerows(rows)
        if not a.quiet: print(f"   wrote {p}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
