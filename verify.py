#!/usr/bin/env python3
"""
Checks that this repo reproduces the numbers printed in the paper.

Regenerates every quantity it can from data/derived and compares it with
expected/paper_values.json. Exits non-zero on any mismatch.

    python verify.py
"""
import csv, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D    = os.path.join(HERE, "data", "derived")
EXP  = json.load(open(os.path.join(HERE, "expected", "paper_values.json"), encoding="utf-8"))

ok_all = True
def check(label, got, want, tol):
    global ok_all
    good = got is not None and abs(got - want) <= tol
    ok_all &= good
    print(f"  {'PASS' if good else 'FAIL'}  {label:<44} got {got!r:>10}  want {want!r}")
    return good

def fit(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((a-mx)**2 for a in xs); syy = sum((b-my)**2 for b in ys)
    sxy = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    return sxy/math.sqrt(sxx*syy), sxy/sxx, n

# ---------- 1. total words (from the word-count workbook, all 168 clips) ----------
def total_words():
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(D, "word_count_comparison_14clips.xlsx"),
                                read_only=True, data_only=True)
    clip_re = re.compile(r"_\d+$")
    num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
    out = {}
    for sheet, key in (("ADULT", "adult"), ("KCHI", "kchild")):
        rows = [r for r in list(wb[sheet].iter_rows(values_only=True))[3:]
                if r and isinstance(r[1], str) and clip_re.search(r[1])]
        for lang, (hc, ac) in {"en": (3, 8), "es": (4, 9)}.items():
            p = [(num(r[hc]), num(r[ac])) for r in rows if num(r[hc]) > 0]
            out[f"{key}_{lang}"] = fit([a for a, _ in p], [b for _, b in p])
    return out

# ---------- 2. distinct words and TTR (from the per-speaker vocab csv) ----------
def vocab(measure):
    rows = list(csv.DictReader(open(os.path.join(D, "vocab_by_speaker.csv"), encoding="utf-8-sig")))
    out = {}
    for spk, key in (("adult", "adult"), ("kchild", "kchild")):
        for lang in ("en", "es"):
            xs, ys = [], []
            for r in rows:
                if r["speaker"] != spk: continue
                ht, at = float(r[f"human_{lang}_tokens"]), float(r[f"vtc2_{lang}_tokens"])
                hy, ay = float(r[f"human_{lang}_types"]),  float(r[f"vtc2_{lang}_types"])
                if measure == "types":
                    if hy > 0: xs.append(hy); ys.append(ay)
                else:
                    if ht > 0 and at > 0: xs.append(hy/ht); ys.append(ay/at)
            out[f"{key}_{lang}"] = fit(xs, ys)
    return out

# ---------- 3. inter-coder WER ----------
def irr():
    rows = [r for r in csv.DictReader(open(os.path.join(D, "irr_symmetric_wer.csv"), encoding="utf-8-sig"))
            if not r.get("degenerate")]
    out = {}
    for stratum, key in (("KCHI", "kchi"), ("ADULT", "adult"), ("OVERALL", "overall")):
        rr = [r for r in rows if r["stratum"] == stratum]
        if not rr: continue
        den = sum((float(r["n_words_a"]) + float(r["n_words_b"]))/2 for r in rr)
        S, Dl, I = (sum(float(r[k]) for r in rr) for k in ("sub", "del", "ins"))
        out[key] = ((S+Dl+I)/den, S/den, Dl/den, I/den)
    return out

# ---------- 4. Krippendorff alpha ----------
def alpha():
    out = {}
    for fn in ("krippendorff_alpha.csv", "language_alpha.csv"):
        p = os.path.join(D, "irr_alpha", fn)
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["clip_id"] == "POOLED":
                out[r["layer"]] = float(r["alpha"])
    return out

print("\n=== total words: automated vs manual ===")
tw = total_words()
for k, v in EXP["agreement_total_words"].items():
    r, s, n = tw[k]
    check(f"{k} r", round(r, 2), v["r"], 1e-9)
    check(f"{k} slope", round(s, 2), v["slope"], 1e-9)
    check(f"{k} n", n, v["n"], 0)

print("\n=== distinct words ===")
dv = vocab("types")
for k, v in EXP["agreement_distinct_words"].items():
    r, s, n = dv[k]
    check(f"{k} r", round(r, 2), v["r"], 1e-9)
    check(f"{k} slope", round(s, 2), v["slope"], 1e-9)
    check(f"{k} n", n, v["n"], 0)

print("\n=== type-token ratio ===")
tv = vocab("ttr")
for k, v in EXP["agreement_type_token_ratio"].items():
    r, s, n = tv[k]
    check(f"{k} r", round(r, 2), v["r"], 1e-9)
    check(f"{k} slope", round(s, 2), v["slope"], 1e-9)
    check(f"{k} n", n, v["n"], 0)

print("\n=== inter-coder agreement ===")
ir = irr()
for k, want in EXP["irr_symmetric_wer"].items():
    check(f"symmetric WER {k}", round(ir[k][0], 3), want, 1e-9)
comp = EXP["irr_kchi_composition"]
check("kchi substitution rate", round(ir["kchi"][1], 3), comp["sub"], 1e-9)
check("kchi deletion rate",     round(ir["kchi"][2], 3), comp["del"], 1e-9)
check("kchi insertion rate",    round(ir["kchi"][3], 3), comp["ins"], 1e-9)

print("\n=== Krippendorff alpha (pooled) ===")
al = alpha()
for k, want in EXP["krippendorff_alpha_pooled"].items():
    check(f"alpha {k}", al.get(k), want, 5e-4)

# ---------- 5. supplementary: WER by pipeline, segmentation gap, corr figure ----------
print("\n=== supplementary: macro WER by pipeline (tab:supp-wer) ===")
sw = {r["pipeline"]: r for r in csv.DictReader(open(os.path.join(D, "supp_wer_by_pipeline.csv"), encoding="utf-8-sig"))}
for pipe, want in EXP["supp_wer_macro"].items():
    row = sw.get(pipe)
    for strat in ("kchi", "adult", "overall"):
        # compare unrounded: 0.7155 and 0.7215 are exact ties at 3 dp, where the
        # manuscript rounds down and Python's round() goes the other way
        got = float(row[f"{strat}_macro_wer"]) if row else None
        check(f"{pipe[:26]} {strat}", got, want[strat], 5.1e-4)

print("\n=== supplementary: segmentation gap (tab:supp-seg-gap) ===")
ctrl = sw["Human-TS + WhisperX (control)"]
for pipe, want in EXP["supp_seg_gap"].items():
    if pipe.startswith("_"): continue
    for strat in ("kchi", "adult", "overall"):
        gap = float(sw[pipe][f"{strat}_macro_wer"]) - float(ctrl[f"{strat}_macro_wer"])
        check(f"gap {pipe[:22]} {strat}", gap, want[strat], 5.1e-4)

print("\n=== supplementary: recording-level correlations (fig:corrmodel) ===")
try:
    import pandas as pd, numpy as np
    sys.path.insert(0, os.path.join(HERE, "stage3_tables_figures"))
    from make_fig_corr_by_model import load as _corr_load, MODELS as _M
    df = _corr_load()
    for m in [x for x in _M if x != "Human"]:
        for lg in ("en", "es"):
            got = round(float(np.corrcoef(df[f"Human_{lg}"], df[f"{m}_{lg}"])[0, 1]), 2)
            check(f"corr {m} {lg}", got, EXP["supp_corr_by_model"][f"{m}_{lg}"], 1e-9)
except Exception as e:
    print(f"  SKIP  correlation checks ({e})")
    ok_all = False

# ---------- 6. supplementary: bootstrap CIs and pairwise tests ----------
def _regen_supp_stats():
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "stage3_tables_figures", "make_supp_stats.py"),
                    "--quiet"], check=True)

print("\n=== supplementary: bootstrap CIs (tab:supp-wer) ===")
_regen_supp_stats()
ci = {f'{r["stratum"]}|{r["pipeline"]}': r
      for r in csv.DictReader(open(os.path.join(D, "supp_bootstrap.csv"), encoding="utf-8-sig"))}
for key, (m, lo, hi) in EXP["supp_bootstrap_ci"].items():
    row = ci[key]
    check(f"{key} mean",  float(row["mean"]),  m,  5.1e-4)
    check(f"{key} CI lo", float(row["ci_lo"]), lo, 5.1e-4)
    check(f"{key} CI hi", float(row["ci_hi"]), hi, 5.1e-4)

print("\n=== supplementary: pairwise bootstrap tests (tab:supp-pairwise-*) ===")
pw = {f'{r["stratum"]}|{r["pipe_a"]}|{r["pipe_b"]}': r
      for r in csv.DictReader(open(os.path.join(D, "supp_pairwise.csv"), encoding="utf-8-sig"))}
for key, (delta, st) in EXP["supp_pairwise"].items():
    row = pw[key]
    check(f"{key} delta", float(row["delta"]), delta, 5.1e-4)
    got_st = row["stars"] or "ns"
    good = got_st == (st or "ns")
    ok_all &= good
    print(f"  {'PASS' if good else 'FAIL'}  {key+' sig':<44} got {got_st!r:>10}  want {(st or 'ns')!r}")

# ---------- 7. further values: agreement, type-token ratios, failure analyses ----------
# Shipped inputs only:
# vocab_by_speaker.csv, the two workbooks, irr_symmetric_wer.csv, irr_alpha/, irr_icc.csv, and
# failure_analysis/ (written by stage2_aggregate/failure_analysis/failure_analysis.py from stage-1 output).
C = EXP["additional"]
def _num(v): return float(v) if isinstance(v, (int, float)) else None

print("\n=== IRR ===")
icc = {r["measure"]: r for r in csv.DictReader(open(os.path.join(D, "irr_icc.csv"), encoding="utf-8-sig"))}
for key, meas in (("adult", "adult_wc"), ("kchild", "child_wc")):
    v, lo, hi = C["icc"][key]
    check(f"ICC {key}", float(icc[meas]["icc"]), v, 5e-3)
    check(f"ICC {key} CI lo", float(icc[meas]["ci_lower"]), lo, 5e-3)
    check(f"ICC {key} CI hi", float(icc[meas]["ci_upper"]), hi, 5e-3)
sw = [r for r in csv.DictReader(open(os.path.join(D, "irr_symmetric_wer.csv"), encoding="utf-8-sig")) if r["sym_wer"]]
def _split(stratum):
    rs = [r for r in sw if r["stratum"] == stratum]
    n = sum((int(r["n_words_a"]) + int(r["n_words_b"])) / 2 for r in rs)
    return (sum(int(r["sub"]) for r in rs) / n, sum(int(r["del"]) for r in rs) / n,
            sum(int(r["sub"]) + int(r["del"]) + int(r["ins"]) for r in rs) / n)
ks, kd, _ = _split("KCHI"); as_, ad, _ = _split("ADULT"); _, _, och = _split("OCH")
for lab, got in (("kchi_sub", ks), ("kchi_del", kd), ("adult_sub", as_), ("adult_del", ad), ("och_total", och)):
    check(f"inter-coder WER {lab}", got, C["irr_wer_split"][lab], 5e-4)
ka = {(r["layer"], r["clip_id"]): r for r in csv.DictReader(open(os.path.join(D, "irr_alpha", "krippendorff_alpha.csv"), encoding="utf-8-sig"))}
r = ka[("speaker_coarse", "POOLED")]
for lab, got, want in (("", r["alpha"], C["speaker_coarse_alpha"][0]), (" CI lo", r["ci_lower"], C["speaker_coarse_alpha"][1]),
                       (" CI hi", r["ci_upper"], C["speaker_coarse_alpha"][2])):
    check("alpha speaker (adults combined)" + lab, float(got), want, 5e-3)

print("\n=== type-token ratio means ===")
vrows = list(csv.DictReader(open(os.path.join(D, "vocab_by_speaker.csv"), encoding="utf-8-sig")))
for key, (hm, am) in C["ttr_means_manual_auto"].items():
    spk, lang = key.split("_")
    T = [r for r in vrows if r["speaker"] == spk and int(r[f"human_{lang}_tokens"]) > 0 and int(r[f"vtc2_{lang}_tokens"]) > 0]
    check(f"TTR {key} manual", sum(int(r[f"human_{lang}_types"]) / int(r[f"human_{lang}_tokens"]) for r in T) / len(T), hm, 5e-3)
    check(f"TTR {key} automated", sum(int(r[f"vtc2_{lang}_types"]) / int(r[f"vtc2_{lang}_tokens"]) for r in T) / len(T), am, 5e-3)

print("\n=== word-count workbook ===")
import openpyxl
_wb = openpyxl.load_workbook(os.path.join(D, "word_count_comparison_14clips.xlsx"), read_only=True, data_only=True)
_num0 = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
for sheet, key in (("ADULT", "adult"), ("KCHI", "kchild")):
    rows = [r for r in list(_wb[sheet].iter_rows(values_only=True))[3:] if r and isinstance(r[1], str) and re.search(r"_\d+$", r[1])]
    he, hs, ae, as2 = ([_num0(r[i]) for r in rows] for i in (3, 4, 8, 9))
    for lang, h, a in (("en", he, ae), ("es", hs, as2)):
        z = [(x, y) for x, y in zip(h, a) if x == 0]
        nz, npos = C["zero_manual_clips_and_pipeline_nonzero"][f"{key}_{lang}"]
        check(f"{key} {lang}: clips with no manual words", len(z), nz, 0)
        check(f"{key} {lang}: ... of which pipeline non-zero", sum(y > 0 for _, y in z), npos, 0)
    P = [(s / (e + s), s2 / (e2 + s2)) for e, s, e2, s2 in zip(he, hs, ae, as2) if e + s > 0 and e2 + s2 > 0]
    rr, _, n = fit([x for x, _ in P], [y for _, y in P])
    check(f"proportion Spanish r, clips ({key})", rr, C["prop_spanish_r"][f"{key}_clip"][0], 5e-3)
    check(f"proportion Spanish n, clips ({key})", n, C["prop_spanish_r"][f"{key}_clip"][1], 0)
    rec = {}
    for r, e, s, e2, s2 in zip(rows, he, hs, ae, as2):
        v = rec.setdefault(r[1].rsplit("_", 1)[0], [0, 0, 0, 0]); v[0] += e; v[1] += s; v[2] += e2; v[3] += s2
    rr, _, _ = fit([v[1] / (v[0] + v[1]) for v in rec.values()], [v[3] / (v[2] + v[3]) for v in rec.values()])
    check(f"proportion Spanish r, recordings ({key})", rr, C["prop_spanish_r"][f"{key}_recording"], 5e-3)

print("\n=== per-recording WER workbook ===")
_wr = openpyxl.load_workbook(os.path.join(D, "wer_per_recording_14clips.xlsx"), read_only=True, data_only=True)
def _blocks(sheet):
    rows = list(_wr[sheet].iter_rows(values_only=True)); top, hdr = rows[1], rows[2]; out = {}
    for b in [x for x in top if x and x not in ("Recording", "Clips")]:
        c0 = top.index(b); col = {hdr[j]: j for j in range(c0, c0 + 14)}
        out[b] = {r[0]: {k: r[j] for k, j in col.items()} for r in rows[3:]
                  if r and isinstance(r[0], str) and r[0].startswith("P") and isinstance(r[col["WER"]], (int, float))}
    return out
WB = {s: _blocks(s) for s in ("KCHI", "MAL", "FEM", "ADULT", "OVERALL")}
def _macro(recs, k, drop=()):
    v = [r[k] for p, r in recs.items() if p not in drop]; return 100 * sum(v) / len(v), len(v)
m, n = _macro(WB["MAL"]["VTC2 + Whisper"], "WER")
check("VTC2 male-adult WER (macro, %)", m, C["vtc2_macro_wer_pct"]["MAL"][0], 0.05)
check("VTC2 male-adult WER: recordings", n, C["vtc2_macro_wer_pct"]["MAL"][1], 0)
check("VTC2 female-adult WER (macro, %)", _macro(WB["FEM"]["VTC2 + Whisper"], "WER")[0], C["vtc2_macro_wer_pct"]["FEM"], 0.05)
check("VTC2 key-child deletions (points)", _macro(WB["KCHI"]["VTC2 + Whisper"], "Deletions %")[0], C["vtc2_macro_wer_pct"]["KCHI_deletion_points"], 0.05)
check("post-hoc VTC2 key-child insertions (%)", _macro(WB["KCHI"]["WhisperX-VAD + VTC2"], "Insertions %")[0], C["posthoc_vtc2_pct"]["kchi_insertions"], 0.05)
check("post-hoc VTC2 adult deletions (%)", _macro(WB["ADULT"]["WhisperX-VAD + VTC2"], "Deletions %")[0], C["posthoc_vtc2_pct"]["adult_deletions"], 0.05)
for stt in ("OVERALL", "KCHI", "ADULT"):
    rank = lambda drop: [b for _, b in sorted((_macro(WB[stt][b], "WER", drop)[0], b) for b in WB[stt])]
    same = rank(()) == rank(("P02_pre",))
    ok_all &= same == C["ranking_unchanged_without_P02"]
    print(f"  {'PASS' if same == C['ranking_unchanged_without_P02'] else 'FAIL'}  {'ranking without P02 unchanged ('+stt+')':<44} got {same!r:>10}  want {C['ranking_unchanged_without_P02']!r}")

print("\n=== failure analyses (data/derived/failure_analysis/) ===")
FA = {r["metric"]: float(r["value"]) for r in csv.DictReader(open(os.path.join(D, "failure_analysis", "summary.csv"), encoding="utf-8-sig"))}
for k, want in C["failure_analysis_summary"].items():
    tol = 0.5 if isinstance(want, int) and k.endswith(("_pct", "_minutes")) else (0.05 if isinstance(want, float) and want > 2 else (5e-3 if isinstance(want, float) else 0))
    check(k, FA[k], want, tol)
SP = {r["utterance"]: r for r in csv.DictReader(open(os.path.join(D, "failure_analysis", "stock_phrases.csv"), encoding="utf-8-sig"))}
for u, (pl, co, ns) in C["stock_phrase_table"].items():
    check(f"stock '{u}' pipeline", int(SP[u]["pipeline"]), pl, 0)
    check(f"stock '{u}' coders", int(SP[u]["coders"]), co, 0)
    check(f"stock '{u}' on non-speech %", float(SP[u]["non_speech_pct"]), ns, 0.5)

print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
sys.exit(0 if ok_all else 1)
