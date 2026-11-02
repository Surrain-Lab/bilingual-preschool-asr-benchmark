#!/usr/bin/env python3
"""
Writes every supplementary table (A1-A9, numbered as printed) to tables_out/supp_*.tex from data/derived.

The manuscript's supplementary_tables.tex \\input's these files. Captions and column
headers are fixed text below; every number is computed from the shipped data:

  A1 recording-summary    word_count_comparison_14clips.xlsx (human counts, COMBINED rows)
  A2 supp-irr             irr_symmetric_wer.csv + irr_alpha/{krippendorff,language}_alpha.csv
  A3 supp-wer             supp_bootstrap.csv (means and 95% CIs)
  A4 supp-pairwise-overall, A5 supp-pairwise-kchi, A6 supp-pairwise-adult   supp_pairwise.csv
  A7 supp-posthoc-vtc2    supp_wer_by_pipeline.csv + supp_pairwise.csv
  A8 supp-per-recording   wer_per_recording_14clips.xlsx (KCHI and ADULT sheets)
  A9 supp-seg-gap         supp_wer_by_pipeline.csv

File names follow the printed table numbers (supp_A2_irr.tex, supp_A3_wer.tex, ...).
Set PERCENT below to choose how WER is printed (the paper's text uses percent).

Run make_supp_stats.py first if the per-recording workbook changed.
"""
import csv, os, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "tables_out"); os.makedirs(OUT, exist_ok=True)

def rcsv(*p):
    return list(csv.DictReader(open(os.path.join(D, *p), encoding="utf-8-sig")))

def wb_rows(name, sheet):
    import openpyxl
    return list(openpyxl.load_workbook(os.path.join(D, name), read_only=True, data_only=True)[sheet]
                .iter_rows(values_only=True))

def write(name, head, rows, tail):
    p = os.path.join(OUT, f"supp_{name}.tex")
    open(p, "w", encoding="utf-8").write(head + "\n" + "\n".join(rows) + "\n" + tail + "\n")
    print(f"wrote {p}")

# short codes (as in the CSVs), table labels, and full names (as in supp_wer_by_pipeline.csv)
PIPES = [("HumanTS", "HumanTS", "Human-TS + WhisperX (control)"),
         ("VTC1", "VTC1", "VTC1 + WhisperX"),
         ("VTC2", "VTC2", "VTC2 + WhisperX"),
         ("VTC2-NL", "VTC2-NL", "VTC2-NoLang + WhisperX"),
         ("WhisperX", "PH-VTC1", "WhisperX-VAD + VTC1 (post-hoc)"),
         ("WhisperX-VTC2", "PH-VTC2", "WhisperX-VAD + VTC2 (post-hoc)")]
WB_BLOCK = {"HumanTS": "Human-TS + Whisper", "VTC1": "VTC1 + Whisper", "VTC2": "VTC2 + Whisper",
            "VTC2-NL": "VTC2-NoLang + Whisper", "WhisperX": "Only Whisper",
            "WhisperX-VTC2": "WhisperX-VAD + VTC2"}

# WER format. True: percent with one decimal (71.4), differences in percentage points -- matches the
# manuscript text and main tables. False: the earlier three-decimal proportions (0.714).
PERCENT = True

def f3(x):   return f"{100*x:.1f}" if PERCENT else f"{x:.3f}"
def ap(x):
    if PERCENT: return f"{100*x:.1f}"
    return f"{x:.3f}".lstrip("0") if x >= 0 else "$-$" + f"{-x:.3f}".lstrip("0")
def fa2(x):  return f"{x:.2f}".lstrip("0") if x >= 0 else "$-$" + f"{-x:.2f}".lstrip("0")   # alpha: never percent
def signed(x, math_inside=True):
    s = "+" if x >= 0 else "-"
    v = f"{100*abs(x):.1f}" if PERCENT else f"{abs(x):.3f}"
    return f"${s}{v}" if math_inside else f"${s}${v}"
def shown(x):  # the value as printed, for ties and "lowest in row"
    return round(100*x, 1) if PERCENT else round(x, 3)
def pfmt(p):  return (f"{p:.3f}" if p < .05 else f"{p:.2f}").lstrip("0")

PAIR = {}
for r in rcsv("supp_pairwise.csv"):
    PAIR[(r["stratum"], r["pipe_a"], r["pipe_b"])] = (float(r["delta"]), float(r["p"]), r["stars"])
def pair(stratum, a, b):
    if (stratum, a, b) in PAIR: return PAIR[(stratum, a, b)]
    d, p, s = PAIR[(stratum, b, a)]; return -d, p, s
# Means and CIs are recomputed at full precision with make_supp_stats.py's own functions (each
# bootstrap re-seeds with 42, so this reproduces supp_bootstrap.csv exactly). Several values sit on a
# rounding tie at three decimals (e.g. Human-TS overall .7155), so the 4-decimal CSVs must not be
# re-rounded.
import sys; sys.path.insert(0, HERE)
import make_supp_stats as ms
STATS = {}
for _sheet in ms.SHEETS:
    _d = ms.load(_sheet)
    for _blk in ms.PIPES:
        STATS[(_sheet, ms.SHORT[_blk])] = ms.bootstrap_ci(list(_d[_blk].values()))
_KEY = {"kchi_macro_wer": "KCHI", "adult_macro_wer": "ADULT", "overall_macro_wer": "OVERALL"}
SW = {full: {k: STATS[(st, code)][0] for k, st in _KEY.items()} for code, _, full in PIPES}

HEAD = {'recording-summary': '\\begin{table}\n\\centering\n\\small\n\\caption{Recording-level descriptive composition of the expert-reference corpus. Each recording contributed 12 five-minute clips.}\n\\label{tab:recording-summary}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrr}\n\\toprule\nParticipant & Clips & KCHI words & KCHI \\%Es & Adult words & Adult \\%Es \\\\\n\\midrule', 'supp-wer': '\\begin{table}\n\\centering\n\\small\n\\caption{Word error rate (WER) per pipeline and speaker type. Values are means with 95\\% bootstrap confidence intervals (5{,}000 resamples at the recording level). $n$ = number of recordings. Human-TS + WhisperX is an expert-segmentation control, not a strict upper bound.}\n\\label{tab:supp-wer}\n\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\n& \\multicolumn{2}{c}{Key Child (KCHI)}\n& \\multicolumn{2}{c}{Adult (MAL+FEM)}\n& \\multicolumn{2}{c}{Overall} \\\\\n\\cmidrule(lr){2-3}\n\\cmidrule(lr){4-5}\n\\cmidrule(lr){6-7}\n\nPipeline\n& WER [95\\% CI] & $n$\n& WER [95\\% CI] & $n$\n& WER [95\\% CI] & $n$ \\\\\n\\midrule', 'supp-pairwise-kchi': '\\begin{table}\n\\centering\n\\small\n\\caption{Pairwise bootstrap significance tests for WER differences between pipelines for key-child speech. Values are $\\Delta$WER = row pipeline $-$ column pipeline. $^{*}p<.05$; $^{**}p<.01$; $^{***}p<.001$; no superscript = not significant. Tests used 10{,}000 recording-level bootstrap resamples. Only recordings with WER values in both pipelines are used per pair. PH-VTC1 and PH-VTC2 are WhisperX-VAD with speaker labels added afterwards (post-hoc) by VTC1 and VTC2 respectively.}\n\\label{tab:supp-pairwise-kchi}\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n& \\textbf{HumanTS} & \\textbf{VTC1} & \\textbf{VTC2} & \\textbf{VTC2-NL} & \\textbf{PH-VTC1} & \\textbf{PH-VTC2} \\\\\n\\midrule', 'supp-pairwise-adult': '\\begin{table}\n\\centering\n\\small\n\\caption{Pairwise bootstrap significance tests for WER differences between pipelines for adult speech. Values are $\\Delta$WER = row pipeline $-$ column pipeline. $^{*}p<.05$; $^{**}p<.01$; $^{***}p<.001$; no superscript = not significant. Tests used 10{,}000 recording-level bootstrap resamples. Only recordings with WER values in both pipelines are used per pair. PH-VTC1 and PH-VTC2 are WhisperX-VAD with speaker labels added afterwards (post-hoc) by VTC1 and VTC2 respectively.}\n\\label{tab:supp-pairwise-adult}\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n& \\textbf{HumanTS} & \\textbf{VTC1} & \\textbf{VTC2} & \\textbf{VTC2-NL} & \\textbf{PH-VTC1} & \\textbf{PH-VTC2} \\\\\n\\midrule', 'supp-pairwise-overall': '\\begin{table}\n\\centering\n\\small\n\\caption{Pairwise bootstrap significance tests for WER differences between pipelines for overall speech. Values are $\\Delta$WER = row pipeline $-$ column pipeline. $^{*}p<.05$; $^{**}p<.01$; $^{***}p<.001$; no superscript = not significant. Tests used 10{,}000 recording-level bootstrap resamples. Only recordings with WER values in both pipelines are used per pair. PH-VTC1 and PH-VTC2 are WhisperX-VAD with speaker labels added afterwards (post-hoc) by VTC1 and VTC2 respectively.}\n\\label{tab:supp-pairwise-overall}\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n& \\textbf{HumanTS} & \\textbf{VTC1} & \\textbf{VTC2} & \\textbf{VTC2-NL} & \\textbf{PH-VTC1} & \\textbf{PH-VTC2} \\\\\n\\midrule', 'supp-per-recording': '\\begin{table}\n\\centering\n\\scriptsize\n\\caption{Per-recording WER for the four pipelines and two comparison conditions on key-child and adult speech. Recordings are sorted by identifier. Bold = lowest WER per row. PH-VTC1 and PH-VTC2 are WhisperX-VAD with speaker labels added afterwards (post-hoc) by VTC1 and VTC2 respectively.}\n\\label{tab:supp-per-recording}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccccc}\n\\toprule\nSpk. & Recording & HumanTS & VTC1 & VTC2 & VTC2-NoLang & PH-VTC1 & PH-VTC2 \\\\\n\\midrule', 'supp-posthoc-vtc2': '\\begin{table}\n\\centering\n\\caption{Effect of the speaker classifier in the post-hoc pipelines. Both rows use the\nsame WhisperX transcript and differ only in which classifier assigns speaker labels\nafterwards. Values are macro-averages over the 14 recordings, taken from Table~\\ref{tab:supp-wer};\n$p$ values are from the pairwise bootstrap tests in Tables~\\ref{tab:supp-pairwise-overall}--\\ref{tab:supp-pairwise-adult}.\nOverall WER can change because words that the classifier assigns to no speaker, or to its generic\nspeech class, are left out of the scored transcript. Differences are computed before rounding.\nFor reference, VTC2~+~WhisperX, in which VTC2 also performs the segmentation, reaches\n$.714$ overall, $.753$ key child and $.721$ adult.}\n\\label{tab:supp-posthoc-vtc2}\n\\small\n\\begin{tabular}{lrrr}\n\\toprule\nSpeaker labels from & Overall & Key child & Adult \\\\\n\\midrule', 'supp-seg-gap': '\\begin{table}\n\\centering\n\\small\n\\caption{Segmentation gap: mean WER for each automated pipeline versus the expert-segmentation control (Human-TS + WhisperX). Gap = automated WER $-$ control WER. Positive gap = WER caused by finding speech and assigning speakers.}\n\\label{tab:supp-seg-gap}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc}\n\\toprule\n& \\multicolumn{2}{c}{Key Child (KCHI)}\n& \\multicolumn{2}{c}{Adult}\n& \\multicolumn{2}{c}{Overall} \\\\\n\\cmidrule(lr){2-3}\n\\cmidrule(lr){4-5}\n\\cmidrule(lr){6-7}\nPipeline & WER & Gap & WER & Gap & WER & Gap \\\\\n\\midrule', 'supp-irr': "\\begin{table}\n\\centering\n\\small\n\\caption{Inter-annotator reliability by clip. $k$ is the number of coders who transcribed\nthe clip. WER columns are the mean symmetric word error rate across all coder pairs,\ncomputed with the normalization and alignment used for the automated pipelines\n(Sections~\\ref{sec:methods-normalization} and~\\ref{sec:methods-wer}), following the same other-child convention as the pipeline comparison: OCH speech is excluded from the key-child and adult columns and retained in the Overall column. OCH speech is likewise excluded from all $\\alpha$ columns. $\\alpha$ columns are Krippendorff's $\\alpha$ for speech detection and speaker type (100~ms bins), intelligibility (transcribed vs.\\ \\texttt{xxx}),\nand utterance-level language tag. Dashes mark strata in which at least one coder produced\nno words, for which a rate is undefined.}\n\\label{tab:supp-irr}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n& & \\multicolumn{3}{c}{Symmetric WER} & \\multicolumn{4}{c}{Krippendorff's $\\alpha$} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\nClip & $k$ & Key child & Adult & Overall & Detect & Speaker & Intell. & Lang. \\\\\n\\midrule"}
TAIL = {'recording-summary': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-wer': '\\bottomrule\n\\end{tabular}%\n}\n\\end{table}', 'supp-pairwise-kchi': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-pairwise-adult': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-pairwise-overall': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-per-recording': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-posthoc-vtc2': '\\bottomrule\n\\end{tabular}\n\\end{table}', 'supp-seg-gap': '\\bottomrule\n\\end{tabular*}\n\\end{table}', 'supp-irr': '\\bottomrule\n\\end{tabular}%\n}\n\\end{table}'}

if PERCENT:
    _v2 = SW["VTC2 + WhisperX"]
    for _k, _old, _new in [
        ("supp-wer", "Values are means with", "Values are means (WER in percent) with"),
        ("supp-pairwise-kchi", r"Values are $\Delta$WER", r"Values are $\Delta$WER in percentage points"),
        ("supp-pairwise-adult", r"Values are $\Delta$WER", r"Values are $\Delta$WER in percentage points"),
        ("supp-pairwise-overall", r"Values are $\Delta$WER", r"Values are $\Delta$WER in percentage points"),
        ("supp-seg-gap", r"Gap = automated WER $-$ control WER.", r"Gap = automated WER $-$ control WER, in percentage points."),
        ("supp-irr", "WER columns are the mean symmetric word error rate", "WER columns are the mean symmetric word error rate (in percent)"),
        ("supp-posthoc-vtc2", "reaches\n$.714$ overall, $.753$ key child and $.721$ adult.",
         "reaches\n" + f"{f3(_v2['overall_macro_wer'])}\\% overall, {f3(_v2['kchi_macro_wer'])}\\% key child and {f3(_v2['adult_macro_wer'])}\\% adult."),
    ]:
        assert _old in HEAD[_k], (_k, _old)
        HEAD[_k] = HEAD[_k].replace(_old, _new)

def a1():
    rows = []
    for sheet in ("KCHI", "ADULT"):
        cur = None; tot = collections.OrderedDict()
        for r in wb_rows("word_count_comparison_14clips.xlsx", sheet)[3:]:
            if r[0]: cur = r[0]
            if not r[1]: continue
            a = tot.setdefault(cur, [0, 0, 0])
            if str(r[1]).endswith("_COMBINED"): a[0], a[1] = r[2], r[4]
            else: a[2] += 1
        rows.append(tot)
    k, ad = rows
    out = []
    for rec in k:
        lab = rec.split("_")[0]
        out.append(f"{lab} & {k[rec][2]} & {k[rec][0]:,} & {100*k[rec][1]/k[rec][0]:.1f} & "
                   f"{ad[rec][0]:,} & {100*ad[rec][1]/ad[rec][0]:.1f} \\\\")
    write("A1_recording_summary", HEAD["recording-summary"], out, TAIL["recording-summary"])

def a2():
    out = [""]
    for code, _, full in PIPES:
        out.append(full)
        cells = []
        for st in ("KCHI", "ADULT", "OVERALL"):
            m, lo, hi = STATS[(st, code)]
            cells.append(f"& {f3(m)} [{f3(lo)}, {f3(hi)}] & 14")
        out += [cells[0], cells[1], cells[2] + " \\\\", ""]
    write("A3_wer", HEAD["supp-wer"], out, TAIL["supp-wer"])

def a345():
    for st, nm, key in (("KCHI", "A5_pairwise_kchi", "supp-pairwise-kchi"),
                        ("ADULT", "A6_pairwise_adult", "supp-pairwise-adult"),
                        ("OVERALL", "A4_pairwise_overall", "supp-pairwise-overall")):
        out = []
        for a, la, _ in PIPES:
            out.append(r"\textbf{" + la + "}")
            for j, (b, _, _) in enumerate(PIPES):
                if a == b: c = "—"
                else:
                    d, p, s = pair(st, a, b)
                    c = signed(d) + (f"^{{{s}}}" if s else "") + "$"
                out.append(f"& {c}" + (" \\\\" if j == len(PIPES) - 1 else ""))
        write(nm, HEAD[key], out, TAIL[key])

def a6():
    blocks = {}
    for sheet in ("KCHI", "ADULT"):
        rows = wb_rows("wer_per_recording_14clips.xlsx", sheet); top, hdr = rows[1], rows[2]
        col = {}
        for code, _, _ in PIPES:
            c0 = top.index(WB_BLOCK[code]); col[code] = next(j for j in range(c0, c0 + 14) if hdr[j] == "WER")
        recs = [r for r in rows[3:] if r and isinstance(r[0], str) and r[0].startswith("P")]
        blocks[sheet] = [(r[0].split("_")[0], [r[col[c]] for c, _, _ in PIPES]) for r in recs]
    out = [r"\multicolumn{8}{l}{\textbf{Key Child speech}} \\"]
    for sheet in ("KCHI", "ADULT"):
        if sheet == "ADULT":
            out += [r"\midrule", r"\multicolumn{8}{l}{\textbf{Adult speech}} \\"]
        for lab, v in blocks[sheet]:
            if not all(isinstance(x, (int, float)) for x in v): continue
            lo = min(shown(x) for x in v)
            cells = [(r"\textbf{" + f3(x) + "}") if shown(x) == lo else f3(x) for x in v]
            out.append(f"& {lab} & " + " & ".join(cells) + " \\\\")
    write("A8_per_recording", HEAD["supp-per-recording"], out, TAIL["supp-per-recording"])

def a7():
    v1, v2 = SW["WhisperX-VAD + VTC1 (post-hoc)"], SW["WhisperX-VAD + VTC2 (post-hoc)"]
    keys = (("overall_macro_wer", "OVERALL"), ("kchi_macro_wer", "KCHI"), ("adult_macro_wer", "ADULT"))
    def wfmt(x):
        if PERCENT: return f"{100*x:.1f}"
        return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"
    out = ["VTC1                                & " + " & ".join(wfmt(float(v1[k])) for k, _ in keys) + " \\\\",
           "VTC2                                & " + " & ".join(wfmt(float(v2[k])) for k, _ in keys) + " \\\\",
           r"\midrule",
           "Difference (VTC2 $-$ VTC1)          & " + " & ".join(
               (signed(float(v2[k]) - float(v1[k]), math_inside=False) if PERCENT else
                signed(float(v2[k]) - float(v1[k]), math_inside=False).replace("0.", ".")) for k, _ in keys) + " \\\\",
           "$p$                                 & " + " & ".join(
               pfmt(pair(st, "WhisperX", "WhisperX-VTC2")[1]) for _, st in keys) + " \\\\"]
    write("A7_posthoc_vtc2", HEAD["supp-posthoc-vtc2"], out, TAIL["supp-posthoc-vtc2"])

def a8():
    keys = ("kchi_macro_wer", "adult_macro_wer", "overall_macro_wer")
    ctl = SW["Human-TS + WhisperX (control)"]
    out = ["Human-TS + WhisperX (control)"] + [f"& {f3(float(ctl[k]))} & —" + (" \\\\" if k == keys[-1] else "") for k in keys]
    out.append(r"\midrule")
    for _, _, full in PIPES[1:]:
        r = SW[full]; out.append(full)
        for k in keys:
            g = float(r[k]) - float(ctl[k])
            out.append(f"& {f3(float(r[k]))} & {signed((round(g, 4) if PERCENT else round(g, 3)) + 0.0, math_inside=False)}" + (" \\\\" if k == keys[-1] else ""))
    write("A9_seg_gap", HEAD["supp-seg-gap"], out, TAIL["supp-seg-gap"])

def a9():
    pairs = collections.defaultdict(list); pooled = {}
    for r in rcsv("irr_symmetric_wer.csv"):
        if r["sym_wer"]: pairs[(r["clip_id"], r["stratum"])].append(r)
    alpha = collections.defaultdict(dict)
    for f in ("krippendorff_alpha.csv", "language_alpha.csv"):
        for r in rcsv("irr_alpha", f): alpha[r["clip_id"]][r["layer"]] = float(r["alpha"])
    clips = sorted({c for c, _ in pairs})
    def mw(c, st):
        v = pairs.get((c, st)); return ap(statistics.mean(float(x["sym_wer"]) for x in v)) if v else "---"
    def al(c, layer):
        x = alpha[c].get(layer, float("nan")); return "---" if x != x else fa2(x)
    out = []
    for i, c in enumerate(clips, 1):
        n_pairs = len(pairs[(c, "OVERALL")]); k = {1: 2, 3: 3, 6: 4}[n_pairs]
        out.append(f"{i} & {k} & {mw(c,'KCHI')} & {mw(c,'ADULT')} & {mw(c,'OVERALL')} & "
                   f"{al(c,'detection')} & {al(c,'speaker')} & {al(c,'intelligibility')} & {al(c,'lang_utterance')} \\\\")
    # pooled symmetric WER: summed errors over summed mean length, across all pairs (as verify.py)
    def pw(st):
        rs = [r for (c, s), v in pairs.items() if s == st for r in v]
        err = sum(int(r["sub"]) + int(r["del"]) + int(r["ins"]) for r in rs)
        n = sum((int(r["n_words_a"]) + int(r["n_words_b"])) / 2 for r in rs)
        return ap(err / n)
    out += [r"\midrule",
            f"Pooled & --- & {pw('KCHI')} & {pw('ADULT')} & {pw('OVERALL')} & {al('POOLED','detection')} & "
            f"{al('POOLED','speaker')} & {al('POOLED','intelligibility')} & {al('POOLED','lang_utterance')} \\\\"]
    write("A2_irr", HEAD["supp-irr"], out, TAIL["supp-irr"])

if __name__ == "__main__":
    a1(); a2(); a345(); a6(); a7(); a8(); a9()
