#!/usr/bin/env python3
"""
Draws fig_total_words.png and fig_lexical_diversity.png.

Checks the r / slope / n values printed in the paper before drawing and stops
if any of them differ.

Data:  data/derived/word_count_comparison_14clips.xlsx  (total words)
       data/derived/vocab_by_speaker.csv  (per clip x speaker: human/vtc2 type
       and token counts, by language)
Fit:   clips with a non-zero MANUAL count in that language (Methods,
       sec:methods-stats). Clips with a manual zero are plotted at x=0 but
       excluded from the fit, r and n.
"""
import os, csv, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV  = os.path.join(ROOT, "data", "derived", "vocab_by_speaker.csv")
XLSX = os.path.join(ROOT, "data", "derived", "word_count_comparison_14clips.xlsx")
OUT  = os.path.join(ROOT, "figures_out")

EN = "#31597d"   # English  - blue
ES = "#b02a3f"   # Spanish  - crimson
IDENT = "#7a7a7a"

SPEAKERS = [("adult", "adults"), ("kchild", "key child")]

PUBLISHED = {  # (measure, speaker, lang) -> (r, slope, n)
    ("tokens","adult","en"):(.93,0.68,75),  ("tokens","adult","es"):(.90,0.64,149),
    ("tokens","kchild","en"):(.88,0.77,79), ("tokens","kchild","es"):(.86,0.53,154),
    ("types","adult","en"):(.88,0.80,63),   ("types","adult","es"):(.87,0.79,148),
    ("types","kchild","en"):(.83,0.88,70),  ("types","kchild","es"):(.89,0.82,152),
}

SHEET = {"adult": "ADULT", "kchild": "KCHI"}
_CLIP = None

def load():
    """types come from the per-speaker vocab csv (158 adult / 161 key-child clips);
    tokens come from the word-count workbook (all 168 clips). Using one source for
    both does not reproduce the published figures - the clip coverage differs."""
    import openpyxl, re as _re
    with open(CSV, encoding="utf-8-sig") as f:
        vocab = list(csv.DictReader(f))
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    clip_re = _re.compile(r"_\d+$")
    num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
    counts = {}
    for spk, sheet in SHEET.items():
        rows = [r for r in list(wb[sheet].iter_rows(values_only=True))[3:]
                if r and isinstance(r[1], str) and clip_re.search(r[1])]
        counts[spk] = [{"clip": r[1],
                        "human_en": num(r[3]), "human_es": num(r[4]),
                        "vtc2_en":  num(r[8]), "vtc2_es":  num(r[9])} for r in rows]
    return {"types": vocab, "tokens": counts}

def series(data, speaker, lang, measure):
    """Returns (x_fit, y_fit, x_zero, y_zero) - manual on x, automated on y."""
    xf, yf, xz, yz = [], [], [], []
    if measure == "types":
        for r in data["types"]:
            if r["speaker"] != speaker:
                continue
            h = float(r[f"human_{lang}_types"]); a = float(r[f"vtc2_{lang}_types"])
            (xf.append(h) or yf.append(a)) if h > 0 else (xz.append(h) or yz.append(a))
    else:
        for r in data["tokens"][speaker]:
            h = r[f"human_{lang}"]; a = r[f"vtc2_{lang}"]
            (xf.append(h) or yf.append(a)) if h > 0 else (xz.append(h) or yz.append(a))
    return np.array(xf), np.array(yf), np.array(xz), np.array(yz)

def ols(x, y):
    n = len(x); mx, my = x.mean(), y.mean()
    sxx = ((x-mx)**2).sum(); sxy = ((x-mx)*(y-my)).sum(); syy = ((y-my)**2).sum()
    slope = sxy/sxx; inter = my - slope*mx
    r = sxy/math.sqrt(sxx*syy)
    resid = y - (inter + slope*x)
    s2 = (resid**2).sum()/(n-2)
    return slope, inter, r, s2, mx, sxx, n

def band(x, xs, slope, inter, s2, mx, sxx, n):
    """95% CI for the mean response."""
    from scipy import stats as st
    t = st.t.ppf(0.975, n-2)
    se = np.sqrt(s2*(1.0/n + (xs-mx)**2/sxx))
    fit = inter + slope*xs
    return fit - t*se, fit + t*se

def validate(rows):
    ok = True
    print("validating against published values")
    for (meas, spk, lang), (pr, ps, pn) in sorted(PUBLISHED.items()):
        xf, yf, _, _ = series(rows, spk, lang, meas)
        slope, inter, r, *_ = ols(xf, yf)
        good = (round(r,2)==pr and round(slope,2)==ps and len(xf)==pn)
        ok &= good
        print(f"   {meas:6s} {spk:6s} {lang}: r={r:.2f}/{pr}  slope={slope:.2f}/{ps}  "
              f"n={len(xf)}/{pn}  {'ok' if good else '<-- MISMATCH'}")
    return ok

def panel(ax, rows, speaker, measure, unit, scale):
    for lang, colour, marker, label in (("en", EN, "^", "English"), ("es", ES, "o", "Spanish")):
        xf, yf, xz, yz = series(rows, speaker, lang, measure)
        slope, inter, r, s2, mx, sxx, n = ols(xf, yf)
        ax.scatter(xz, yz, s=scale["dot"], marker=marker, c=colour, alpha=.65,
                   linewidths=0, zorder=2)
        ax.scatter(xf, yf, s=scale["dot"], marker=marker, c=colour, alpha=.85,
                   linewidths=0, zorder=3,
                   label=f"{label}  (r = {r:.2f}, slope = {slope:.2f}, n = {n})")
        xs = np.linspace(xf.min(), xf.max(), 100)
        lo, hi = band(xf, xs, slope, inter, s2, mx, sxx, n)
        ax.fill_between(xs, lo, hi, color=colour, alpha=.15, hatch="///",
                        edgecolor=colour, linewidth=0, zorder=1)
        ax.plot(xs, inter + slope*xs, color=colour, lw=scale["line"], zorder=4)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", color=IDENT, lw=scale["line"]*0.8, zorder=1)
    ax.set_xlim(left=-lim*0.015, right=lim)
    ax.set_ylim(bottom=-lim*0.015, top=lim)
    ax.text(lim*0.88, lim*0.90, "y = x", rotation=45, rotation_mode="default",
            ha="right", va="bottom", color=IDENT, fontsize=scale["annot"])
    ax.set_xlabel(f"Manual counts ({unit})")
    ax.set_ylabel(f"Automated counts ({unit})")
    ax.legend(loc="upper left", frameon=False, fontsize=scale["legend"],
              handletextpad=.4, borderpad=.2)
    ax.grid(alpha=.25, lw=.7)
    for s in ("top","right"): ax.spines[s].set_visible(False)

def figure(rows, measure, unit, title, fname, scale):
    fig, axes = plt.subplots(1, 2, figsize=scale["figsize"])
    for ax, (key, nice) in zip(axes, SPEAKERS):
        panel(ax, rows, key, measure, unit, scale)
        ax.set_title(f"{title}: {nice}", fontsize=scale["title"], pad=12)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.35,
                    help="multiply all font sizes (1.35 is used in the paper)")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    rows = load()
    if not validate(rows):
        raise SystemExit("validation failed - refusing to overwrite the figures")
    if a.validate_only:
        print("validation only; no figures written"); return
    k = a.scale
    scale = dict(figsize=(15, 6.2), dot=52*k**0.5, line=2.0,
                 title=13*k, legend=11*k, annot=10*k)
    plt.rcParams.update({
        "font.size": 11*k, "axes.labelsize": 12*k,
        "xtick.labelsize": 11*k, "ytick.labelsize": 11*k,
        "font.family": "DejaVu Sans",
    })
    os.makedirs(OUT, exist_ok=True)
    figure(rows, "tokens", "words", "Total number of words",
           "fig_total_words.png", scale)
    figure(rows, "types", "number of different words", "Lexical diversity",
           "fig_lexical_diversity.png", scale)

if __name__ == "__main__":
    main()
