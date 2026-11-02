#!/usr/bin/env python3
"""
Supplementary figure: recording-level Pearson r between each pipeline's
per-language word counts and the manual counts (fig:corrmodel).

Error bars are 95% Fisher confidence intervals. Checks the published r values
before drawing and stops if any of them differ.
"""
import os, math, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
XLSX = os.path.join(ROOT, "data", "derived", "VTC2_HUman.xlsx")
OUT  = os.path.join(ROOT, "figures_out")

MODELS = ["Human", "HumanTS", "VTC1", "VTC2", "OnlyWhisper", "OnlyWhisperVTC2"]
BLOCK0 = {"Human": 2, "HumanTS": 7, "VTC1": 12, "VTC2": 17, "OnlyWhisper": 22, "OnlyWhisperVTC2": 27}
LABEL  = {"HumanTS": "Human-TS\n+ WhisperX", "VTC1": "VTC1\n+ WhisperX",
          "VTC2": "VTC2\n+ WhisperX", "OnlyWhisper": "WhisperX-VAD\n+ VTC1",
          "OnlyWhisperVTC2": "WhisperX-VAD\n+ VTC2"}
EN_C, ES_C = "#31597d", "#d99b3f"
PUBLISHED = {("HumanTS","en"):0.93, ("HumanTS","es"):0.94, ("VTC1","en"):0.87,
             ("VTC1","es"):0.85, ("VTC2","en"):0.97, ("VTC2","es"):0.96,
             ("OnlyWhisper","en"):0.84, ("OnlyWhisper","es"):0.85}

def load(sheet="OVERALL"):
    raw = pd.read_excel(XLSX, sheet_name=sheet, header=None)
    d = raw.iloc[3:].copy()
    rec  = d.iloc[:, 0].ffill().astype(str)
    clip = d.iloc[:, 1].astype(str)
    out = pd.DataFrame({"recording": rec.values, "clip": clip.values})
    for m in MODELS:
        c = BLOCK0[m]
        for j, k in ((0, "total"), (1, "en"), (2, "es")):
            out[f"{m}_{k}"] = pd.to_numeric(d.iloc[:, c + j].values, errors="coerce")
    # a real clip row ends in _<digits>; the per-recording subtotal does not.
    # (matching on an id prefix breaks once identifiers are anonymised)
    keep = out["clip"].str.match(r"^.+_\d+$") & ~out["clip"].str.contains("COMBINED", case=False)
    out = out[keep].copy()
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].fillna(0)
    return out.groupby("recording", as_index=False).sum(numeric_only=True)

def fisher_ci(r, n, conf=0.95):
    if n < 4 or abs(r) >= 1: return (r, r)
    z = 0.5 * math.log((1 + r) / (1 - r)); se = 1 / math.sqrt(n - 3)
    from scipy import stats
    c = stats.norm.ppf(0.5 + conf / 2)
    lo, hi = z - c * se, z + c * se
    t = lambda v: (math.exp(2 * v) - 1) / (math.exp(2 * v) + 1)
    return t(lo), t(hi)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    df = load(); n = len(df)
    comp = [m for m in MODELS if m != "Human"]
    r = {(m, lg): float(np.corrcoef(df[f"Human_{lg}"], df[f"{m}_{lg}"])[0, 1])
         for m in comp for lg in ("en", "es")}
    ok = True
    print(f"validating against published values (N = {n} recordings)")
    for k, want in PUBLISHED.items():
        good = round(r[k], 2) == want; ok &= good
        print(f"   {k[0]:12s} {k[1]}: r={r[k]:.2f}/{want}  {'ok' if good else '<-- MISMATCH'}")
    if not ok: raise SystemExit("validation failed - refusing to write the figure")
    if a.validate_only:
        print("validation only; no figure written"); return

    ci = {k: fisher_ci(v, n) for k, v in r.items()}
    x = np.arange(len(comp)); w = 0.36
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for off, lg, col, lab in ((-w/2, "en", EN_C, "English words"),
                              ( w/2, "es", ES_C, "Spanish words")):
        vals = [r[(m, lg)] for m in comp]
        lo   = [r[(m, lg)] - ci[(m, lg)][0] for m in comp]
        hi   = [ci[(m, lg)][1] - r[(m, lg)] for m in comp]
        ax.bar(x + off, vals, w, color=col, edgecolor="white", label=lab,
               yerr=[lo, hi], capsize=4, error_kw=dict(lw=1.2, ecolor="#3a3a3a"))
        for xi, v, h in zip(x + off, vals, hi):
            ax.text(xi, v + h + 0.02, f"{v:.2f}", ha="center", fontsize=11.5)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in comp], fontsize=11.5)
    ax.set_ylabel("Pearson $r$ with manual counts", fontsize=13)
    ax.set_ylim(0, 1.16); ax.grid(axis="x", visible=False); ax.grid(axis="y", alpha=.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=12.5, columnspacing=1.8)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "corr_by_model.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"   wrote {p}")

if __name__ == "__main__":
    main()
