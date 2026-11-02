#!/usr/bin/env python3
"""
Krippendorff's alpha for the IRR clips, rebuilt from the transcripts.

Purpose: the alpha columns of tab:supp-irr were produced by a script that no
longer exists, so it is unknown whether other-child speech was excluded from
them. This recomputes alpha under both conventions; whichever reproduces the
published values identifies what was done.

Four measures, as captioned in the paper:
  detect   speech vs silence, 100 ms bins
  speaker  speaker type, 100 ms bins
  intell   intelligible vs xxx
  lang     utterance-level language tag
"""
import argparse, os, re, sys
from collections import defaultdict
from itertools import combinations

ROLE = [(re.compile(r'^CHI$'),'KCHI'), (re.compile(r'^FA\d*$'),'FEM'),
        (re.compile(r'^MA\d*$',re.I),'MAL'), (re.compile(r'^(UC|MC|FC)\d*$'),'OCH'),
        (re.compile(r'^(UA|ME)\d*$'),'ADULT_UNSPEC')]
UNMAPPED = {'DBT','UI1','FI1','EE1'}
LANG_MAP = {'spa':'es','eng':'en','both':'both','unsure':'unsure'}
BIN_MS = 100
XXX_RE = re.compile(r'\bxxx\b', re.I)

def role_of(code, dbt='adult'):
    for pat,r in ROLE:
        if pat.match(code): return r
    if code in UNMAPPED:
        return 'ADULT_UNSPEC' if (dbt=='adult' and code=='DBT') else None
    return None

def parse(path, dbt='adult'):
    """-> (utterances, lang_rows). utterance = (start,end,role,text)"""
    utts, langs = [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 6: continue
        tier, code, a, b, _d, text = p[0], p[1], p[2], p[3], p[4], p[5]
        try: a, b = int(a), int(b)
        except ValueError: continue
        if tier.startswith('lang@'):
            langs.append((a, b, LANG_MAP.get(text.strip(), None)))
            continue
        if '@' in tier:            # lex@, mwu@, xds@ -- not used here
            continue
        r = role_of(code, dbt)
        if r is None: continue
        utts.append((a, b, r, text))
    return utts, langs

def alpha(units):
    """Krippendorff's alpha, nominal. units = list of lists of category labels."""
    o = defaultdict(float)
    for vals in units:
        m = len(vals)
        if m < 2: continue
        for c in vals:
            for k in vals:
                if c is k: continue
        # ordered pairs within the unit, weighted by 1/(m-1)
        for i in range(m):
            for j in range(m):
                if i == j: continue
                o[(vals[i], vals[j])] += 1.0/(m-1)
    if not o: return None
    cats = sorted({c for c,_ in o} | {k for _,k in o})
    n_c = {c: sum(o[(c,k)] for k in cats) for c in cats}
    n = sum(n_c.values())
    if n <= 1: return None
    Do = sum(v for (c,k),v in o.items() if c != k)
    De = sum(n_c[c]*n_c[k] for c in cats for k in cats if c != k)/(n-1)
    if De == 0: return None
    return 1.0 - Do/De

def bin_units(per_coder, lo, hi, include_och, mode):
    """per_coder: {coder: utterances}. -> list of per-bin rating lists"""
    units = []
    for t in range(lo, hi, BIN_MS):
        vals = []
        for coder, utts in per_coder.items():
            hit = None
            for (a,b,r,_txt) in utts:
                if a <= t < b:
                    if r == 'OCH' and not include_och: continue
                    hit = r; break
            if mode == 'detect':
                vals.append('speech' if hit else 'silence')
            else:                                  # speaker
                vals.append(hit if hit else 'none')
        if mode == 'speaker' and all(v == 'none' for v in vals):
            continue                               # no one heard speech here
        units.append(vals)
    return units

def utt_units(per_coder, include_och, mode, lang_by_coder=None):
    """Match utterances across coders by 100 ms bin overlap of their midpoint."""
    buckets = defaultdict(dict)
    for coder, utts in per_coder.items():
        src = utts
        for (a,b,r,txt) in src:
            if r == 'OCH' and not include_och: continue
            key = ((a+b)//2)//500                  # 500 ms bucket
            if mode == 'intell':
                v = 'xxx' if XXX_RE.search(txt) else 'transcribed'
            else:
                v = None
                for (la,lb,lv) in lang_by_coder.get(coder, []):
                    if la <= (a+b)//2 < lb or (la == a and lb == b):
                        v = lv; break
                if v is None: continue
            buckets[key].setdefault(coder, v)
    return [list(d.values()) for d in buckets.values() if len(d) >= 2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--irr-dir', default='irr_transcripts')
    args = ap.parse_args()
    files = defaultdict(dict)
    for fn in sorted(os.listdir(args.irr_dir)):
        if not fn.endswith('.txt'): continue
        base = fn[:-4]
        clip, coder = base.rsplit('-', 1)
        files[clip][coder] = os.path.join(args.irr_dir, fn)

    PAPER = {  # clip order confirmed against the WER columns
      'P06_pre_097':(.72,.65,.41,.44), 'P07_pre_093':(.86,.31,.56,.69),
      'P08_pre_018':(.69,.55,.48,.46), 'P08_pre_030':(.71,.66,.37,.45),
      'P10_pre_054':(.13,.08,.21,-.10),'P11_pre_084':(.71,.62,.36,.88),
      'P13_pre_034':(.75,.53,.08,.08), 'P14_pre_1_021':(.63,-.40,.10,None)}

    for include_och in (True, False):
        tag = 'OCH INCLUDED' if include_och else 'OCH EXCLUDED'
        print(f'\n{"="*74}\n{tag}\n{"="*74}')
        print(f'{"clip":26s} {"detect":>14s} {"speaker":>14s} {"intell":>14s} {"lang":>14s}')
        print('-'*84)
        pooled = {k: [] for k in ('detect','speaker','intell','lang')}
        for clip in sorted(files):
            per_coder, lang_by = {}, {}
            for coder, path in files[clip].items():
                u, l = parse(path)
                per_coder[coder] = u; lang_by[coder] = l
            allu = [x for u in per_coder.values() for x in u]
            if not allu: continue
            lo = (min(a for a,_,_,_ in allu)//BIN_MS)*BIN_MS
            hi = (max(b for _,b,_,_ in allu)//BIN_MS+1)*BIN_MS
            res = {}
            for mode in ('detect','speaker'):
                units = bin_units(per_coder, lo, hi, include_och, mode)
                res[mode] = alpha(units); pooled[mode] += units
            for mode in ('intell','lang'):
                units = utt_units(per_coder, include_och, mode, lang_by)
                res[mode] = alpha(units); pooled[mode] += units
            p = PAPER.get(clip, (None,)*4)
            row = f'{clip:26s}'
            for i,mode in enumerate(('detect','speaker','intell','lang')):
                v = res[mode]; pv = p[i]
                s = f'{v:.2f}' if v is not None else '---'
                ps = f'{pv:.2f}' if pv is not None else '---'
                row += f'  {s:>5s}/{ps:<5s}'
            print(row)
        print('-'*84)
        row = f'{"POOLED":26s}'
        for mode in ('detect','speaker','intell','lang'):
            v = alpha(pooled[mode])
            row += f'  {v:.2f}/{"":5s}' if v is not None else '  ---       '
        print(row + '   paper: .63 / .47 / .48 / .49')

if __name__ == '__main__':
    main()
