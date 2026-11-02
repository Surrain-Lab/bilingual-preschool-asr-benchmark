#!/usr/bin/env python3
"""
Inter-annotator WER, word counts and ICC, scored with the same text cleanup
and WER code as the pipelines.

Run from the repository root:
    python stage2_aggregate/irr_symmetric_wer.py --irr-dir irr_transcripts

How this differs from irr_interannotator.py:
  * normalization is VTC2_Pipeline/evaluation.py:clean_text(), not a
    reimplementation (so hyphens MERGE compounds, "de el"->"del", inverted
    punctuation is stripped, and the CHAT "0" marker is KEPT, exactly as the
    reference corpus is scored)
  * strata are the paper's: KCHI, ADULT (MAL+FEM), OVERALL. OCH is excluded from
    the speaker-stratified columns and RETAINED in OVERALL, which is
    speaker-agnostic -- the same rule the pipeline comparison uses
  * WER comes from jiwer.compute_measures, the same call the pipeline uses, so
    hits/substitutions/deletions/insertions decompose identically
  * word counts and ICC(1,1) replicate irr_counts.csv / irr_results.csv, which
    is the validation that this parse matches the existing process

Every pair is scored in both directions and symmetrically; no coder is the
reference.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'stage1_pipeline', 'VTC2_Pipeline'))
# Both come straight from the pipeline's own scorer -- clean_text is the exact
# normalizer applied to the reference corpus, and evaluation.py already wraps
# jiwer.process_words in its own compute_measures shim.
from evaluation import clean_text, compute_measures    # noqa: E402

# The paper's four categories (methods.tex, Annotation procedure).
ROLE = [
    (re.compile(r'^CHI$'),            'KCHI'),
    (re.compile(r'^FA\d*$'),          'FEM'),
    (re.compile(r'^MA\d*$'),          'MAL'),
    (re.compile(r'^(UC|MC|FC)\d*$'),  'OCH'),
    (re.compile(r'^(UA|ME)\d*$'),     'ADULT_UNSPEC'),   # adult, gender not established
]
# DBT / UI1 / FI1 / EE1 map to no paper category. irr_counts.csv folds DBT into
# adult_wc, so that convention is replicated here under --dbt adult.
UNMAPPED = {'DBT', 'UI1', 'FI1', 'EE1'}
MAX_UTT_MS = 120_000
LANG_MAP = {'spa': 'es', 'eng': 'en', 'both': 'both', 'unsure': 'unsure'}
BRACE_RE = re.compile(r'\{([^}]*)\}')

STRATA = {
    'KCHI':    ['KCHI'],
    'ADULT':   ['FEM', 'MAL', 'ADULT_UNSPEC'],
    # OVERALL is a speaker-agnostic stratum, so it retains OCH -- matching how the
    # pipeline comparison scores the pooled <clip>_clean.txt, which contains all four
    # speaker types. The previous setting excluded OCH here, which made this column
    # incomparable with the pipeline OVERALL it is read against.
    'OVERALL': ['KCHI', 'FEM', 'MAL', 'ADULT_UNSPEC', 'OCH'],
    'OCH':     ['OCH'],                                   # reported, not used in the paper's WER
}


def role_of(code, dbt_mode):
    for pat, r in ROLE:
        if pat.match(code):
            return r
    if code in UNMAPPED:
        return 'ADULT_UNSPEC' if (dbt_mode == 'adult' and code == 'DBT') else None
    return None


def parse(path, dbt_mode):
    utts, langs = [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 6:
            continue
        tier, code = parts[0], (parts[1] or parts[0]).strip()
        try:
            start, end = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        text = '\t'.join(parts[5:]).strip()
        if tier.startswith('lang@'):
            v = LANG_MAP.get(text.strip().lower())
            if v:
                langs.append((code, start, end, v))
            continue
        if '@' in tier or end <= start or (end - start) > MAX_UTT_MS:
            continue
        r = role_of(code, dbt_mode)
        if r is None:
            continue
        utts.append({'role': r, 'code': code, 'start': start, 'end': end,
                     'text': text, 'lang': 'unsure'})
    for u in utts:
        best, ov_best = None, 0
        for code, s, e, v in langs:
            if code != u['code']:
                continue
            ov = min(u['end'], e) - max(u['start'], s)
            if ov > ov_best:
                best, ov_best = v, ov
        if best:
            u['lang'] = best
    utts.sort(key=lambda u: (u['start'], u['end']))
    return utts


def stratum_text(utts, roles):
    """Concatenate, in time order, exactly as the pipeline builds a transcript."""
    return ' '.join(t for u in utts if u['role'] in roles
                    for t in clean_text(u['text']).split())


def language_text(utts, lang):
    """Words attributed to one language stream. Braced spans are English."""
    out = []
    for u in utts:
        if u['role'] not in STRATA['OVERALL']:
            continue
        pos, text = 0, u['text']
        for m in BRACE_RE.finditer(text):
            if u['lang'] == lang:
                out += clean_text(text[pos:m.start()]).split()
            if lang == 'en':
                out += clean_text(m.group(1)).split()
            pos = m.end()
        if u['lang'] == lang:
            out += clean_text(text[pos:]).split()
    return ' '.join(out)


def sym(a, b):
    """Symmetric WER: total edits over the mean of the two lengths."""
    if not a.split() and not b.split():
        return None
    m = compute_measures(a, b)
    edits = m['substitutions'] + m['deletions'] + m['insertions']
    denom = (len(a.split()) + len(b.split())) / 2
    return edits / denom if denom else None


def icc11(groups):
    """ICC(1,1), one-way random, single measure, absolute agreement.
    Matches the model named in irr_results.csv."""
    groups = [g for g in groups if len(g) >= 2]
    m = len(groups)
    n = sum(len(g) for g in groups)
    if m < 2:
        return None
    grand = sum(v for g in groups for v in g) / n
    msb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups) / (m - 1)
    msw = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g) / (n - m)
    k0 = (n - sum(len(g) ** 2 for g in groups) / n) / (m - 1)
    den = msb + (k0 - 1) * msw
    return (msb - msw) / den if den else None, k0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--irr-dir', default='irr_transcripts')
    ap.add_argument('--dbt', choices=['adult', 'drop'], default='adult',
                    help="'adult' replicates irr_counts.csv; 'drop' excludes DBT")
    ap.add_argument('--out', default='irr_symmetric_wer.csv')
    args = ap.parse_args()

    data = defaultdict(dict)
    for name in sorted(os.listdir(args.irr_dir)):
        if name.endswith('.txt'):
            clip, _, rater = name[:-4].rpartition('-')
            data[clip][rater] = parse(os.path.join(args.irr_dir, name), args.dbt)
    clips = sorted(data)

    # ---- 1. word counts: validation against irr_counts.csv --------------
    print('=== WORD COUNTS (validate against irr_counts.csv) ===')
    print(f"{'clip':26s} {'coder':5s} {'KCHI':>6s} {'ADULT':>6s} {'OCH':>6s} {'TOTAL':>6s}")
    counts = defaultdict(dict)
    for clip in clips:
        for r in sorted(data[clip]):
            c = {s: len(stratum_text(data[clip][r], roles).split())
                 for s, roles in STRATA.items()}
            counts[clip][r] = c
            print(f'{clip:26s} {r:5s} {c["KCHI"]:6d} {c["ADULT"]:6d} '
                  f'{c["OCH"]:6d} {c["KCHI"]+c["ADULT"]+c["OCH"]:6d}')

    print('\n=== ICC(1,1) (compare to irr_results.csv) ===')
    for label, key in (('adult_wc', 'ADULT'), ('child_wc', 'KCHI')):
        groups = [[counts[c][r][key] for r in sorted(counts[c])] for c in clips]
        res = icc11(groups)
        if res:
            icc, k0 = res
            print(f'  {label:10s} ICC = {icc:.4f}   k0 = {k0:.4f}   '
                  f'm = {len(clips)}   n = {sum(len(g) for g in groups)}')

    # ---- 2. pairwise WER, the paper's strata ---------------------------
    rows = []
    for clip in clips:
        raters = sorted(data[clip])
        for stratum, roles in STRATA.items():
            txt = {r: stratum_text(data[clip][r], roles) for r in raters}
            for a, b in combinations(raters, 2):
                ta, tb = txt[a], txt[b]
                if not ta.split() or not tb.split():
                    rows.append({'clip_id': clip, 'stratum': stratum, 'rater_a': a,
                                 'rater_b': b, 'n_words_a': len(ta.split()),
                                 'n_words_b': len(tb.split()), 'wer_a_ref': '',
                                 'wer_b_ref': '', 'sym_wer': '', 'sub': '', 'del': '',
                                 'ins': '', 'degenerate': 'yes'})
                    continue
                m = compute_measures(ta, tb)
                rows.append({
                    'clip_id': clip, 'stratum': stratum, 'rater_a': a, 'rater_b': b,
                    'n_words_a': len(ta.split()), 'n_words_b': len(tb.split()),
                    'wer_a_ref': round(m['wer'], 4),
                    'wer_b_ref': round(compute_measures(tb, ta)['wer'], 4),
                    'sym_wer': round(sym(ta, tb), 4),
                    'sub': m['substitutions'], 'del': m['deletions'],
                    'ins': m['insertions'], 'degenerate': ''})
        for lang in ('es', 'en'):
            txt = {r: language_text(data[clip][r], lang) for r in raters}
            for a, b in combinations(raters, 2):
                ta, tb = txt[a], txt[b]
                if not ta.split() or not tb.split():
                    continue
                m = compute_measures(ta, tb)
                rows.append({
                    'clip_id': clip, 'stratum': f'LANG_{lang}', 'rater_a': a,
                    'rater_b': b, 'n_words_a': len(ta.split()),
                    'n_words_b': len(tb.split()),
                    'wer_a_ref': round(m['wer'], 4),
                    'wer_b_ref': round(compute_measures(tb, ta)['wer'], 4),
                    'sym_wer': round(sym(ta, tb), 4),
                    'sub': m['substitutions'], 'del': m['deletions'],
                    'ins': m['insertions'], 'degenerate': ''})

    with open(args.out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 3. summary in the paper's format -------------------------------
    print('\n=== HUMAN-HUMAN WER, paper strata (clean_text + jiwer) ===')
    print(f"{'stratum':10s} {'clip-median':>12s} {'clip-mean':>10s} "
          f"{'pooled':>8s} {'pairs':>6s}")
    for stratum in ('KCHI', 'ADULT', 'OVERALL', 'OCH', 'LANG_es', 'LANG_en'):
        per_clip = []
        tot_e = tot_n = 0
        for clip in clips:
            v = [r['sym_wer'] for r in rows
                 if r['clip_id'] == clip and r['stratum'] == stratum and r['sym_wer'] != '']
            if v:
                per_clip.append(sum(v) / len(v))
            for r in rows:
                if r['clip_id'] == clip and r['stratum'] == stratum and r['sym_wer'] != '':
                    tot_e += r['sub'] + r['del'] + r['ins']
                    tot_n += (r['n_words_a'] + r['n_words_b']) / 2
        if not per_clip:
            continue
        per_clip.sort()
        med = per_clip[len(per_clip) // 2] if len(per_clip) % 2 else \
            (per_clip[len(per_clip) // 2 - 1] + per_clip[len(per_clip) // 2]) / 2
        n_pairs = sum(1 for r in rows if r['stratum'] == stratum and r['sym_wer'] != '')
        print(f'{stratum:10s} {med:12.3f} {sum(per_clip)/len(per_clip):10.3f} '
              f'{tot_e/tot_n:8.3f} {n_pairs:6d}')
    print(f'\nwrote {args.out}   (DBT mode: {args.dbt})')


if __name__ == '__main__':
    main()
