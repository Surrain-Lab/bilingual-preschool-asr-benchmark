#!/usr/bin/env python3
"""
Language-tag agreement between annotators, the layer missing from
irr_interannotator.py.

    python3 irr_language_agreement.py --irr-dir irr_transcripts

Language is annotated two ways in these transcripts:
  * a time-aligned dependent tier  lang@<SPEAKER>  with values spa/eng/both/unsure
  * CHAT curly braces marking embedded English inside a Spanish matrix,
    e.g.  "Mama y {ranch} y ketchup."   (methods.tex, annotation conventions)

A token's language is therefore: English if it sits inside braces, otherwise
the lang@ value of the utterance carrying it.

Two units of analysis, both Krippendorff's alpha so they handle 2-4 raters:

  lang_utterance  unit = a time-overlap cluster of utterances; a rater's value
                  is the language they assigned there (longest-overlapping
                  utterance wins when they segmented it into several).

  lang_token      unit = (cluster, word type) for word types that at least two
                  raters both transcribed in that cluster; a rater's value is
                  the language they assigned that word. This is agreement about
                  language *conditional on having transcribed the same word*,
                  which is the only way a token-level comparison is defined
                  when coders produce different token sequences.

Each is reported over all four labels (es/en/both/unsure) and again restricted
to units where every rater committed to es or en, which is the quantity the
percent-Spanish measure actually depends on.
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

from transcript_tags import clean_tokens, wer_clean_tokens
from irr_interannotator import (
    role_of, is_speech, coincidence, alpha_from_coincidence, pooled_alpha,
    bootstrap_ci, NOMINAL, SPEECH_ROLES, CHAT_ZERO, XXX_RE,
)

LANG_MAP = {'spa': 'es', 'eng': 'en', 'both': 'both', 'unsure': 'unsure'}
BRACE_RE = re.compile(r'\{([^}]*)\}')


def norm_tokens(text):
    return [t for t in wer_clean_tokens(clean_tokens(text)) if t != CHAT_ZERO]


def token_languages(text, utt_lang):
    """(token, language) pairs. Braced spans are English regardless of the
    utterance-level tag; everything else inherits the utterance tag."""
    out, pos = [], 0
    for m in BRACE_RE.finditer(text):
        for t in norm_tokens(text[pos:m.start()]):
            out.append((t, utt_lang))
        for t in norm_tokens(m.group(1)):
            out.append((t, 'en'))
        pos = m.end()
    for t in norm_tokens(text[pos:]):
        out.append((t, utt_lang))
    return out


class LUtt:
    __slots__ = ('role', 'code', 'start', 'end', 'text', 'tokens', 'lang',
                 'tok_langs')

    def __init__(self, role, code, start, end, text):
        self.role, self.code = role, code
        self.start, self.end = start, end
        self.text = text
        self.tokens = norm_tokens(text)
        self.lang = 'unsure'
        self.tok_langs = []


def parse(path):
    mains, langs = [], []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 6:
                continue
            tier, code, start, end = parts[0], (parts[1] or parts[0]).strip(), parts[2], parts[3]
            text = '\t'.join(parts[5:]).strip()
            try:
                start, end = int(start), int(end)
            except ValueError:
                continue
            if end <= start:
                continue
            if tier.startswith('lang@'):
                v = LANG_MAP.get(text.strip().lower())
                if v:
                    langs.append((code, start, end, v))
                continue
            if '@' in tier:
                continue
            role = role_of(code)
            u = LUtt(role, code, start, end, text)
            if not is_speech(u.tokens, text, end - start):
                continue
            mains.append(u)

    for u in mains:
        best, best_ov = None, 0
        for code, s, e, v in langs:
            if code != u.code:
                continue
            ov = min(u.end, e) - max(u.start, s)
            if ov > best_ov:
                best, best_ov = v, ov
        if best:
            u.lang = best
        u.tok_langs = token_languages(u.text, u.lang)
    mains.sort(key=lambda u: (u.start, u.end))
    return mains


def cluster(by_rater):
    items = []
    for rater, utts in by_rater.items():
        for u in utts:
            if u.role in SPEECH_ROLES and u.tokens:
                items.append((u.start, u.end, rater, u))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    groups, cur, cur_end = [], [items[0]], items[0][1]
    for it in items[1:]:
        if it[0] < cur_end:
            cur.append(it)
            cur_end = max(cur_end, it[1])
        else:
            groups.append(cur)
            cur, cur_end = [it], it[1]
    groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--irr-dir', default='irr_transcripts')
    ap.add_argument('--csv-dir', default='irr_interannotator_csv')
    args = ap.parse_args()

    data = defaultdict(dict)
    for name in sorted(os.listdir(args.irr_dir)):
        if not name.endswith('.txt'):
            continue
        clip, _, rater = name[:-4].rpartition('-')
        data[clip][rater] = parse(os.path.join(args.irr_dir, name))
    clips = sorted(data)

    layers = {k: {} for k in ('lang_utterance', 'lang_token',
                              'lang_utterance_esen', 'lang_token_esen')}
    units_n = {k: {} for k in layers}
    label_counts = Counter()
    diag = []

    for clip in clips:
        u_units, t_units = [], []
        for grp in cluster(data[clip]):
            # utterance-level: longest-overlapping utterance per rater
            per_rater_lang, per_rater_tok = {}, defaultdict(dict)
            best_dur = defaultdict(int)
            for s, e, rater, u in grp:
                if (e - s) > best_dur[rater]:
                    best_dur[rater] = e - s
                    per_rater_lang[rater] = u.lang
                for tok, lg in u.tok_langs:
                    per_rater_tok[rater].setdefault(tok, lg)
            if len(per_rater_lang) >= 2:
                u_units.append(list(per_rater_lang.values()))
                label_counts.update(per_rater_lang.values())
            types = set()
            for d in per_rater_tok.values():
                types.update(d.keys())
            for tok in sorted(types):
                vals = [d[tok] for d in per_rater_tok.values() if tok in d]
                if len(vals) >= 2:
                    t_units.append(vals)

        for key, units in (('lang_utterance', u_units), ('lang_token', t_units)):
            layers[key][clip], units_n[key][clip] = coincidence(units)
            esen = [v for v in units if all(x in ('es', 'en') for x in v)]
            layers[f'{key}_esen'][clip], units_n[f'{key}_esen'][clip] = coincidence(esen)

        diag.append({'clip_id': clip, 'n_raters': len(data[clip]),
                     'n_utterance_units': len(u_units),
                     'n_token_units': len(t_units)})

    rows = []
    for layer in ('lang_utterance', 'lang_utterance_esen',
                  'lang_token', 'lang_token_esen'):
        for clip in clips:
            rows.append({'layer': layer, 'clip_id': clip,
                         'n_raters': len(data[clip]),
                         'n_units': units_n[layer][clip],
                         'alpha': round(alpha_from_coincidence(layers[layer][clip], NOMINAL), 4),
                         'ci_lower': '', 'ci_upper': ''})
        lo, hi = bootstrap_ci(layers[layer], NOMINAL)
        rows.append({'layer': layer, 'clip_id': 'POOLED', 'n_raters': '',
                     'n_units': sum(units_n[layer].values()),
                     'alpha': round(pooled_alpha(layers[layer], NOMINAL), 4),
                     'ci_lower': lo, 'ci_upper': hi})

    os.makedirs(args.csv_dir, exist_ok=True)
    for name, data_rows in (('language_alpha', rows), ('language_diagnostics', diag)):
        with open(os.path.join(args.csv_dir, f'{name}.csv'), 'w', newline='',
                  encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
            w.writeheader()
            w.writerows(data_rows)

    print('utterance-level label distribution:', dict(label_counts.most_common()))
    print()
    print(f"{'layer':22s} {'alpha':>8s} {'95% CI':>18s} {'units':>7s}")
    for r in rows:
        if r['clip_id'] == 'POOLED':
            ci = f"[{r['ci_lower']}, {r['ci_upper']}]"
            print(f"{r['layer']:22s} {r['alpha']:>8} {ci:>18s} {r['n_units']:>7}")
    print()
    print('per clip (alpha):')
    print(f"{'clip':26s} {'k':>2s} " + ' '.join(f'{l[-12:]:>13s}' for l in
          ('lang_utterance', 'lang_utterance_esen', 'lang_token', 'lang_token_esen')))
    for clip in clips:
        cells = []
        for layer in ('lang_utterance', 'lang_utterance_esen', 'lang_token', 'lang_token_esen'):
            v = [r for r in rows if r['layer'] == layer and r['clip_id'] == clip][0]
            cells.append(f"{v['alpha']:>13}")
        print(f"{clip:26s} {len(data[clip]):>2d} " + ' '.join(cells))
    print(f"\nwrote {args.csv_dir}/language_alpha.csv")


if __name__ == '__main__':
    main()
