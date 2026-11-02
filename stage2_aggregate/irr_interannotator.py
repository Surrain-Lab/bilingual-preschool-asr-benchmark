#!/usr/bin/env python3
"""
Inter-annotator reliability for the IRR transcripts (2-4 annotators per clip,
no gold standard).

    python3 irr_interannotator.py --irr-dir irr_transcripts --out irr_interannotator.xlsx

Every measure here is symmetric: no annotator is treated as the reference.

  1. pairwise word-level distance  -- descriptive. For each rater pair, the
     Levenshtein distance between their word sequences, normalized two ways:
         sym_wer = 2*lev / (len_a + len_b)      symmetric, WER-comparable
         norm_ed = lev / max(len_a, len_b)      bounded [0,1], a true metric
     Reported overall and split by speaker role.

  2. Krippendorff's alpha -- chance-corrected, handles 2-4 raters and missing
     raters natively. Three layers, because they answer different questions:
         detection   100 ms bin, binary speech / no-speech, whole clip
         speaker     100 ms bin, which role is talking, restricted to bins
                     where >=1 rater heard speech
         content     time-overlap utterance cluster, delta = normalized edit
                     distance between the raters' word sequences
         intelligib. 100 ms bin, transcribed vs xxx, restricted to bins where
                     >=2 raters heard speech

Text normalization is transcript_tags.clean_tokens + wer_clean_tokens, i.e.
bit-identical to the pipeline's WER scoring, so these numbers sit on the same
scale as the human-vs-ASR WER tables.
"""
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations

from transcript_tags import clean_tokens, wer_clean_tokens

BIN_MS = 100
BOOTSTRAP = 2000
SEED = 20260823

# --- speaker code -> analysis role ---------------------------------------
# CHI target child; FA/MA female/male adult; UC/MC/FC other child;
# UA/ME other adult. EE1 (electronic media), DBT, UI1, FI1 carry no lexical
# content and span most of the clip, so they are not speech for any purpose.
ROLE_PATTERNS = [
    (re.compile(r'^CHI$'), 'KCHI'),
    (re.compile(r'^(UC|MC|FC)\d*$'), 'OCH'),
    (re.compile(r'^FA\d*$'), 'FEM'),
    (re.compile(r'^MA\d*$'), 'MAL'),
    (re.compile(r'^(UA|ME)\d*$'), 'ADULT_OTHER'),
]
# DBT ("doubtful"), UI1, FI1, EE1: speaker not established by the annotator.
# One annotator put ALL of a clip's speech on DBT, so these cannot be discarded
# -- they become UNSPEC, which is speech with an undetermined speaker.
# OCH is excluded throughout, matching the benchmark (methods.tex, Annotation
# procedure: "we did not include speech labeled as OCH in our benchmark analysis").
SPEECH_ROLES = ['KCHI', 'FEM', 'MAL', 'ADULT_OTHER', 'UNSPEC']
# Coarse speaker view = the paper's own strata: key child vs adult.
COARSE = {'KCHI': 'KCHI', 'FEM': 'ADULT', 'MAL': 'ADULT',
          'ADULT_OTHER': 'ADULT', 'UNSPEC': 'ADULT', 'none': 'none'}
MAX_UTT_MS = 120_000      # above this it is a clip-level environment marker
CHAT_ZERO = '0'           # CHAT "no verbalization"; not a word
SCOPES = {
    'OVERALL': SPEECH_ROLES,
    'KCHI': ['KCHI'],
    'ADULT': ['FEM', 'MAL', 'ADULT_OTHER', 'UNSPEC'],
}
XXX_RE = re.compile(r'\bxxx\b', re.IGNORECASE)


def role_of(code):
    for pat, role in ROLE_PATTERNS:
        if pat.match(code):
            return role
    return 'UNSPEC'


def is_speech(tokens, text, dur_ms):
    """An utterance is speech if it carries words or an unintelligible marker.
    A bare '0.' is CHAT's no-verbalization marker, not speech -- that is what
    EE1 (media), whole-clip DBT spans and UI1/FI1 markers all are."""
    if dur_ms > MAX_UTT_MS:
        return False
    return bool(tokens) or bool(XXX_RE.search(text))


# ------------------------------------------------------------------ parsing

class Utt:
    __slots__ = ('role', 'code', 'start', 'end', 'text', 'tokens', 'has_xxx')

    def __init__(self, role, code, start, end, text):
        self.role, self.code = role, code
        self.start, self.end = start, end
        self.text = text
        self.tokens = [t for t in wer_clean_tokens(clean_tokens(text))
                       if t != CHAT_ZERO]
        self.has_xxx = bool(XXX_RE.search(text))


def parse_clan(path):
    """Main-tier utterances only (dependent tiers carry an '@' in column 1)."""
    utts, skipped = [], Counter()
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                skipped['malformed'] += 1
                continue
            tier, code, start, end, _dur = parts[:5]
            text = '\t'.join(parts[5:]).strip()
            if '@' in tier:
                continue                       # dependent tier
            code = (code or tier).strip()
            role = role_of(code)
            if role is None:
                skipped[code] += 1
                continue
            try:
                start, end = int(start), int(end)
            except ValueError:
                skipped['bad_time'] += 1
                continue
            if end <= start:
                skipped['zero_len'] += 1
                continue
            u = Utt(role, code, start, end, text)
            if not is_speech(u.tokens, text, end - start):
                skipped[f'{code}:no_speech_content'] += 1
                continue
            utts.append(u)
    utts.sort(key=lambda u: (u.start, u.end))
    return utts, skipped


def load(irr_dir):
    """{clip: {rater: [Utt, ...]}}"""
    data = defaultdict(dict)
    skipped = Counter()
    for name in sorted(os.listdir(irr_dir)):
        if not name.endswith('.txt'):
            continue
        stem = name[:-4]
        clip, _, rater = stem.rpartition('-')
        utts, sk = parse_clan(os.path.join(irr_dir, name))
        data[clip][rater] = utts
        skipped.update(sk)
    return data, skipped


# ------------------------------------------------------------- edit distance

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_ed(a, b):
    """Levenshtein / max(len) -- bounded [0,1], symmetric, a true metric."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


# --------------------------------------------------- Krippendorff's alpha

def coincidence(units):
    """units: list of lists of values (one per rater present for that unit).

    Returns the coincidence Counter keyed (value, value). Units carrying fewer
    than two values are dropped, per Krippendorff.
    """
    o = Counter()
    used = 0
    for values in units:
        m = len(values)
        if m < 2:
            continue
        used += 1
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(values[i], values[j])] += w
    return o, used


def alpha_from_coincidence(o, delta):
    """delta(c, k) -> difference in [0, inf); 0 iff identical."""
    if not o:
        return float('nan')
    marg = Counter()
    for (c, k), v in o.items():
        marg[c] += v
    n = sum(marg.values())
    if n <= 1:
        return float('nan')
    cats = list(marg)
    d_obs = sum(v * delta(c, k) for (c, k), v in o.items())
    d_exp = 0.0
    for c in cats:
        for k in cats:
            if c != k:
                d_exp += marg[c] * marg[k] * delta(c, k)
    d_exp /= (n - 1)
    if d_exp == 0:
        return float('nan')
    return 1.0 - d_obs / d_exp


NOMINAL = lambda c, k: 0.0 if c == k else 1.0        # noqa: E731


def pooled_alpha(per_clip_coincidence, delta, clips=None):
    total = Counter()
    for clip in (clips if clips is not None else per_clip_coincidence):
        total.update(per_clip_coincidence[clip])
    return alpha_from_coincidence(total, delta)


def bootstrap_ci(per_clip_coincidence, delta, n_iter=BOOTSTRAP, seed=SEED):
    """Resample clips with replacement. Coincidence matrices are additive over
    clips, so this is cheap and exact for the resampled data."""
    clips = list(per_clip_coincidence)
    if len(clips) < 2:
        return ('', '')
    rng = random.Random(seed)
    vals = []
    for _ in range(n_iter):
        draw = [rng.choice(clips) for _ in clips]
        a = pooled_alpha(per_clip_coincidence, delta, draw)
        if a == a:                                    # not NaN
            vals.append(a)
    if len(vals) < 20:
        return ('', '')
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return (round(lo, 4), round(hi, 4))


# ------------------------------------------------------------------- layers

def bin_labels(utts, n_bins, roles):
    """Per-bin role label for one rater; 'none' where they marked no speech.
    When speakers overlap in a bin, the one covering more of it wins."""
    best = [(0, 'none')] * n_bins
    for u in utts:
        if u.role not in roles:
            continue
        b0, b1 = u.start // BIN_MS, min((u.end - 1) // BIN_MS, n_bins - 1)
        for b in range(b0, b1 + 1):
            lo, hi = b * BIN_MS, (b + 1) * BIN_MS
            ov = min(u.end, hi) - max(u.start, lo)
            if ov > best[b][0]:
                best[b] = (ov, u.role)
    return [r for _ov, r in best]


def bin_intelligibility(utts, n_bins, roles):
    """Per-bin 'int' / 'unint' / None(no speech). An utterance containing xxx
    marks its bins unintelligible; lexical content marks them intelligible.
    Where both apply in one bin, the longer-overlapping utterance wins."""
    best = [(0, None)] * n_bins
    for u in utts:
        if u.role not in roles:
            continue
        if u.has_xxx:
            lab = 'unint'
        elif u.tokens:
            lab = 'int'
        else:
            continue                                   # e.g. bare "0."
        b0, b1 = u.start // BIN_MS, min((u.end - 1) // BIN_MS, n_bins - 1)
        for b in range(b0, b1 + 1):
            lo, hi = b * BIN_MS, (b + 1) * BIN_MS
            ov = min(u.end, hi) - max(u.start, lo)
            if ov > best[b][0]:
                best[b] = (ov, lab)
    return [lab for _ov, lab in best]


def cluster_utterances(by_rater, roles, min_overlap_ms=1):
    """Single-linkage clusters of temporally overlapping utterances across
    raters. Each cluster is one alpha unit; a rater's value is their tokens in
    that cluster, concatenated in time order."""
    items = []
    for rater, utts in by_rater.items():
        for u in utts:
            if u.role in roles and u.tokens:
                items.append((u.start, u.end, rater, u))
    if not items:
        return []
    items.sort()
    clusters, cur, cur_end = [], [items[0]], items[0][1]
    for it in items[1:]:
        if it[0] < cur_end - (min_overlap_ms - 1):
            cur.append(it)
            cur_end = max(cur_end, it[1])
        else:
            clusters.append(cur)
            cur, cur_end = [it], it[1]
    clusters.append(cur)

    out = []
    for cl in clusters:
        per_rater = defaultdict(list)
        for start, _end, rater, u in sorted(cl):
            per_rater[rater].extend(u.tokens)
        span = (min(c[0] for c in cl), max(c[1] for c in cl))
        out.append((span, {r: tuple(t) for r, t in per_rater.items()}))
    return out


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--irr-dir', default='irr_transcripts')
    ap.add_argument('--out', default='irr_interannotator.xlsx')
    ap.add_argument('--csv-dir', default='irr_interannotator_csv')
    args = ap.parse_args()

    data, skipped = load(args.irr_dir)
    clips = sorted(data)

    # ---------------------------------------------------------- layer 1: WER
    pair_rows = []
    for clip in clips:
        raters = sorted(data[clip])
        n_bins_end = max((u.end for r in raters for u in data[clip][r]),
                         default=0)
        for scope, roles in SCOPES.items():
            seqs = {r: [t for u in data[clip][r] if u.role in roles
                        for t in u.tokens] for r in raters}
            for a, b in combinations(raters, 2):
                sa, sb = seqs[a], seqs[b]
                lev = levenshtein(sa, sb)
                denom = len(sa) + len(sb)
                pair_rows.append({
                    'clip_id': clip, 'scope': scope,
                    'rater_a': a, 'rater_b': b,
                    'n_words_a': len(sa), 'n_words_b': len(sb),
                    'lev_distance': lev,
                    'sym_wer': round(2 * lev / denom, 4) if denom else '',
                    'norm_edit_dist': round(norm_ed(sa, sb), 4),
                    'wer_a_as_ref': round(lev / len(sa), 4) if sa else '',
                    'wer_b_as_ref': round(lev / len(sb), 4) if sb else '',
                    'degenerate': 'yes' if (not sa or not sb) else '',
                    'clip_end_ms': n_bins_end,
                })

    # -------------------------------------------------------- layers 2-4: alpha
    layers = {}          # layer -> {clip: coincidence Counter}
    layer_units = {}     # layer -> {clip: n_units_used}
    for key in ('detection', 'speaker', 'speaker_coarse',
                'intelligibility', 'content'):
        layers[key] = {}
        layer_units[key] = {}

    content_values = set()
    cluster_diag = []

    for clip in clips:
        raters = sorted(data[clip])
        end_ms = max((u.end for r in raters for u in data[clip][r]), default=0)
        n_bins = max(1, -(-end_ms // BIN_MS))
        roles = SPEECH_ROLES

        labs = {r: bin_labels(data[clip][r], n_bins, roles) for r in raters}

        det_units, spk_units, spk_coarse = [], [], []
        for b in range(n_bins):
            vals = [labs[r][b] for r in raters]
            det_units.append(['speech' if v != 'none' else 'none' for v in vals])
            if any(v != 'none' for v in vals):
                spk_units.append(vals)
                spk_coarse.append([COARSE[v] for v in vals])
        layers['detection'][clip], layer_units['detection'][clip] = coincidence(det_units)
        layers['speaker'][clip], layer_units['speaker'][clip] = coincidence(spk_units)
        layers['speaker_coarse'][clip], layer_units['speaker_coarse'][clip] = \
            coincidence(spk_coarse)

        ilabs = {r: bin_intelligibility(data[clip][r], n_bins, roles) for r in raters}
        int_units = []
        for b in range(n_bins):
            vals = [ilabs[r][b] for r in raters if ilabs[r][b] is not None]
            if len(vals) >= 2:
                int_units.append(vals)
        layers['intelligibility'][clip], layer_units['intelligibility'][clip] = \
            coincidence(int_units)

        clusters = cluster_utterances(data[clip], roles)
        con_units = []
        for span, per_rater in clusters:
            vals = list(per_rater.values())
            if len(vals) >= 2:
                con_units.append(vals)
                content_values.update(vals)
        layers['content'][clip], layer_units['content'][clip] = coincidence(con_units)
        cluster_diag.append({
            'clip_id': clip, 'n_raters': len(raters),
            'n_clusters': len(clusters),
            'n_clusters_multi_rater': len(con_units),
            'max_cluster_span_ms': max((s[1] - s[0] for s, _ in clusters), default=0),
            'median_cluster_span_ms': sorted(s[1] - s[0] for s, _ in clusters)[len(clusters) // 2] if clusters else 0,
        })

    # content delta: memoized normalized edit distance over token tuples
    _cache = {}

    def content_delta(c, k):
        if c == k:
            return 0.0
        key = (c, k) if c < k else (k, c)
        v = _cache.get(key)
        if v is None:
            v = norm_ed(list(c), list(k))
            _cache[key] = v
        return v

    alpha_rows = []
    for layer, delta in (('detection', NOMINAL), ('speaker', NOMINAL),
                         ('speaker_coarse', NOMINAL),
                         ('intelligibility', NOMINAL), ('content', content_delta)):
        for clip in clips:
            alpha_rows.append({
                'layer': layer, 'clip_id': clip,
                'n_raters': len(data[clip]),
                'n_units': layer_units[layer][clip],
                'alpha': round(alpha_from_coincidence(layers[layer][clip], delta), 4),
                'ci_lower': '', 'ci_upper': '',
            })
        lo, hi = bootstrap_ci(layers[layer], delta)
        alpha_rows.append({
            'layer': layer, 'clip_id': 'POOLED',
            'n_raters': '', 'n_units': sum(layer_units[layer].values()),
            'alpha': round(pooled_alpha(layers[layer], delta), 4),
            'ci_lower': lo, 'ci_upper': hi,
        })

    # ------------------------------------------------------------- summaries
    by_clip_scope = defaultdict(list)
    for r in pair_rows:
        if r['sym_wer'] != '' and not r['degenerate']:
            by_clip_scope[(r['clip_id'], r['scope'])].append(r['sym_wer'])
    clip_rows = []
    for (clip, scope), vals in sorted(by_clip_scope.items()):
        clip_rows.append({
            'clip_id': clip, 'scope': scope,
            'n_raters': len(data[clip]), 'n_pairs': len(vals),
            'mean_sym_wer': round(sum(vals) / len(vals), 4),
            'min_sym_wer': round(min(vals), 4),
            'max_sym_wer': round(max(vals), 4),
        })

    scope_rows = []
    for scope in SCOPES:
        per_clip = [r['mean_sym_wer'] for r in clip_rows if r['scope'] == scope]
        allp = [r['sym_wer'] for r in pair_rows
                if r['scope'] == scope and r['sym_wer'] != '' and not r['degenerate']]
        n_degen = sum(1 for r in pair_rows
                      if r['scope'] == scope and r['degenerate'])
        scope_rows.append({
            'scope': scope, 'n_clips': len(per_clip), 'n_pairs': len(allp),
            'mean_of_clip_means': round(sum(per_clip) / len(per_clip), 4) if per_clip else '',
            'mean_over_all_pairs': round(sum(allp) / len(allp), 4) if allp else '',
            'min_pair': round(min(allp), 4) if allp else '',
            'max_pair': round(max(allp), 4) if allp else '',
            'n_pairs_excluded_degenerate': n_degen,
        })

    rater_rows = []
    agg = defaultdict(list)
    for r in pair_rows:
        if r['scope'] == 'OVERALL' and r['sym_wer'] != '' and not r['degenerate']:
            agg[r['rater_a']].append(r['sym_wer'])
            agg[r['rater_b']].append(r['sym_wer'])
    for rater in sorted(agg):
        v = agg[rater]
        rater_rows.append({
            'rater_id': rater, 'n_clips': sum(1 for c in clips if rater in data[c]),
            'n_pairs': len(v),
            'mean_sym_wer_vs_others': round(sum(v) / len(v), 4),
        })

    # ----------------------------------------------------------------- write
    sheets = [(name, list(rows[0]), rows) for name, rows in (
        ('pairwise_wer', pair_rows), ('wer_by_clip', clip_rows),
        ('wer_by_scope', scope_rows), ('wer_by_rater', rater_rows),
        ('krippendorff_alpha', alpha_rows), ('cluster_diagnostics', cluster_diag),
    ) if rows]

    os.makedirs(args.csv_dir, exist_ok=True)
    for name, cols, rows in sheets:
        with open(os.path.join(args.csv_dir, f'{name}.csv'), 'w',
                  newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    summary = [
        ('DESIGN', ''),
        ('clips', len(clips)),
        ('annotators', ', '.join(sorted({r for c in clips for r in data[c]}))),
        ('transcripts', sum(len(data[c]) for c in clips)),
        ('raters per clip', ', '.join(f'{c}={len(data[c])}' for c in clips)),
        ('speech definition',
         "an utterance counts as speech if it carries words or an xxx marker and "
         "spans <= 120 s. A bare '0.' is CHAT's no-verbalization marker, which is "
         "what EE1 (electronic media), whole-clip DBT spans and UI1/FI1 markers "
         "are -- those are excluded."),
        ('UNSPEC role',
         "DBT / UI1 / FI1 / EE1 and any unrecognised code with real content become "
         "UNSPEC (speech, speaker undetermined). One annotator coded an entire "
         "clip's speech as DBT, so discarding it would have shown that annotator "
         "as having transcribed nothing."),
        ('degenerate pairs',
         'a pair where one annotator has zero words in that scope has no meaningful '
         'WER; those are flagged in pairwise_wer and excluded from the means.'),
        ('lines skipped by code', ', '.join(f'{k}={v}' for k, v in skipped.most_common())),
        ('', ''),
        ('MEASURES', ''),
        ('sym_wer', '2*lev / (len_a + len_b) -- symmetric, so neither annotator '
                    'is the reference. On the same scale as a conventional WER.'),
        ('norm_edit_dist', 'lev / max(len_a, len_b) -- bounded [0,1], a true metric; '
                           'used as the alpha difference function for content.'),
        ('alpha: detection', 'binary speech / no-speech per 100 ms bin, whole clip.'),
        ('alpha: speaker', 'which role is speaking per 100 ms bin, restricted to bins '
                           'where at least one annotator heard speech (including all '
                           'bins would mostly measure agreement about silence).'),
        ('alpha: intelligibility', 'transcribed vs xxx per 100 ms bin, restricted to '
                                   'bins where at least two annotators heard speech.'),
        ('alpha: content', 'unit = single-linkage cluster of temporally overlapping '
                           'utterances; value = that annotator\'s words in the cluster; '
                           'difference = normalized edit distance.'),
        ('CIs', f'{BOOTSTRAP} bootstrap resamples over clips (the unit of independence).'),
        ('normalization', 'transcript_tags.clean_tokens + wer_clean_tokens -- identical '
                          'to the pipeline WER scoring, so these numbers are directly '
                          'comparable to the human-vs-ASR tables.'),
    ]

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
        wb = Workbook()
        wb.remove(wb.active)
        for name, cols, rows in sheets:
            ws = wb.create_sheet(name[:31])
            ws.append(cols)
            for c in ws[1]:
                c.font = Font(bold=True)
            for row in rows:
                ws.append([row.get(c, '') for c in cols])
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(cols, 1):
                ws.column_dimensions[get_column_letter(i)].width = max(len(col) + 2, 12)
        ws = wb.create_sheet('summary')
        ws.column_dimensions['A'].width = 42
        ws.column_dimensions['B'].width = 96
        for k, v in summary:
            ws.append([k, v])
            if v == '':
                ws.cell(ws.max_row, 1).font = Font(bold=True)
        wb.save(args.out)
        wrote = args.out
    except ImportError:
        wrote = f'{args.csv_dir}/ (openpyxl unavailable, CSVs only)'

    for k, v in summary:
        if v != '' or k in ('DESIGN', 'MEASURES'):
            print(f'{k:44s} {v}')
    print('\n--- mean symmetric WER between annotators ---')
    for r in scope_rows:
        print(f"  {r['scope']:14s} clip-mean={r['mean_of_clip_means']}  "
              f"pair-mean={r['mean_over_all_pairs']}  "
              f"range={r['min_pair']}-{r['max_pair']}  (n_pairs={r['n_pairs']})")
    print('\n--- Krippendorff alpha (pooled) ---')
    for r in alpha_rows:
        if r['clip_id'] == 'POOLED':
            print(f"  {r['layer']:16s} alpha={r['alpha']}  "
                  f"95% CI [{r['ci_lower']}, {r['ci_upper']}]  units={r['n_units']}")
    print(f'\nwrote {wrote} and {args.csv_dir}/')


if __name__ == '__main__':
    main()
