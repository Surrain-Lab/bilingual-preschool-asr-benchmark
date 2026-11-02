#!/usr/bin/env python3
"""
Vocabulary (type/token) size per clip, per speaker group, for en/es only.

Speaker tags collapse as:
    MAL, FEM  -> adult
    KCHI      -> kchild
    OCH       -> other_child

    python3 vocab_by_speaker.py --ref-dir <human_root> --hyp-dir <vtc2_root> \
            --ref-tag paren --hyp-tag trailing --out vocab_by_speaker.csv
"""
import argparse
import csv
from collections import defaultdict

from transcript_tags import (
    discover_clips, parse_line, clean_tokens, normalize_type,
    canonical_key, anonymize_recordings,
)

SUFFIX = {'ref': '_tagged.txt', 'hyp': '_tagged_langdetect.txt'}
LANGUAGES = ('en', 'es')
SPEAKER_GROUP = {
    'MAL': 'adult',
    'FEM': 'adult',
    'KCHI': 'kchild',
    'OCH': 'other_child',
}


def collect_side(root, tag_mode, suffix):
    """dict[(recording, clip_label)][speaker_group][language] -> {'token_count': int, 'types': set()}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'token_count': 0, 'types': set()})))
    clip_keys = set()
    unmapped = set()
    for _parent_recording, clip_folder, path in discover_clips(root, suffix):
        recording, clip = canonical_key(clip_folder)
        clip_keys.add((recording, clip))
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                parsed = parse_line(line, tag_mode)
                if not parsed:
                    continue
                speaker, language, text = parsed
                if speaker.startswith('EE'):
                    continue
                if language not in LANGUAGES:
                    continue
                group = SPEAKER_GROUP.get(speaker)
                if group is None:
                    unmapped.add(speaker)
                    continue
                tokens = clean_tokens(text)
                if not tokens:
                    continue
                bucket = data[(recording, clip)][group][language]
                bucket['token_count'] += len(tokens)
                for tok in tokens:
                    norm = normalize_type(tok)
                    if norm:
                        bucket['types'].add(norm)
    if unmapped:
        print(f"WARNING: unmapped speaker tags in {root}: {sorted(unmapped)}")
    return data, clip_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--hyp-dir', required=True)
    ap.add_argument('--ref-tag', choices=['paren', 'trailing'], default='paren')
    ap.add_argument('--hyp-tag', choices=['paren', 'trailing'], default='trailing')
    ap.add_argument('--out', default='vocab_by_speaker.csv')
    args = ap.parse_args()

    ref_data, ref_keys = collect_side(args.ref_dir, args.ref_tag, SUFFIX['ref'])
    hyp_data, hyp_keys = collect_side(args.hyp_dir, args.hyp_tag, SUFFIX['hyp'])

    all_recordings = {rec for rec, _ in ref_keys | hyp_keys}
    rec_map = anonymize_recordings(all_recordings)

    rows = []
    all_keys = sorted(ref_keys | hyp_keys)
    for recording, clip in all_keys:
        rec_id = rec_map[recording]
        clip_id = clip
        ref_speakers = ref_data.get((recording, clip), {})
        hyp_speakers = hyp_data.get((recording, clip), {})
        speakers_present = sorted(set(ref_speakers) | set(hyp_speakers))
        for speaker in speakers_present:
            ref_langs = ref_speakers.get(speaker, {})
            hyp_langs = hyp_speakers.get(speaker, {})
            row = {'recording': rec_id, 'clip': clip_id, 'speaker': speaker}
            for lang in LANGUAGES:
                ref_bucket = ref_langs.get(lang, {'token_count': 0, 'types': set()})
                hyp_bucket = hyp_langs.get(lang, {'token_count': 0, 'types': set()})
                row[f'human_{lang}_types'] = len(ref_bucket['types'])
                row[f'vtc2_{lang}_types'] = len(hyp_bucket['types'])
                row[f'human_{lang}_tokens'] = ref_bucket['token_count']
                row[f'vtc2_{lang}_tokens'] = hyp_bucket['token_count']
            rows.append(row)

    fieldnames = ['recording', 'clip', 'speaker',
                  'human_en_types', 'human_es_types', 'vtc2_en_types', 'vtc2_es_types',
                  'human_en_tokens', 'human_es_tokens', 'vtc2_en_tokens', 'vtc2_es_tokens']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"{len(all_keys)} clips, {len(rec_map)} recordings")


if __name__ == '__main__':
    main()
