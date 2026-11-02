#!/usr/bin/env python3
"""
WER per clip, broken down by language (based on the ref's per-line paren tag).

    python3 wer_by_language.py --ref-dir <human_root> --hyp-dir <vtc2_root> \
            --ref-tag paren --hyp-tag trailing --out wer_by_language.csv
"""
import argparse
import csv
from collections import defaultdict

import jiwer

from transcript_tags import (
    discover_clips, parse_line, clean_tokens, wer_clean_tokens,
    canonical_key, anonymize_recordings,
)

SUFFIX = {'ref': '_tagged.txt', 'hyp': '_tagged_langdetect.txt'}


def collect_side(root, tag_mode, suffix):
    """dict[(recording, clip_label)][language] -> list of cleaned word tokens"""
    data = defaultdict(lambda: defaultdict(list))
    clip_keys = set()
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
                tokens = wer_clean_tokens(clean_tokens(text))
                if tokens:
                    data[(recording, clip)][language].extend(tokens)
    return data, clip_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--hyp-dir', required=True)
    ap.add_argument('--ref-tag', choices=['paren', 'trailing'], default='paren')
    ap.add_argument('--hyp-tag', choices=['paren', 'trailing'], default='trailing')
    ap.add_argument('--out', default='wer_by_language.csv')
    args = ap.parse_args()

    ref_data, ref_keys = collect_side(args.ref_dir, args.ref_tag, SUFFIX['ref'])
    hyp_data, hyp_keys = collect_side(args.hyp_dir, args.hyp_tag, SUFFIX['hyp'])

    all_recordings = {rec for rec, _ in ref_keys | hyp_keys}
    rec_map = anonymize_recordings(all_recordings)

    rows = []
    for recording, clip in sorted(ref_keys):
        rec_id = rec_map[recording]
        clip_id = clip
        ref_langs = ref_data.get((recording, clip), {})
        hyp_langs = hyp_data.get((recording, clip), {})
        for language, ref_tokens in sorted(ref_langs.items()):
            if not ref_tokens:
                continue
            hyp_tokens = hyp_langs.get(language, [])
            ref_str = ' '.join(ref_tokens)
            hyp_str = ' '.join(hyp_tokens) if hyp_tokens else ''

            if hyp_str:
                measures = jiwer.process_words(ref_str, hyp_str)
                wer = measures.wer
                hits, subs = measures.hits, measures.substitutions
                dels, ins = measures.deletions, measures.insertions
            else:
                wer = 1.0
                hits, subs, ins = 0, 0, 0
                dels = len(ref_tokens)

            rows.append({
                'recording': rec_id,
                'clip': clip_id,
                'language': language,
                'ref_words': len(ref_tokens),
                'hyp_words': len(hyp_tokens),
                'wer': round(wer, 4),
                'hits': hits,
                'substitutions': subs,
                'deletions': dels,
                'insertions': ins,
            })

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['recording', 'clip', 'language', 'ref_words', 'hyp_words',
                      'wer', 'hits', 'substitutions', 'deletions', 'insertions']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"{len(ref_keys)} ref clips, {len(rec_map)} recordings")


if __name__ == '__main__':
    main()
