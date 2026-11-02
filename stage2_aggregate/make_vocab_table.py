#!/usr/bin/env python3
"""
Vocabulary (type/token) size per clip, per side (ref/hyp), per language.

    python3 make_vocab_table.py --ref-dir <human_root> --hyp-dir <vtc2_root> --inspect
    python3 make_vocab_table.py --ref-dir <human_root> --hyp-dir <vtc2_root> \
            --ref-tag paren --hyp-tag trailing --out vocab_by_clip.csv
"""
import argparse
import csv
from collections import defaultdict

from transcript_tags import (
    discover_clips, parse_line, clean_tokens, normalize_type,
    canonical_key, anonymize_recordings,
)

SUFFIX = {'ref': '_tagged.txt', 'hyp': '_tagged_langdetect.txt'}


def collect_side(root, tag_mode, suffix):
    """dict[(recording, clip_label)][language] -> {'token_count': int, 'types': set()}"""
    data = defaultdict(lambda: defaultdict(lambda: {'token_count': 0, 'types': set()}))
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
                tokens = clean_tokens(text)
                if not tokens:
                    continue
                bucket = data[(recording, clip)][language]
                bucket['token_count'] += len(tokens)
                for tok in tokens:
                    norm = normalize_type(tok)
                    if norm:
                        bucket['types'].add(norm)
    return data, clip_keys


def inspect_side(root, tag_mode, suffix, label, n=8):
    print(f"\n=== {label}: sample from first file under {root} (tag_mode={tag_mode}) ===")
    for recording, clip, path in discover_clips(root, suffix):
        print(f"file: {path}")
        shown = 0
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                if shown >= n:
                    break
                parsed = parse_line(line, tag_mode)
                print(f"RAW:    {line}")
                if parsed:
                    speaker, language, text = parsed
                    tokens = clean_tokens(text)
                    print(f"PARSED: lang={language} speaker={speaker} n_tokens={len(tokens)}")
                else:
                    print("PARSED: None")
                shown += 1
        break
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--hyp-dir', required=True)
    ap.add_argument('--ref-tag', choices=['paren', 'trailing'], default='paren')
    ap.add_argument('--hyp-tag', choices=['paren', 'trailing'], default='trailing')
    ap.add_argument('--out', default='vocab_by_clip.csv')
    ap.add_argument('--inspect', action='store_true')
    args = ap.parse_args()

    if args.inspect:
        inspect_side(args.ref_dir, args.ref_tag, SUFFIX['ref'], 'REF')
        inspect_side(args.hyp_dir, args.hyp_tag, SUFFIX['hyp'], 'HYP')
        return

    ref_data, ref_keys = collect_side(args.ref_dir, args.ref_tag, SUFFIX['ref'])
    hyp_data, hyp_keys = collect_side(args.hyp_dir, args.hyp_tag, SUFFIX['hyp'])

    all_recordings = {rec for rec, _ in ref_keys | hyp_keys}
    rec_map = anonymize_recordings(all_recordings)

    rows = []
    all_keys = sorted(ref_keys | hyp_keys)
    for recording, clip in all_keys:
        rec_id = rec_map[recording]
        clip_id = clip
        for side, data in (('ref', ref_data), ('hyp', hyp_data)):
            langs = data.get((recording, clip), {})
            for language, bucket in sorted(langs.items()):
                rows.append({
                    'recording': rec_id,
                    'clip': clip_id,
                    'side': side,
                    'language': language,
                    'token_count': bucket['token_count'],
                    'type_count': len(bucket['types']),
                })

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['recording', 'clip', 'side', 'language', 'token_count', 'type_count']
        )
        writer.writeheader()
        writer.writerows(rows)

    totals = defaultdict(int)
    for r in rows:
        totals[(r['side'], r['language'])] += r['token_count']

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"{len(all_keys)} clips, {len(rec_map)} recordings")
    print("\nToken-count totals by side/language (sanity-check against Stage 2 workbook):")
    for (side, language), token_count in sorted(totals.items()):
        print(f"  {side:4s} {language:10s} tokens={token_count}")


if __name__ == '__main__':
    main()
