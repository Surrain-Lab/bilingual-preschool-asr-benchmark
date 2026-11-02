# BUGFIX overlap: speaker-tag regexes now accept an optional [OVERLAP] flag
# BUGFIX clip-id: use clip FOLDER name (filename regex truncated _1_NNN stems)
#!/usr/bin/env python3
"""
count_words.py  — NO-LANGUAGE-BARRIER version.

Counts words per speaker AND per language (any language, not just en/es) from the
langdetect-tagged transcripts. For every clip and every recording it reports:
  * total words per speaker (summed across all languages)
  * a breakdown of those words by each detected language
  * the same for speaker groups (ALL CHILDREN = KCHI+OCH, ALL ADULTS = MAL+FEM)
  * overall totals

Outputs (per recording folder):
  <rec>/<clip>/<clip>_language_report.txt          ← readable per-clip report
  <rec>/<rec>_language_report.txt                  ← readable per-recording report
  <rec>/word_counts.csv                            ← long format: clip,speaker,language,words
and a top-level all_recordings_word_counts.csv across everything.

Usage: count_words.py -i <langdetect_dir> -o <output_dir>
"""
import re
import os
import csv
import argparse
from collections import defaultdict

CHILD_SPEAKERS = ('KCHI', 'OCH')
ADULT_SPEAKERS = ('MAL', 'FEM')

# Languages reported individually; every other detected tag is collapsed into
# 'other' (Whisper's per-segment language ID is unreliable on short/noisy child
# speech, so exotic tags like zh/ko/ja are very likely mis-detections).
KEEP_LANGS = ('en', 'es')


def extract_clip_id(filename):
    m = re.match(r'^([A-Z]+-\d+_[a-z]+_\d+)', filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


def parse_line(line):
    """Parse: [ts] (origlang) [FEM]: text here[lang]  ->  (speaker, text, lang) or None."""
    line = line.strip()
    if not line:
        return None
    # the leading (origlang) tag is optional
    m = re.match(r'\[[\d:.\s-]+\]\s*(?:\(\w+\)\s*)?\[([^\]]+)\](?:\s*\[OVERLAP\])?:\s*(.+)$', line)
    if not m:
        return None
    speaker = m.group(1)
    text = m.group(2).strip()
    lm = re.match(r'^(.+?)\s*\[(\w+)\]$', text)   # trailing [lang] appended by langdetect
    if not lm:
        return None
    return speaker, lm.group(1).strip(), lm.group(2).lower()


def count_words(text):
    return len(text.split())


def process_transcript(input_file):
    """Return speaker_stats: {speaker: {language: words}} for one clip."""
    stats = defaultdict(lambda: defaultdict(int))
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            speaker, text, lang = parsed
            if speaker.startswith('EE'):     # exclude electronic/echo speakers, as before
                continue
            bucket = lang if lang in KEEP_LANGS else 'other'   # collapse non-en/es into 'other'
            stats[speaker][bucket] += count_words(text)
    return {spk: dict(langs) for spk, langs in stats.items()}


def merge_stats(dst, src):
    for spk, langs in src.items():
        for lang, n in langs.items():
            dst[spk][lang] += n


def all_languages(stats):
    langs = set()
    for d in stats.values():
        langs.update(d.keys())
    return sorted(langs)


def group_langs(stats, speakers):
    out = defaultdict(int)
    for spk in speakers:
        for lang, n in stats.get(spk, {}).items():
            out[lang] += n
    return dict(out)


def overall_langs(stats):
    out = defaultdict(int)
    for langs in stats.values():
        for lang, n in langs.items():
            out[lang] += n
    return dict(out)


def fmt_langs(langs_dict):
    """Show each language as 'lang=count (pct% of total)', where total = ALL words
    across every language (not just en+es). So en% = en/total, es% = es/total, etc."""
    if not langs_dict:
        return "(none)"
    total = sum(langs_dict.values()) or 1
    return ", ".join(f"{k}={v} ({v/total*100:.1f}%)"
                     for k, v in sorted(langs_dict.items(), key=lambda x: -x[1]))


def write_report(output_dir, base_name, stats):
    os.makedirs(output_dir, exist_ok=True)
    ov = overall_langs(stats)
    total = sum(ov.values())
    child = group_langs(stats, CHILD_SPEAKERS)
    adult = group_langs(stats, ADULT_SPEAKERS)
    with open(os.path.join(output_dir, f'{base_name}_language_report.txt'), 'w', encoding='utf-8') as f:
        f.write("LANGUAGE ANALYSIS REPORT (no language barrier)\n")
        f.write("=" * 60 + "\n")
        f.write(f"File: {base_name}\n\n")
        f.write("OVERALL\n" + "-" * 60 + "\n")
        f.write(f"  Total words: {total}\n")
        f.write(f"  By language: {fmt_langs(ov)}\n\n")
        f.write("BY SPEAKER\n" + "-" * 60 + "\n")
        for spk in sorted(stats.keys()):
            langs = stats[spk]
            f.write(f"  {spk}: total={sum(langs.values())} | {fmt_langs(langs)}\n")
        f.write("\nGROUPS\n" + "-" * 60 + "\n")
        f.write(f"  ALL CHILDREN (KCHI+OCH): total={sum(child.values())} | {fmt_langs(child)}\n")
        f.write(f"  ALL ADULTS  (MAL+FEM):   total={sum(adult.values())} | {fmt_langs(adult)}\n")


def stats_to_rows(clip_id, stats):
    """Long-format rows: clip_id, speaker, language, words, pct_of_speaker_total.
    'words' for language=TOTAL is the speaker's grand total across ALL languages;
    pct_of_speaker_total = words / that grand total * 100 (so en% and es% are out
    of the total, and all languages for a speaker sum to 100%)."""
    rows = []
    for spk in sorted(stats.keys()):
        langs = stats[spk]
        spk_total = sum(langs.values())
        for lang in sorted(langs.keys()):
            w = langs[lang]
            pct = round(w / spk_total * 100, 2) if spk_total else 0
            rows.append({'clip_id': clip_id, 'speaker': spk, 'language': lang,
                         'words': w, 'pct_of_speaker_total': pct})
        rows.append({'clip_id': clip_id, 'speaker': spk, 'language': 'TOTAL',
                     'words': spk_total, 'pct_of_speaker_total': 100.0 if spk_total else 0})
    return rows


CSV_FIELDS = ['clip_id', 'speaker', 'language', 'words', 'pct_of_speaker_total']


def is_combined_file(filename):
    base = filename[:-4] if filename.endswith('.txt') else filename
    if not base.endswith('_tagged_langdetect'):
        return False
    rem = base[:-len('_tagged_langdetect')]
    skip = ('_KCHI', '_FEM', '_MAL', '_OCH', '_CHI', '_FA1', '_FA2', '_MA1', '_MA2',
            '_EE1', '_EE2', '_UNKNOWN', '_unfiltered', '_SPEECH')
    return not any(rem.endswith(s) for s in skip)


def detect_structure(input_dir):
    for entry in os.scandir(input_dir):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_dir():
                    return '3level'
                if sub.name.endswith('.txt'):
                    return '2level'
    return '2level'


def process_folder(input_folder, output_folder):
    input_path = os.path.abspath(input_folder)
    if not os.path.exists(input_path):
        print(f"Error: input folder '{input_folder}' not found.")
        return
    structure = detect_structure(input_path)
    print(f"Detected folder structure: {structure}")
    recordings = sorted(d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d)))
    if not recordings:
        print(f"No recording folders in '{input_folder}'")
        return

    global_rows = []
    for rec in recordings:
        rec_path = os.path.join(input_path, rec)
        print(f"Recording: {rec}")
        clips = []
        if structure == '2level':
            for fn in sorted(os.listdir(rec_path)):
                fp = os.path.join(rec_path, fn)
                if os.path.isfile(fp) and is_combined_file(fn):
                    clips.append((extract_clip_id(fn), fp))
        else:
            for clip_name in sorted(os.listdir(rec_path)):
                cp = os.path.join(rec_path, clip_name)
                if not os.path.isdir(cp):
                    continue
                for fn in sorted(os.listdir(cp)):
                    fp = os.path.join(cp, fn)
                    if os.path.isfile(fp) and is_combined_file(fn):
                        clips.append((clip_name, fp))
        if not clips:
            print("  (no combined _tagged_langdetect.txt files, skipping)\n")
            continue

        rec_stats = defaultdict(lambda: defaultdict(int))
        rec_rows = []
        for clip_id, fp in clips:
            stats = process_transcript(fp)
            write_report(os.path.join(output_folder, rec, clip_id), clip_id, stats)
            merge_stats(rec_stats, stats)
            rec_rows.extend(stats_to_rows(clip_id, stats))
            print(f"  {clip_id}: total={sum(overall_langs(stats).values())} | {fmt_langs(overall_langs(stats))}")

        rec_stats = {spk: dict(l) for spk, l in rec_stats.items()}
        write_report(os.path.join(output_folder, rec), rec, rec_stats)
        rec_rows.extend(stats_to_rows(f"{rec}_COMBINED", rec_stats))

        os.makedirs(os.path.join(output_folder, rec), exist_ok=True)
        with open(os.path.join(output_folder, rec, 'word_counts.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader(); w.writerows(rec_rows)
        global_rows.extend(r | {'recording': rec} for r in rec_rows)
        print(f"  -> {rec} combined total={sum(overall_langs(rec_stats).values())}\n")

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'all_recordings_word_counts.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['recording'] + CSV_FIELDS); w.writeheader(); w.writerows(global_rows)
    print(f"Done. Reports + CSVs in: {output_folder}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Count words per speaker and per language (no language barrier).')
    p.add_argument('-i', '--input', required=True, help='Input: langdetect-tagged transcripts dir')
    p.add_argument('-o', '--output', required=True, help='Output dir for reports + CSVs')
    args = p.parse_args()
    process_folder(args.input, args.output)
