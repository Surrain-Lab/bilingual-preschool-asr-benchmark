# BUGFIX overlap: speaker-tag regexes now accept an optional [OVERLAP] flag
import re
import os
import csv
import argparse
from collections import defaultdict
from langdetect import detect, DetectorFactory

# Set seed for consistent results
DetectorFactory.seed = 0


def extract_clip_id(filename):
    """
    P02_pre_010_tagged_langdetect.txt  →  P02_pre_010
    """
    m = re.match(r'^([A-Z]+-\d+_[a-z]+_\d+)', filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]


def parse_line(line):
    """
    Parse: [0:00:00.0 - 0:00:01.0] (es) [FEM]: text here[es]
    Returns (speaker, text, lang_tag) or None.
    """
    line = line.strip()
    if not line:
        return None

    pattern = r'\[[\d:.\s-]+\]\s*\(\w+\)\s*\[([^\]]+)\](?:\s*\[OVERLAP\])?:\s*(.+)$'
    match = re.match(pattern, line)
    if not match:
        return None

    speaker = match.group(1)
    text    = match.group(2).strip()
    if not text:
        return None

    lang_match = re.match(r'^(.+?)\s*\[(\w+)\]$', text)
    if not lang_match:
        return None

    text     = lang_match.group(1).strip()
    lang_tag = lang_match.group(2)

    return speaker, text, lang_tag


def count_words(text):
    return len(text.split())


def empty_speaker_stats():
    return {'english': 0, 'spanish': 0}


def process_transcript(input_file):
    """Parse a single _tagged_langdetect.txt and return word data."""
    speaker_stats     = defaultdict(empty_speaker_stats)
    all_english_lines = []
    all_spanish_lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue

        speaker, text, lang_tag = parsed

        if speaker.startswith('EE'):
            continue

        language   = 'spanish' if lang_tag.lower() == 'es' else 'english'
        word_count = count_words(text)

        speaker_stats[speaker][language] += word_count

        if language == 'spanish':
            all_spanish_lines.append(f"[{speaker}]: {text}")
        else:
            all_english_lines.append(f"[{speaker}]: {text}")

    return dict(speaker_stats), all_english_lines, all_spanish_lines


def compute_group_totals(speaker_stats):
    """Compute per-speaker and group totals."""
    def get(spk, lang):
        return speaker_stats.get(spk, {}).get(lang, 0)

    kchi_eng  = get('KCHI', 'english');  kchi_spa  = get('KCHI', 'spanish')
    och_eng   = get('OCH',  'english');  och_spa   = get('OCH',  'spanish')
    mal_eng   = get('MAL',  'english');  mal_spa   = get('MAL',  'spanish')
    fem_eng   = get('FEM',  'english');  fem_spa   = get('FEM',  'spanish')

    child_eng = kchi_eng + och_eng;  child_spa = kchi_spa + och_spa
    adult_eng = mal_eng  + fem_eng;  adult_spa = mal_spa  + fem_spa

    total_eng = sum(v['english'] for v in speaker_stats.values())
    total_spa = sum(v['spanish'] for v in speaker_stats.values())

    return {
        'total_english': total_eng, 'total_spanish': total_spa,
        'kchi_english':  kchi_eng,  'kchi_spanish':  kchi_spa,
        'och_english':   och_eng,   'och_spanish':   och_spa,
        'child_english': child_eng, 'child_spanish': child_spa,
        'mal_english':   mal_eng,   'mal_spanish':   mal_spa,
        'fem_english':   fem_eng,   'fem_spanish':   fem_spa,
        'adult_english': adult_eng, 'adult_spanish': adult_spa,
    }


def write_language_report(output_dir, base_name, speaker_stats, g):
    os.makedirs(output_dir, exist_ok=True)
    total_words = g['total_english'] + g['total_spanish']

    with open(os.path.join(output_dir, f'{base_name}_language_report.txt'), 'w', encoding='utf-8') as f:
        f.write("LANGUAGE ANALYSIS REPORT - AUDIO TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"File: {base_name}\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total English words: {g['total_english']}\n")
        f.write(f"Total Spanish words: {g['total_spanish']}\n")
        f.write(f"Total words: {total_words}\n")
        if total_words > 0:
            f.write(f"English: {g['total_english']/total_words*100:.2f}%\n")
            f.write(f"Spanish: {g['total_spanish']/total_words*100:.2f}%\n")
        f.write("\n")

        f.write("STATISTICS BY SPEAKER\n")
        f.write("-" * 60 + "\n")
        for speaker in sorted(speaker_stats.keys()):
            eng   = speaker_stats[speaker]['english']
            spa   = speaker_stats[speaker]['spanish']
            total = eng + spa
            label = "KEY CHILD" if speaker == 'KCHI' else speaker
            f.write(f"\n{label}:\n")
            f.write(f"  English words: {eng}\n")
            f.write(f"  Spanish words: {spa}\n")
            f.write(f"  Total words: {total}\n")
            if total > 0:
                f.write(f"  English: {eng/total*100:.2f}%\n")
                f.write(f"  Spanish: {spa/total*100:.2f}%\n")

        f.write("\n")
        f.write("CHILDREN VS. ADULTS COMBINED\n")
        f.write("-" * 60 + "\n")
        for label, ek, sk in [
            ("KEY CHILD (KCHI)",               'kchi_english',  'kchi_spanish'),
            ("OTHER CHILDREN (OCH)",           'och_english',   'och_spanish'),
            ("ALL CHILDREN COMBINED (KCHI+OCH)", 'child_english', 'child_spanish'),
            ("ALL ADULTS COMBINED (MAL+FEM)",  'adult_english', 'adult_spanish'),
        ]:
            eng   = g[ek]; spa = g[sk]; total = eng + spa
            f.write(f"\n{label}:\n")
            f.write(f"  English words: {eng}\n")
            f.write(f"  Spanish words: {spa}\n")
            f.write(f"  Total words: {total}\n")
            if total > 0:
                f.write(f"  English: {eng/total*100:.2f}%\n")
                f.write(f"  Spanish: {spa/total*100:.2f}%\n")


def write_word_lists(output_dir, base_name, english_lines, spanish_lines):
    os.makedirs(output_dir, exist_ok=True)

    def wc(lines):
        return sum(len(l.split(':', 1)[1].split()) for l in lines if ':' in l)

    with open(os.path.join(output_dir, f'{base_name}_english_words.txt'), 'w', encoding='utf-8') as f:
        f.write("ENGLISH WORDS LIST - AUDIO TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {wc(english_lines)} words\n\n")
        if english_lines:
            f.write("All English utterances:\n\n")
            for line in english_lines:
                f.write(line + "\n")

    with open(os.path.join(output_dir, f'{base_name}_spanish_words.txt'), 'w', encoding='utf-8') as f:
        f.write("SPANISH WORDS LIST - AUDIO TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {wc(spanish_lines)} words\n\n")
        if spanish_lines:
            f.write("All Spanish utterances:\n\n")
            for line in spanish_lines:
                f.write(line + "\n")


def prepare_csv_row(name, speaker_stats, g):
    """Build a CSV row dict."""
    def pct(a, b):
        t = a + b
        return (round(a/t*100, 2) if t > 0 else 0,
                round(b/t*100, 2) if t > 0 else 0)

    total_words  = g['total_english'] + g['total_spanish']
    e_pct, s_pct = pct(g['total_english'], g['total_spanish'])

    kchi_t = g['kchi_english']  + g['kchi_spanish']
    och_t  = g['och_english']   + g['och_spanish']
    chd_t  = g['child_english'] + g['child_spanish']
    mal_t  = g['mal_english']   + g['mal_spanish']
    fem_t  = g['fem_english']   + g['fem_spanish']
    adt_t  = g['adult_english'] + g['adult_spanish']

    ke_p, ks_p   = pct(g['kchi_english'],  g['kchi_spanish'])
    oe_p, os_p   = pct(g['och_english'],   g['och_spanish'])
    ce_p, cs_p   = pct(g['child_english'], g['child_spanish'])
    me_p, ms_p   = pct(g['mal_english'],   g['mal_spanish'])
    fe_p, fs_p   = pct(g['fem_english'],   g['fem_spanish'])
    ae_p, as_p   = pct(g['adult_english'], g['adult_spanish'])

    return {
        'Audio_File': name,
        'Total_Words': total_words,
        'Total_English_Words': g['total_english'],
        'Total_Spanish_Words': g['total_spanish'],
        'English_Percentage': e_pct, 'Spanish_Percentage': s_pct,
        'KCHI_Total_Words': kchi_t,
        'KCHI_English_Words': g['kchi_english'], 'KCHI_Spanish_Words': g['kchi_spanish'],
        'KCHI_English_Percentage': ke_p, 'KCHI_Spanish_Percentage': ks_p,
        'OCH_Total_Words': och_t,
        'OCH_English_Words': g['och_english'], 'OCH_Spanish_Words': g['och_spanish'],
        'OCH_English_Percentage': oe_p, 'OCH_Spanish_Percentage': os_p,
        'Children_Combined_Total_Words': chd_t,
        'Children_Combined_English_Words': g['child_english'],
        'Children_Combined_Spanish_Words': g['child_spanish'],
        'Children_Combined_English_Percentage': ce_p,
        'Children_Combined_Spanish_Percentage': cs_p,
        'MAL_Total_Words': mal_t,
        'MAL_English_Words': g['mal_english'], 'MAL_Spanish_Words': g['mal_spanish'],
        'MAL_English_Percentage': me_p, 'MAL_Spanish_Percentage': ms_p,
        'FEM_Total_Words': fem_t,
        'FEM_English_Words': g['fem_english'], 'FEM_Spanish_Words': g['fem_spanish'],
        'FEM_English_Percentage': fe_p, 'FEM_Spanish_Percentage': fs_p,
        'Adults_Combined_Total_Words': adt_t,
        'Adults_Combined_English_Words': g['adult_english'],
        'Adults_Combined_Spanish_Words': g['adult_spanish'],
        'Adults_Combined_English_Percentage': ae_p,
        'Adults_Combined_Spanish_Percentage': as_p,
    }


CSV_FIELDNAMES = [
    'Audio_File',
    'Total_Words', 'Total_English_Words', 'Total_Spanish_Words',
    'English_Percentage', 'Spanish_Percentage',
    'KCHI_Total_Words', 'KCHI_English_Words', 'KCHI_Spanish_Words',
    'KCHI_English_Percentage', 'KCHI_Spanish_Percentage',
    'OCH_Total_Words', 'OCH_English_Words', 'OCH_Spanish_Words',
    'OCH_English_Percentage', 'OCH_Spanish_Percentage',
    'Children_Combined_Total_Words', 'Children_Combined_English_Words',
    'Children_Combined_Spanish_Words', 'Children_Combined_English_Percentage',
    'Children_Combined_Spanish_Percentage',
    'MAL_Total_Words', 'MAL_English_Words', 'MAL_Spanish_Words',
    'MAL_English_Percentage', 'MAL_Spanish_Percentage',
    'FEM_Total_Words', 'FEM_English_Words', 'FEM_Spanish_Words',
    'FEM_English_Percentage', 'FEM_Spanish_Percentage',
    'Adults_Combined_Total_Words', 'Adults_Combined_English_Words',
    'Adults_Combined_Spanish_Words', 'Adults_Combined_English_Percentage',
    'Adults_Combined_Spanish_Percentage',
]


def write_csv(path, rows):
    """Write a CSV file with the standard fieldnames."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"   CSV saved: {path}")


def detect_structure(input_dir):
    """Auto-detect 2-level or 3-level folder structure."""
    for entry in os.scandir(input_dir):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_dir():
                    return '3level'
                if sub.name.endswith('.txt'):
                    return '2level'
    return '2level'


def is_combined_file(filename):
    """True only for the all-speaker _tagged_langdetect.txt (not speaker-specific files)."""
    base = filename[:-4] if filename.endswith('.txt') else filename
    if not base.endswith('_tagged_langdetect'):
        return False
    remainder = base[:-len('_tagged_langdetect')]
    skip_suffixes = (
        '_KCHI', '_FEM', '_MAL', '_OCH', '_CHI',
        '_FA1', '_FA2', '_MA1', '_MA2',
        '_EE1', '_EE2', '_UNKNOWN', '_unfiltered', '_SPEECH'
    )
    return not any(remainder.endswith(s) for s in skip_suffixes)


def process_folder(input_folder, output_folder):
    """
    Walk structure (auto-detected):

    3-LEVEL:
      input/ → DL-xxx_pre/ → DL-xxx_pre_010/ → DL-xxx_pre_010_tagged_langdetect.txt

    2-LEVEL:
      input/ → DL-xxx_pre/ → DL-xxx_pre_010_tagged_langdetect.txt

    Output per recording folder:
      output/
        P02_pre/
          combined_statistics.csv       ← clips of THIS recording only
          P02_pre_language_report.txt
          P02_pre_english_words.txt
          P02_pre_spanish_words.txt
          P02_pre_010/
            P02_pre_010_language_report.txt
            P02_pre_010_english_words.txt
            P02_pre_010_spanish_words.txt
        P03_pre/
          combined_statistics.csv       ← clips of THIS recording only
          ...
    """
    input_path = os.path.abspath(input_folder)
    if not os.path.exists(input_path):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    structure = detect_structure(input_path)
    print(f"Detected folder structure: {structure}")
    print("=" * 60)

    recording_folders = sorted([
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ])

    if not recording_folders:
        print(f"No recording folders found in '{input_folder}'")
        return

    print(f"Found {len(recording_folders)} recording folder(s)\n")

    for recording_name in recording_folders:
        recording_path = os.path.join(input_path, recording_name)
        print(f"Recording: {recording_name}")

        # Collect (clip_id, input_file) pairs for this recording
        clips = []

        if structure == '2level':
            for fname in sorted(os.listdir(recording_path)):
                fpath = os.path.join(recording_path, fname)
                if os.path.isfile(fpath) and is_combined_file(fname):
                    clips.append((extract_clip_id(fname), fpath))
        else:
            for clip_name in sorted(os.listdir(recording_path)):
                clip_path = os.path.join(recording_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue
                for fname in sorted(os.listdir(clip_path)):
                    fpath = os.path.join(clip_path, fname)
                    if os.path.isfile(fpath) and is_combined_file(fname):
                        # BUGFIX: use the clip FOLDER name as the clip id.
                        # extract_clip_id(fname) truncates stems like
                        # P04_pre_1_003 to P04_pre_1, which
                        # collapses every clip of a "_1" recording onto one id
                        # so later clips overwrite earlier ones in the outputs.
                        clips.append((clip_name, fpath))

        if not clips:
            print(f"  WARNING: No combined _tagged_langdetect.txt files found, skipping.\n")
            continue

        # Per-recording accumulators
        rec_speaker_stats   = defaultdict(empty_speaker_stats)
        rec_english_lines   = []
        rec_spanish_lines   = []
        recording_csv_rows  = []   # ← rows for THIS recording's CSV only

        for clip_id, input_file in clips:
            speaker_stats, eng_lines, spa_lines = process_transcript(input_file)
            g = compute_group_totals(speaker_stats)

            # Per-clip reports
            clip_out = os.path.join(output_folder, recording_name, clip_id)
            write_language_report(clip_out, clip_id, speaker_stats, g)
            write_word_lists(clip_out, clip_id, eng_lines, spa_lines)

            # Accumulate for recording-level
            for spk, counts in speaker_stats.items():
                rec_speaker_stats[spk]['english'] += counts['english']
                rec_speaker_stats[spk]['spanish'] += counts['spanish']
            rec_english_lines.extend(eng_lines)
            rec_spanish_lines.extend(spa_lines)

            # Add clip row to THIS recording's CSV
            recording_csv_rows.append(prepare_csv_row(clip_id, speaker_stats, g))

            total = g['total_english'] + g['total_spanish']
            print(f"   {clip_id}  →  EN:{g['total_english']} ES:{g['total_spanish']} Total:{total}")

        # Combined recording-level reports
        rec_stats = dict(rec_speaker_stats)
        rec_g     = compute_group_totals(rec_stats)
        rec_out   = os.path.join(output_folder, recording_name)
        write_language_report(rec_out, recording_name, rec_stats, rec_g)
        write_word_lists(rec_out, recording_name, rec_english_lines, rec_spanish_lines)

        # Add combined row as the last row in this recording's CSV
        recording_csv_rows.append(prepare_csv_row(f"{recording_name}_COMBINED", rec_stats, rec_g))

        # Write CSV for THIS recording only
        csv_path = os.path.join(output_folder, recording_name, 'combined_statistics.csv')
        write_csv(csv_path, recording_csv_rows)

        rec_total = rec_g['total_english'] + rec_g['total_spanish']
        print(f"  COMBINED  →  EN:{rec_g['total_english']} ES:{rec_g['total_spanish']} Total:{rec_total}\n")

    print("=" * 60)
    print(f"All done! Reports saved in: {output_folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Count English/Spanish words from langdetect-tagged transcripts'
    )
    parser.add_argument('-i', '--input',  required=True,
                        help='Input folder containing *_tagged_langdetect.txt files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output folder for language reports and CSV')
    args = parser.parse_args()
    process_folder(args.input, args.output)
