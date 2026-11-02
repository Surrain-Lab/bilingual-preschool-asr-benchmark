#!/usr/bin/env python3
import re
import os
import csv
import argparse
import unicodedata
from collections import defaultdict


def remove_diacritics(text):
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def extract_language_words(text):
    """
    Extract English and Spanish words from a transcript line's text portion.
    English words are in {curly braces}, Spanish words are everything else
    (after removing excluded content).
    Returns: (english_words_list, spanish_words_list)
    """
    english_words = []
    spanish_words = []

    # 1. Extract English words from {curly braces}
    for match in re.findall(r'\{([^}]+)\}', text):
        cleaned = match.strip()
        if cleaned:
            english_words.extend(cleaned.split())

    # Remove {curly brace} content from text
    text = re.sub(r'\{[^}]+\}', '', text)

    # 2. Handle [: formal equivalent] — extract the formal Spanish form
    for form in re.findall(r'<[^>]*>\s*\[:\s*([^\]]+)\]', text):
        words = [w for w in form.strip().split() if not w.startswith('&')]
        spanish_words.extend(words)
    text = re.sub(r'<[^>]*>\s*\[:\s*[^\]]+\]', '', text)

    # 3. Extract words from <angle brackets> (Spanish, include them)
    for content in re.findall(r'<([^>]+)>', text):
        words = [w for w in content.strip().split() if not w.startswith('&')]
        spanish_words.extend(words)
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Remove ALL content in [square brackets]
    text = re.sub(r'\[.*?\]', '', text)

    # 5. Remove @tags
    text = re.sub(r'@\w+', '', text)

    # 6. Remove words starting with &
    text = re.sub(r'&[^\s]*', '', text)

    # 7. Remove xxx
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)

    # 8. Remove punctuation markers like +//, +/, +<, +...
    text = re.sub(r'\+[/\.<]+', '', text)

    # 9. Remove punctuation
    text = re.sub(r'[.?!,;:]', '', text)

    # 10. Remove + and -
    text = text.replace('+', '').replace('-', '')

    # 11. Remaining words are Spanish
    words = [w for w in text.split() if not w.startswith('&') and w.strip()]
    spanish_words.extend(words)

    return english_words, spanish_words


def map_speaker(speaker_code):
    """
    Map speaker codes to standard format.
    Returns one of: MAL, FEM, KCHI, OCH — or None to skip.
    """
    if speaker_code.startswith('EE'):
        return None
    if speaker_code in ('FEM',) or speaker_code.startswith('FA'):
        return 'FEM'
    if speaker_code in ('MAL',) or speaker_code.startswith('MA'):
        return 'MAL'
    if speaker_code in ('KCHI',) or speaker_code.startswith('CHI') or speaker_code.startswith('Tar'):
        return 'KCHI'
    if speaker_code == 'OCH':
        return 'OCH'
    return None


def parse_transcript_line(line):
    """
    Parse a line in the format:
      [0:00:00.000000  - 0:00:01.190000] (es) [FEM]: puedes ver a xxx.

    Returns (speaker_code, text) or (None, None) if line doesn't match.
    """
    m = re.match(
        r'^\[.*?\]\s*\([^)]+\)\s*\[([A-Za-z0-9_]+)\]:\s*(.*)',
        line.strip()
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


def empty_data():
    return {
        'all_english': [],
        'all_spanish': [],
        'english_by_speaker': {'MAL': [], 'FEM': [], 'KCHI': [], 'OCH': []},
        'spanish_by_speaker': {'MAL': [], 'FEM': [], 'KCHI': [], 'OCH': []},
    }


def collect_language_words(input_file):
    """Parse a single transcript file and return word data dict."""
    data = empty_data()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        speaker_code, text = parse_transcript_line(line)
        if speaker_code is None:
            continue

        speaker = map_speaker(speaker_code)
        if not speaker:
            continue

        english_words, spanish_words = extract_language_words(text)

        data['all_english'].extend(english_words)
        data['all_spanish'].extend(spanish_words)
        data['english_by_speaker'][speaker].extend(english_words)
        data['spanish_by_speaker'][speaker].extend(spanish_words)

    return data


def merge_word_data(data_list):
    """Sum up multiple word-data dicts into one combined dict."""
    merged = empty_data()
    for d in data_list:
        merged['all_english'].extend(d['all_english'])
        merged['all_spanish'].extend(d['all_spanish'])
        for spk in ['MAL', 'FEM', 'KCHI', 'OCH']:
            merged['english_by_speaker'][spk].extend(d['english_by_speaker'][spk])
            merged['spanish_by_speaker'][spk].extend(d['spanish_by_speaker'][spk])
    return merged


def write_language_report(output_dir, base_name,
                           all_english_words, all_spanish_words,
                           english_by_speaker, spanish_by_speaker):
    """Write the 3 report files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute CHILDREN (KCHI + OCH) and ADULTS (MAL + FEM)
    child_english = len(english_by_speaker['KCHI']) + len(english_by_speaker['OCH'])
    child_spanish = len(spanish_by_speaker['KCHI']) + len(spanish_by_speaker['OCH'])
    child_total   = child_english + child_spanish

    adult_english = len(english_by_speaker['MAL']) + len(english_by_speaker['FEM'])
    adult_spanish = len(spanish_by_speaker['MAL']) + len(spanish_by_speaker['FEM'])
    adult_total   = adult_english + adult_spanish

    total_words = len(all_english_words) + len(all_spanish_words)

    # ── language_report.txt ──────────────────────────────────────────────────
    with open(os.path.join(output_dir, f'{base_name}_language_report.txt'), 'w', encoding='utf-8') as f:
        f.write("LANGUAGE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"File: {base_name}\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total English words: {len(all_english_words)}\n")
        f.write(f"Total Spanish words: {len(all_spanish_words)}\n")
        f.write(f"Total words: {total_words}\n")
        if total_words > 0:
            f.write(f"English: {len(all_english_words)/total_words*100:.2f}%\n")
            f.write(f"Spanish: {len(all_spanish_words)/total_words*100:.2f}%\n")
        f.write("\n")

        f.write("STATISTICS BY SPEAKER\n")
        f.write("-" * 60 + "\n")
        for speaker in ['MAL', 'FEM', 'KCHI', 'OCH']:
            eng   = len(english_by_speaker[speaker])
            spa   = len(spanish_by_speaker[speaker])
            total = eng + spa
            f.write(f"\n{speaker}:\n")
            f.write(f"  English words: {eng}\n")
            f.write(f"  Spanish words: {spa}\n")
            f.write(f"  Total words: {total}\n")
            if total > 0:
                f.write(f"  English: {eng/total*100:.2f}%\n")
                f.write(f"  Spanish: {spa/total*100:.2f}%\n")

        f.write("\n")
        f.write("CHILDREN VS. ADULTS COMBINED\n")
        f.write("-" * 60 + "\n")

        f.write("\nALL CHILDREN COMBINED (KCHI + OCH):\n")
        f.write(f"  English words: {child_english}\n")
        f.write(f"  Spanish words: {child_spanish}\n")
        f.write(f"  Total words: {child_total}\n")
        if child_total > 0:
            f.write(f"  English: {child_english/child_total*100:.2f}%\n")
            f.write(f"  Spanish: {child_spanish/child_total*100:.2f}%\n")

        f.write("\nALL ADULTS COMBINED (MAL + FEM):\n")
        f.write(f"  English words: {adult_english}\n")
        f.write(f"  Spanish words: {adult_spanish}\n")
        f.write(f"  Total words: {adult_total}\n")
        if adult_total > 0:
            f.write(f"  English: {adult_english/adult_total*100:.2f}%\n")
            f.write(f"  Spanish: {adult_spanish/adult_total*100:.2f}%\n")

    # ── english_words.txt ────────────────────────────────────────────────────
    with open(os.path.join(output_dir, f'{base_name}_english_words.txt'), 'w', encoding='utf-8') as f:
        f.write("ENGLISH WORDS LIST\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(all_english_words)} words\n\n")
        if all_english_words:
            f.write("All English words (in order of appearance):\n")
            f.write(" ".join(all_english_words) + "\n\n")
            for speaker in ['MAL', 'FEM', 'KCHI', 'OCH']:
                words = english_by_speaker[speaker]
                if words:
                    f.write(f"\n{speaker} English words ({len(words)} words):\n")
                    f.write(" ".join(words) + "\n")

    # ── spanish_words.txt ────────────────────────────────────────────────────
    with open(os.path.join(output_dir, f'{base_name}_spanish_words.txt'), 'w', encoding='utf-8') as f:
        f.write("SPANISH WORDS LIST\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(all_spanish_words)} words\n\n")
        if all_spanish_words:
            f.write("All Spanish words (in order of appearance):\n")
            f.write(" ".join(all_spanish_words) + "\n\n")
            for speaker in ['MAL', 'FEM', 'KCHI', 'OCH']:
                words = spanish_by_speaker[speaker]
                if words:
                    f.write(f"\n{speaker} Spanish words ({len(words)} words):\n")
                    f.write(" ".join(words) + "\n")


# ============================================================
# CSV helpers (mirrors Script 1 style)
# ============================================================

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


def pct(a, b):
    t = a + b
    return (round(a / t * 100, 2) if t > 0 else 0,
            round(b / t * 100, 2) if t > 0 else 0)


def prepare_csv_row(name, data):
    """Build a CSV row dict from a word-data dict."""
    eng  = {spk: len(data['english_by_speaker'][spk]) for spk in ['MAL', 'FEM', 'KCHI', 'OCH']}
    spa  = {spk: len(data['spanish_by_speaker'][spk]) for spk in ['MAL', 'FEM', 'KCHI', 'OCH']}

    total_eng = len(data['all_english'])
    total_spa = len(data['all_spanish'])
    total_words = total_eng + total_spa

    child_eng = eng['KCHI'] + eng['OCH']
    child_spa = spa['KCHI'] + spa['OCH']
    adult_eng = eng['MAL']  + eng['FEM']
    adult_spa = spa['MAL']  + spa['FEM']

    e_pct,  s_pct  = pct(total_eng,  total_spa)
    ke_pct, ks_pct = pct(eng['KCHI'], spa['KCHI'])
    oe_pct, os_pct = pct(eng['OCH'],  spa['OCH'])
    ce_pct, cs_pct = pct(child_eng,   child_spa)
    me_pct, ms_pct = pct(eng['MAL'],  spa['MAL'])
    fe_pct, fs_pct = pct(eng['FEM'],  spa['FEM'])
    ae_pct, as_pct = pct(adult_eng,   adult_spa)

    return {
        'Audio_File': name,
        'Total_Words': total_words,
        'Total_English_Words': total_eng,
        'Total_Spanish_Words': total_spa,
        'English_Percentage': e_pct,
        'Spanish_Percentage': s_pct,
        'KCHI_Total_Words': eng['KCHI'] + spa['KCHI'],
        'KCHI_English_Words': eng['KCHI'],
        'KCHI_Spanish_Words': spa['KCHI'],
        'KCHI_English_Percentage': ke_pct,
        'KCHI_Spanish_Percentage': ks_pct,
        'OCH_Total_Words': eng['OCH'] + spa['OCH'],
        'OCH_English_Words': eng['OCH'],
        'OCH_Spanish_Words': spa['OCH'],
        'OCH_English_Percentage': oe_pct,
        'OCH_Spanish_Percentage': os_pct,
        'Children_Combined_Total_Words': child_eng + child_spa,
        'Children_Combined_English_Words': child_eng,
        'Children_Combined_Spanish_Words': child_spa,
        'Children_Combined_English_Percentage': ce_pct,
        'Children_Combined_Spanish_Percentage': cs_pct,
        'MAL_Total_Words': eng['MAL'] + spa['MAL'],
        'MAL_English_Words': eng['MAL'],
        'MAL_Spanish_Words': spa['MAL'],
        'MAL_English_Percentage': me_pct,
        'MAL_Spanish_Percentage': ms_pct,
        'FEM_Total_Words': eng['FEM'] + spa['FEM'],
        'FEM_English_Words': eng['FEM'],
        'FEM_Spanish_Words': spa['FEM'],
        'FEM_English_Percentage': fe_pct,
        'FEM_Spanish_Percentage': fs_pct,
        'Adults_Combined_Total_Words': adult_eng + adult_spa,
        'Adults_Combined_English_Words': adult_eng,
        'Adults_Combined_Spanish_Words': adult_spa,
        'Adults_Combined_English_Percentage': ae_pct,
        'Adults_Combined_Spanish_Percentage': as_pct,
    }


def write_csv(path, rows):
    """Write a CSV file with the standard fieldnames."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"   CSV saved: {path}")


# ============================================================


def extract_clip_id(filename):
    """
    Extract the clip base ID from a filename like:
      R04_pre_010_human.txt  →  R04_pre_010
    """
    m = re.match(r'^([A-Z]+-\d+_[a-z]+_\d+)', filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]


def print_summary(name, data):
    eng   = len(data['all_english'])
    spa   = len(data['all_spanish'])
    total = eng + spa
    line  = f"  English: {eng}, Spanish: {spa}, Total: {total}"
    if total > 0:
        line += f"  ({eng/total*100:.1f}% EN / {spa/total*100:.1f}% ES)"
    print(line)


def process_folder(input_folder, output_base="human_count"):
    """
    Walk structure:
      input_folder/
        R04_pre/
          R04_pre_010_human.txt
          R04_pre_011_human.txt

    Outputs per recording folder:
      output_base/
        R04_pre/
          combined_statistics.csv       ← one row per clip + combined row
          R04_pre_language_report.txt
          R04_pre_english_words.txt
          R04_pre_spanish_words.txt
          R04_pre_010/
            R04_pre_010_language_report.txt
            R04_pre_010_english_words.txt
            R04_pre_010_spanish_words.txt
    """
    input_path = os.path.abspath(input_folder)
    if not os.path.exists(input_path):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    recording_folders = sorted([
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ])

    if not recording_folders:
        print(f"No recording folders found in '{input_folder}'")
        return

    print(f"Found {len(recording_folders)} recording folder(s)")
    print("=" * 60)

    for recording_name in recording_folders:
        recording_path = os.path.join(input_path, recording_name)

        txt_files = sorted([
            f for f in os.listdir(recording_path)
            if f.endswith('.txt') and os.path.isfile(os.path.join(recording_path, f))
        ])

        if not txt_files:
            print(f"WARNING: No .txt files found in '{recording_name}', skipping.")
            continue

        print(f"\nRecording: {recording_name}  ({len(txt_files)} clips)")

        recording_data_list = []
        recording_csv_rows  = []

        for txt_filename in txt_files:
            # BUGFIX clip-id: files are canonically named — use the stem as-is
            # (the regex truncated R..._1_NNN stems to R..._1)
            clip_id         = os.path.splitext(txt_filename)[0]
            transcript_file = os.path.join(recording_path, txt_filename)

            data = collect_language_words(transcript_file)
            recording_data_list.append(data)

            # Per-clip output
            clip_output_dir = os.path.join(output_base, recording_name, clip_id)
            write_language_report(
                clip_output_dir, clip_id,
                data['all_english'], data['all_spanish'],
                data['english_by_speaker'], data['spanish_by_speaker']
            )

            # Add clip row to CSV
            recording_csv_rows.append(prepare_csv_row(clip_id, data))

            print(f"   {clip_id}  →", end="")
            print_summary(clip_id, data)

        if not recording_data_list:
            print(f"  WARNING: No valid clips processed for '{recording_name}'")
            continue

        # Combined recording-level report
        combined = merge_word_data(recording_data_list)
        recording_output_dir = os.path.join(output_base, recording_name)
        write_language_report(
            recording_output_dir, recording_name,
            combined['all_english'], combined['all_spanish'],
            combined['english_by_speaker'], combined['spanish_by_speaker']
        )

        # Add combined row as last row in CSV
        recording_csv_rows.append(prepare_csv_row(f"{recording_name}_COMBINED", combined))

        # Write CSV for this recording
        csv_path = os.path.join(output_base, recording_name, 'combined_statistics.csv')
        write_csv(csv_path, recording_csv_rows)

        print(f"   COMBINED ({recording_name})  →", end="")
        print_summary(recording_name, combined)

    print(f"\n{'=' * 60}")
    print(f" All done! Reports saved in: {output_base}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Count English/Spanish words from human transcripts'
    )
    parser.add_argument('-i', '--input',  required=True,
                        help='Input folder containing human transcript txt files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output folder for language reports and CSV')
    args = parser.parse_args()
    process_folder(args.input, args.output)
