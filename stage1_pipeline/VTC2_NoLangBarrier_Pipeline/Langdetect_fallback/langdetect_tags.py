# BUGFIX overlap: speaker-tag regexes now accept an optional [OVERLAP] flag
# BUGFIX clip-id: use clip FOLDER name (filename regex truncated _1_NNN stems)
import os
import re
import sys
from collections import defaultdict, Counter
from langdetect import detect, LangDetectException


def extract_clip_id(filename):
    """
    Extract the clip base ID from a filename like:
      P02_pre_010-VOR-AM_human.txt  →  P02_pre_010
      P02_pre_002_tagged.txt        →  P02_pre_002
    """
    m = re.match(r'^([A-Z]+-\d+_[a-z]+_\d+)', filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]


def detect_language(text, lang_hint=None):
    """
    NO-LANGUAGE-BARRIER version: keep the REAL language (any ISO-639-1 code),
    not just en/es.

    Priority:
      1. Trust the upstream WhisperX language hint (the (xx) tag from the
         transcription step) — it is acoustic + text based and reliable.
      2. Otherwise run langdetect over the text (any language).
      3. Fall back to 'unknown'.
    """
    if lang_hint:
        return lang_hint
    try:
        text = text.strip()
        if not text:
            return 'unknown'
        # Spanish-specific characters are an unambiguous signal
        if any(char in text.lower() for char in 'áéíóúñü¿¡'):
            return 'es'
        # consensus over a few attempts (langdetect is non-deterministic-ish)
        results = []
        for _ in range(7):
            try:
                results.append(detect(text))
            except Exception:
                pass
        if results:
            return Counter(results).most_common(1)[0][0]
        return 'unknown'
    except Exception:
        return 'unknown'


def process_line(line):
    """
    Process a single transcript line — detect language and append [lang] tag.
    Input:  [0:00:00.000000  - 0:00:01.190000] (es) [FEM]: puedes ver a xxx.
    Output: [0:00:00.000000  - 0:00:01.190000] (es) [FEM]: puedes ver a xxx.[es]
    Returns (processed_line, speaker) — speaker is None if line doesn't match.
    """
    line = line.strip()
    if not line:
        return line, None

    pattern = r'(\[[\d:.\s-]+\])\s*(\([a-z]+\))?\s*(\[[A-Z0-9_]+\](?:\s*\[OVERLAP\])?:)\s*(.+)'
    match = re.match(pattern, line)

    if match:
        timestamp     = match.group(1)
        orig_lang_tag = match.group(2) if match.group(2) else ''
        speaker_tag   = match.group(3)
        sentence      = match.group(4).strip()

        # Extract speaker name from [FEM]: → FEM
        speaker = re.search(r'\[([A-Z0-9_]+)\]', speaker_tag).group(1)

        # Extract language hint
        lang_hint = None
        if orig_lang_tag:
            lm = re.search(r'\(([a-z]+)\)', orig_lang_tag)
            if lm:                       # accept ANY language code, not just en/es
                lang_hint = lm.group(1)

        detected_lang = detect_language(sentence, lang_hint)

        if orig_lang_tag:
            processed = f"{timestamp} {orig_lang_tag} {speaker_tag} {sentence}[{detected_lang}]"
        else:
            processed = f"{timestamp} {speaker_tag} {sentence}[{detected_lang}]"

        return processed, speaker
    else:
        return line, None


def write_txt(path, lines):
    """Write a list of lines to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def process_file(input_file, output_dir, clip_id):
    """
    Process a single transcript file.
    Writes:
      - {clip_id}_tagged_langdetect.txt               (all speakers combined)
      - {clip_id}_{SPEAKER}_tagged_langdetect.txt      (one per speaker found)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        all_processed = []
        by_speaker    = defaultdict(list)

        for line in lines:
            processed_line, speaker = process_line(line)
            all_processed.append(processed_line)
            if speaker:
                by_speaker[speaker].append(processed_line)

        # Combined file — all speakers
        combined_path = os.path.join(output_dir, f"{clip_id}_tagged_langdetect.txt")
        write_txt(combined_path, all_processed)

        # Per-speaker files
        for speaker, speaker_lines in sorted(by_speaker.items()):
            speaker_path = os.path.join(output_dir, f"{clip_id}_{speaker}_tagged_langdetect.txt")
            write_txt(speaker_path, speaker_lines)

        return True, sorted(by_speaker.keys())

    except Exception as e:
        print(f"  ERROR: Error processing {input_file}: {e}")
        return False, []


def detect_structure(input_dir):
    """
    Auto-detect whether input is 2-level or 3-level:
      2-level: input/ → recording/ → file.txt
      3-level: input/ → recording/ → clip/ → file.txt
    """
    for entry in os.scandir(input_dir):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_dir():
                    return '3level'
                if sub.name.endswith('.txt'):
                    return '2level'
    return '2level'


def process_folder(input_dir, output_dir):
    """
    Supports two structures automatically:

    2-LEVEL (human transcripts):
      input/
        P02_pre/
          P02_pre_010-VOR-AM_human.txt

    3-LEVEL (automated transcripts):
      input/
        P02_pre/
          P02_pre_002/
            P02_pre_002_tagged.txt

    Output (same for both):
      output/
        P02_pre/
          P02_pre_010/
            P02_pre_010_tagged_langdetect.txt
            P02_pre_010_FEM_tagged_langdetect.txt
            P02_pre_010_MAL_tagged_langdetect.txt
            P02_pre_010_KCHI_tagged_langdetect.txt
            P02_pre_010_OCH_tagged_langdetect.txt
            ... (only speakers present in that clip)
    """
    input_path = os.path.abspath(input_dir)
    if not os.path.exists(input_path):
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    structure = detect_structure(input_path)
    print(f"Detected folder structure: {structure}")
    print("=" * 60)

    recording_folders = sorted([
        d for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ])

    if not recording_folders:
        print(f"No recording folders found in '{input_dir}'")
        return

    print(f"Found {len(recording_folders)} recording folder(s)\n")

    total_processed = 0

    for recording_name in recording_folders:
        recording_path = os.path.join(input_path, recording_name)
        print(f"Recording: {recording_name}")

        # Build list of (clip_id, input_file) pairs
        clips = []

        if structure == '2level':
            txt_files = sorted([
                f for f in os.listdir(recording_path)
                if f.endswith('.txt') and os.path.isfile(os.path.join(recording_path, f))
            ])
            if not txt_files:
                print(f"  WARNING: No .txt files found, skipping.\n")
                continue
            for txt_filename in txt_files:
                clips.append((extract_clip_id(txt_filename),
                               os.path.join(recording_path, txt_filename)))

        else:  # 3level
            clip_folders = sorted([
                d for d in os.listdir(recording_path)
                if os.path.isdir(os.path.join(recording_path, d))
            ])
            if not clip_folders:
                print(f"  WARNING: No clip folders found, skipping.\n")
                continue
            for clip_name in clip_folders:
                clip_path = os.path.join(recording_path, clip_name)
                txt_files = sorted([
                    f for f in os.listdir(clip_path)
                    if f.endswith('.txt') and os.path.isfile(os.path.join(clip_path, f))
                ])
                for txt_filename in txt_files:
                    clips.append((clip_name,
                                   os.path.join(clip_path, txt_filename)))

        # Process each clip
        for clip_id, input_file in clips:
            out_dir = os.path.join(output_dir, recording_name, clip_id)
            success, speakers = process_file(input_file, out_dir, clip_id)
            if success:
                total_processed += 1
                print(f"   {clip_id}  →  speakers: {', '.join(speakers) if speakers else 'none'}")

        print()

    print("=" * 60)
    print(f"Done! Total clips processed: {total_processed}")
    print(f"  Output saved in: {output_dir}/")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect language in transcript files and write per-speaker [lang]-tagged outputs.'
    )
    parser.add_argument('-i', '--input',  required=True,
                        help='Input directory (contains recording folders)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')

    args = parser.parse_args()
    process_folder(args.input, args.output)
