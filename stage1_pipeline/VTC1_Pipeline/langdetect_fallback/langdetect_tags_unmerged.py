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
    Detect language of the given text.
    Returns 'en', 'es', or 'unknown'.
    """
    try:
        text = text.strip()
        if not text:
            return 'unknown'

        text_lower = text.lower()

        # Strategy 1: Spanish-specific characters (never in English)
        if any(char in text_lower for char in 'áéíóúñü¿¡'):
            return 'es'

        word_count = len(text.split())

        # Strategy 2: Short text — trust hint if langdetect roughly agrees
        if word_count <= 2 and lang_hint in ['en', 'es']:
            quick_check = []
            for _ in range(5):
                try:
                    quick_check.append(detect(text))
                except Exception:
                    pass
            if lang_hint in quick_check or len(set(quick_check)) > 2:
                return lang_hint

        # Strategy 3: Multiple langdetect attempts with consensus
        results = []
        for _ in range(20):
            try:
                detected = detect(text)
                if detected in ['en', 'es']:
                    results.append(detected)
            except Exception:
                pass

        if results:
            counts = Counter(results)
            most_common_lang, most_common_count = counts.most_common(1)[0]
            confidence = most_common_count / len(results)
            required_confidence = 0.5 if word_count <= 3 else 0.4

            if confidence >= required_confidence:
                return most_common_lang
            other = 'en' if most_common_lang == 'es' else 'es'
            if most_common_count >= 2 * counts.get(other, 0):
                return most_common_lang

        # Strategy 4: Vowel-ratio heuristic
        if len(text) >= 10 and results:
            vowels = sum(1 for c in text_lower if c in 'aeiou')
            total_letters = sum(1 for c in text_lower if c.isalpha())
            if total_letters > 0 and (vowels / total_letters) > 0.47:
                counts = Counter(results)
                if counts.get('es', 0) > 0 and abs(counts.get('es', 0) - counts.get('en', 0)) <= 2:
                    return 'es'

        # Strategy 5: Return most common even with low confidence
        if results:
            return Counter(results).most_common(1)[0][0]

        # Strategy 6: Fall back to hint
        if lang_hint in ['en', 'es']:
            return lang_hint

        return 'unknown'

    except Exception:
        if lang_hint in ['en', 'es']:
            return lang_hint
        return 'unknown'


def is_speech_line(line):
    """
    Return True if the line contains a [SPEECH] speaker tag.
    Matches lines like: [timestamp] (lang) [SPEECH]: text
    """
    return bool(re.search(r'\[SPEECH\](?:\s*\[OVERLAP\])?:', line))


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
            if lm and lm.group(1) in ['en', 'es']:
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
      {clip_id}_tagged_langdetect.txt
          — all speakers, SPEECH lines EXCLUDED

      {clip_id}_with_SPEECH_tagged_langdetect.txt
          — all speakers, SPEECH lines INCLUDED

      {clip_id}_{SPEAKER}_tagged_langdetect.txt
          — one file per non-SPEECH speaker found (SPEECH excluded)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Lines with [SPEECH] tag processed but kept separate
        all_no_speech   = []   # combined, SPEECH excluded
        all_with_speech = []   # combined, SPEECH included
        by_speaker      = defaultdict(list)   # non-SPEECH speakers only

        for line in lines:
            speech_line = is_speech_line(line)
            processed_line, speaker = process_line(line)

            # Always add to the with-SPEECH file
            all_with_speech.append(processed_line)

            if speech_line:
                # SPEECH lines go ONLY into the with_SPEECH file, not the others
                continue

            # Non-SPEECH lines go into the clean combined file and per-speaker files
            all_no_speech.append(processed_line)
            if speaker:
                by_speaker[speaker].append(processed_line)

        # ── Write combined file WITHOUT [SPEECH] ─────────────────────────────
        write_txt(
            os.path.join(output_dir, f"{clip_id}_tagged_langdetect.txt"),
            all_no_speech
        )

        # ── Write combined file WITH [SPEECH] ────────────────────────────────
        write_txt(
            os.path.join(output_dir, f"{clip_id}_with_SPEECH_tagged_langdetect.txt"),
            all_with_speech
        )

        # ── Write per-speaker files (SPEECH excluded) ────────────────────────
        for speaker, speaker_lines in sorted(by_speaker.items()):
            write_txt(
                os.path.join(output_dir, f"{clip_id}_{speaker}_tagged_langdetect.txt"),
                speaker_lines
            )

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

    Output per clip (same for both):
      output/
        P02_pre/
          P02_pre_010/
            P02_pre_010_tagged_langdetect.txt              ← no SPEECH
            P02_pre_010_with_SPEECH_tagged_langdetect.txt  ← with SPEECH
            P02_pre_010_FEM_tagged_langdetect.txt
            P02_pre_010_MAL_tagged_langdetect.txt
            P02_pre_010_KCHI_tagged_langdetect.txt
            P02_pre_010_OCH_tagged_langdetect.txt
            ... (only non-SPEECH speakers present in that clip)
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
