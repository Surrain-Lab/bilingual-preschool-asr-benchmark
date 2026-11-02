import os
import re
import argparse
from pathlib import Path

def parse_tagged_line(line):
    """Parse a tagged transcript line and extract components.

    Handles both formats:
      [ts - ts] (lang) [SPEAKER]: text
      [ts - ts] (lang) [SPEAKER] [OVERLAP]: text
    """
    pattern = r'\[([^\]]+)\s*-\s*([^\]]+)\]\s*\((\w+)\)\s*\[(\w+)\](\s*\[OVERLAP\])?:\s*(.*)'
    match = re.match(pattern, line.strip())
    if match:
        start_time, end_time, language, speaker, overlap_marker, text = match.groups()
        return {
            'start_time': start_time.strip(),
            'end_time': end_time.strip(),
            'language': language,
            'speaker': speaker,
            'text': text.strip(),
            'is_overlap': overlap_marker is not None,
        }
    return None


def process_tagged_file(tagged_file_path, audio_filename, output_dir):
    """Process a tagged file and split by speaker."""
    print(f"\nProcessing: {tagged_file_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(tagged_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    by_speaker = {
        'FEM': [],
        'MAL': [],
        'KCHI': [],
        'OCH': [],
        'ADULT': [],      # FEM + MAL
        'CHILDREN': [],   # KCHI + OCH
        'all': [],
        'OVERLAP': [],    # all segments flagged as overlapping speech
    }

    for line in lines:
        parsed = parse_tagged_line(line)
        if parsed:
            speaker = parsed['speaker']

            by_speaker['all'].append(parsed)

            if speaker in ['FEM', 'MAL', 'KCHI', 'OCH']:
                by_speaker[speaker].append(parsed)

            if speaker in ['FEM', 'MAL']:
                by_speaker['ADULT'].append(parsed)

            if speaker in ['KCHI', 'OCH']:
                by_speaker['CHILDREN'].append(parsed)

            if parsed.get('is_overlap'):
                by_speaker['OVERLAP'].append(parsed)

    # Sort merged groups by start_time
    for group in ['ADULT', 'CHILDREN']:
        by_speaker[group].sort(key=lambda x: x['start_time'])

    for speaker, entries in by_speaker.items():
        if not entries:
            continue

        if speaker == 'all':
            tagged_filename       = f"{audio_filename}_tagged.txt"
            clean_filename        = f"{audio_filename}_clean.txt"
            with_unknown_filename = f"{audio_filename}_clean_with_unknown.txt"
        elif speaker == 'OVERLAP':
            tagged_filename       = f"{audio_filename}_OVERLAP_tagged.txt"
            clean_filename        = f"{audio_filename}_OVERLAP_clean.txt"
            with_unknown_filename = f"{audio_filename}_OVERLAP_clean_with_unknown.txt"
        else:
            tagged_filename       = f"{audio_filename}_{speaker}_tagged.txt"
            clean_filename        = f"{audio_filename}_{speaker}_clean.txt"
            with_unknown_filename = f"{audio_filename}_{speaker}_clean_with_unknown.txt"

        # Tagged output — keep overlap lines visible but marked
        tagged_lines = []
        for entry in entries:
            overlap_tag = " [OVERLAP]" if entry.get('is_overlap') else ""
            tagged_lines.append(
                f"[{entry['start_time']} - {entry['end_time']}] ({entry['language']}) [{entry['speaker']}]{overlap_tag}: {entry['text']}"
            )

        # Clean output — include all segments (overlap flag is informational only)
        clean_texts = [entry['text'] for entry in entries if entry['text']]
        skipped_overlap = sum(1 for entry in entries if entry.get('is_overlap'))
        clean_paragraph = ' '.join(clean_texts)

        # Clean with unknown — [unknown] for empty text entries
        with_unknown_texts = [
            entry['text'] if entry['text'] else '[unknown]'
            for entry in entries
        ]
        with_unknown_paragraph = ' '.join(with_unknown_texts)

        tagged_path = os.path.join(output_dir, tagged_filename)
        with open(tagged_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tagged_lines))

        clean_path = os.path.join(output_dir, clean_filename)
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(clean_paragraph)

        with_unknown_path = os.path.join(output_dir, with_unknown_filename)
        with open(with_unknown_path, 'w', encoding='utf-8') as f:
            f.write(with_unknown_paragraph)

        print(f"  Saved: {tagged_filename} ({len(tagged_lines)} lines)")
        print(f"  Saved: {clean_filename} ({len(clean_texts)} segments, {skipped_overlap} overlap segments excluded)")
        print(f"  Saved: {with_unknown_filename} ({len(with_unknown_texts)} segments, {skipped_overlap} overlap segments excluded)")

    print(f"  Output directory: {output_dir}")


def main(input_folder=None, output_folder=None):
    """Process all tagged files in nested subdirectories."""
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' not found")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Find all *_tagged.txt files recursively, exclude *_tagged_with_speech.txt
    tagged_files = [
        f for f in input_path.rglob("*_tagged.txt")
        if not f.name.endswith("_tagged_with_speech.txt")
    ]

    if not tagged_files:
        print(f"No tagged files found in {input_folder}")
        return

    print(f"Found {len(tagged_files)} tagged files to process")
    print(f"Output will be saved to: {output_folder}")

    processed_count = 0
    for tagged_file in sorted(tagged_files):
        clip_folder      = tagged_file.parent
        recording_folder = clip_folder.parent

        audio_name     = clip_folder.name
        recording_name = recording_folder.name

        output_dir = os.path.join(output_folder, recording_name, audio_name)

        try:
            process_tagged_file(str(tagged_file), audio_name, output_dir)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {audio_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed {processed_count} audio files")
    print(f"All outputs organized in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VTC1 tagged transcripts by speaker")
    parser.add_argument("-i", "--input",  required=True, help="Input folder containing tagged transcript files")
    parser.add_argument("-o", "--output", required=True, help="Output folder for per-speaker files")
    args = parser.parse_args()
    main(args.input, args.output)
