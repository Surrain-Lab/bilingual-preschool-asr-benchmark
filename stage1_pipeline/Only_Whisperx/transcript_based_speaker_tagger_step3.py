from pathlib import Path
import re
from datetime import timedelta

# ========== CONFIG ==========
import argparse
# =============================

def parse_timestamp(timestamp_str):
    timestamp_str = timestamp_str.strip('[]')
    parts = timestamp_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    return text.strip()

def parse_speaker_line(line):
    pattern = r'\[([^\]]+)\s*-\s*([^\]]+)\]\s*Speaker:\s*(\w+)'
    match = re.match(pattern, line.strip())
    if match:
        start_time = parse_timestamp(match.group(1).strip())
        end_time   = parse_timestamp(match.group(2).strip())
        speaker    = match.group(3).strip()
        return (start_time, end_time, speaker)
    return None

def parse_transcription_line(line):
    pattern = r'\[([^\]]+)\s*-\s*([^\]]+)\]\s*\((\w+)\):\s*(.*)'
    match = re.match(pattern, line.strip())
    if match:
        start_time = parse_timestamp(match.group(1).strip())
        end_time   = parse_timestamp(match.group(2).strip())
        language   = match.group(3).strip()
        text       = match.group(4).strip()
        return (start_time, end_time, language, text)
    return None

def find_speaker_for_timestamp(start_time, end_time, speaker_segments):
    matched_speakers = []
    for speaker_start, speaker_end, speaker in speaker_segments:
        if start_time < speaker_end and end_time > speaker_start:
            matched_speakers.append(speaker)

    if not matched_speakers:
        return "UNKNOWN"

    # Priority: KCHI > OCH > FEM > MAL > others > SPEECH
    if "KCHI" in matched_speakers:
        return "KCHI"
    elif "OCH" in matched_speakers:
        return "OCH"
    elif "FEM" in matched_speakers:
        return "FEM"
    elif "MAL" in matched_speakers:
        return "MAL"
    else:
        for speaker in matched_speakers:
            if speaker not in ["SPEECH", "UNKNOWN"]:
                return speaker
        return matched_speakers[0]

def process_audio_folder(audio_folder_name, clip_name, speaker_file_dir, source_file_dir, output_dir):
    print(f"\nProcessing: {audio_folder_name} / {clip_name}")

    speaker_folder = Path(speaker_file_dir) / audio_folder_name / clip_name
    source_folder  = Path(source_file_dir)  / audio_folder_name / clip_name

    if not speaker_folder.exists():
        print(f"  Speaker folder not found: {speaker_folder}")
        return False
    if not source_folder.exists():
        print(f"  Source folder not found: {source_folder}")
        return False

    speaker_files = list(speaker_folder.glob("*_all_speakers.txt"))
    if not speaker_files:
        print(f"  No *_all_speakers.txt found in {speaker_folder}")
        return False
    speaker_file = speaker_files[0]

    source_files = list(source_folder.glob("*_all_merged_chunks.txt"))
    if not source_files:
        print(f"  No *_all_merged_chunks.txt found in {source_folder}")
        return False
    source_file = source_files[0]

    print(f"  Speaker file: {speaker_file.name}")
    print(f"  Source file:  {source_file.name}")

    # Read speaker segments, remapping CHI -> OCH
    speaker_segments = []
    with open(speaker_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_speaker_line(line)
            if result:
                start, end, speaker = result
                if speaker == "CHI":
                    speaker = "OCH"
                speaker_segments.append((start, end, speaker))

    if not speaker_segments:
        print(f"  No speaker segments found in {speaker_file.name}")
        return False
    print(f"  Loaded {len(speaker_segments)} speaker segments")

    # Initialize clean text buckets
    # CHILDREN = KCHI + OCH combined (like ADULT = MAL + FEM)
    clean_texts_by_speaker = {
        'KCHI': [], 'OCH': [], 'FEM': [], 'MAL': [],
        'UNKNOWN': [], 'ADULT': [], 'CHILDREN': []
    }

    tagged_lines       = []
    clean_texts_all    = []
    ordered_clean      = []   # (speaker, cleaned_text) in time order, for the overall file
    transcription_count = 0

    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_transcription_line(line)
            if result:
                start_time, end_time, language, text = result
                speaker = find_speaker_for_timestamp(start_time, end_time, speaker_segments)

                cleaned_text = clean_text(text)
                if not cleaned_text:
                    cleaned_text = '[unknown]'

                clean_texts_all.append(cleaned_text)
                ordered_clean.append((speaker, cleaned_text))

                # Per-speaker bucket
                if speaker not in clean_texts_by_speaker:
                    clean_texts_by_speaker[speaker] = []
                clean_texts_by_speaker[speaker].append(cleaned_text)

                # ADULT bucket (MAL + FEM)
                if speaker in ['MAL', 'FEM']:
                    clean_texts_by_speaker['ADULT'].append(cleaned_text)

                # CHILDREN bucket (KCHI + OCH)
                if speaker in ['KCHI', 'OCH']:
                    clean_texts_by_speaker['CHILDREN'].append(cleaned_text)

                start_str  = str(timedelta(seconds=start_time))
                end_str    = str(timedelta(seconds=end_time))
                tagged_line = f"[{start_str} - {end_str}] ({language}) [{speaker}]: {text}\n"
                tagged_lines.append(tagged_line)
                transcription_count += 1
            else:
                if line.strip():
                    tagged_lines.append(line)

    if transcription_count == 0:
        print(f"  No transcription lines found in {source_file.name}")
        return False
    print(f"  Tagged {transcription_count} transcription lines")

    # Output directory
    output_path = Path(output_dir) / audio_folder_name / clip_name
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Unfiltered combined tagged + clean ──
    unfiltered_tagged_file = output_path / f"{clip_name}_unfiltered_tagged.txt"
    with open(unfiltered_tagged_file, 'w', encoding='utf-8') as f:
        f.writelines(tagged_lines)
    print(f"  Unfiltered tagged: {unfiltered_tagged_file.name}")

    unfiltered_clean_file = output_path / f"{clip_name}_unfiltered_clean.txt"
    with open(unfiltered_clean_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(clean_texts_all))
    print(f"  Unfiltered clean:  {unfiltered_clean_file.name}")

    # ── Per-speaker tagged + clean files ──
    speaker_lines = {}
    for line in tagged_lines:
        m = re.search(r'\[(\w+)\]:', line)
        if m:
            spk = m.group(1)
            if spk not in speaker_lines:
                speaker_lines[spk] = []
            speaker_lines[spk].append(line)

    for spk, lines in speaker_lines.items():
        if lines:
            spk_tagged_file = output_path / f"{clip_name}_{spk}_tagged.txt"
            with open(spk_tagged_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"  {spk} tagged: {spk_tagged_file.name} ({len(lines)} lines)")

            if clean_texts_by_speaker.get(spk):
                spk_clean_file = output_path / f"{clip_name}_{spk}_clean.txt"
                with open(spk_clean_file, 'w', encoding='utf-8') as f:
                    f.write(' '.join(clean_texts_by_speaker[spk]))
                print(f"  {spk} clean:  {spk_clean_file.name}")

    # ── ADULT combined tagged + clean ──
    adult_tagged_lines = [l for l in tagged_lines
                          if re.search(r'\[(\w+)\]:', l) and
                          re.search(r'\[(\w+)\]:', l).group(1) in ['MAL', 'FEM']]
    if adult_tagged_lines:
        adult_tagged_file = output_path / f"{clip_name}_ADULT_tagged.txt"
        with open(adult_tagged_file, 'w', encoding='utf-8') as f:
            f.writelines(adult_tagged_lines)
        print(f"  ADULT tagged: {adult_tagged_file.name} ({len(adult_tagged_lines)} lines)")

    if clean_texts_by_speaker['ADULT']:
        adult_clean_file = output_path / f"{clip_name}_ADULT_clean.txt"
        with open(adult_clean_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(clean_texts_by_speaker['ADULT']))
        print(f"  ADULT clean:  {adult_clean_file.name}")

    # ── CHILDREN combined tagged + clean (KCHI + OCH) ──
    children_tagged_lines = [l for l in tagged_lines
                              if re.search(r'\[(\w+)\]:', l) and
                              re.search(r'\[(\w+)\]:', l).group(1) in ['KCHI', 'OCH']]
    if children_tagged_lines:
        children_tagged_file = output_path / f"{clip_name}_CHILDREN_tagged.txt"
        with open(children_tagged_file, 'w', encoding='utf-8') as f:
            f.writelines(children_tagged_lines)
        print(f"  CHILDREN tagged: {children_tagged_file.name} ({len(children_tagged_lines)} lines)")

    if clean_texts_by_speaker['CHILDREN']:
        children_clean_file = output_path / f"{clip_name}_CHILDREN_clean.txt"
        with open(children_clean_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(clean_texts_by_speaker['CHILDREN']))
        print(f"  CHILDREN clean:  {children_clean_file.name}")

    # ── Filtered tagged + clean (exclude UNKNOWN and SPEECH) ──
    filtered_tagged_lines = []
    filtered_clean_texts  = []

    for line in tagged_lines:
        m = re.search(r'\[(\w+)\]:', line)
        if m and m.group(1) not in ['UNKNOWN', 'SPEECH']:
            filtered_tagged_lines.append(line)

    # Keep time order. The overall file is scored against a reference that is in time order,
    # so it must not be grouped by speaker (looping over clean_texts_by_speaker wrote every
    # KCHI word first, then OCH, FEM, MAL, and turned correctly transcribed words into
    # substitutions -- overall WER came out ~13 points too high).
    for spk, text in ordered_clean:
        if spk not in ['UNKNOWN', 'SPEECH']:
            filtered_clean_texts.append(text)

    if filtered_tagged_lines:
        filtered_tagged_file = output_path / f"{clip_name}_tagged.txt"
        with open(filtered_tagged_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_tagged_lines)
        print(f"  Filtered tagged: {filtered_tagged_file.name} ({len(filtered_tagged_lines)} lines)")

    if filtered_clean_texts:
        filtered_clean_file = output_path / f"{clip_name}_clean.txt"
        with open(filtered_clean_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(filtered_clean_texts))
        print(f"  Filtered clean:  {filtered_clean_file.name}")

    # Speaker distribution
    speaker_counts = {spk: len(lines) for spk, lines in speaker_lines.items()}
    print(f"  Speaker distribution: {speaker_counts}")

    return True


def process_all_folders(speaker_file_dir, source_file_dir, output_dir):
    speaker_path = Path(speaker_file_dir)
    source_path  = Path(source_file_dir)

    if not speaker_path.exists():
        print(f"Error: Speaker directory '{speaker_file_dir}' not found!")
        return
    if not source_path.exists():
        print(f"Error: Source directory '{source_file_dir}' not found!")
        return

    audio_folders = []
    for recording in sorted(speaker_path.iterdir()):
        if recording.is_dir():
            for clip in sorted(recording.iterdir()):
                if clip.is_dir():
                    audio_folders.append((recording.name, clip.name))

    if not audio_folders:
        print(f"No folders found in '{speaker_file_dir}'")
        return

    print(f"Found {len(audio_folders)} clip(s)")
    print("=" * 60)

    processed_count = 0
    failed_count    = 0

    for audio_folder, clip_name in audio_folders:
        success = process_audio_folder(audio_folder, clip_name, speaker_file_dir, source_file_dir, output_dir)
        if success:
            processed_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Successfully processed: {processed_count}")
    if failed_count > 0:
        print(f"  Failed: {failed_count}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Assign speaker labels to WhisperX transcripts using VTC1 RTTM timestamps")
    parser.add_argument("--speaker-dir", required=True, help="Directory containing VTC1 rttm_txt speaker timestamp files")
    parser.add_argument("--source-dir",  required=True, help="Directory containing merged WhisperX chunk transcripts")
    parser.add_argument("--output-dir",  required=True, help="Output directory for speaker-tagged transcripts")
    args = parser.parse_args()
    print("=" * 60)
    print("Timestamp-Based Speaker Tagger")
    print("=" * 60)
    print(f"Speaker files: {args.speaker_dir}")
    print(f"Source files:  {args.source_dir}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 60)
    process_all_folders(args.speaker_dir, args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
