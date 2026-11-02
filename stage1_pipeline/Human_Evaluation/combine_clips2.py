#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

# File types to combine per speaker
FILE_TYPES = [
    ('', 'tagged'),
    ('', 'clean'),
    ('MAL', 'tagged'),
    ('MAL', 'clean'),
    ('FEM', 'tagged'),
    ('FEM', 'clean'),
    ('KCHI', 'tagged'),
    ('KCHI', 'clean'),
    ('OCH', 'tagged'),
    ('OCH', 'clean'),
    ('ADULT', 'tagged'),
    ('ADULT', 'clean'),
    ('CHILDREN', 'tagged'),   # <-- add this
    ('CHILDREN', 'clean'),
]

def get_clip_number(clip_folder_name):
    """Extract numeric clip ID for sorting. Example: R02_pre_018 -> 18"""
    match = re.match(r'^.*_(\d+)$', clip_folder_name)
    if match:
        return int(match.group(1))
    return 0

def combine_clips(input_base_dir, output_base_dir):
    """
    New structure:
      human/
        R02_pre/          <- recording folder
          R02_pre_001/    <- clip folder
          R02_pre_002/
        R03_pre/
          R03_pre_070/
    """
    input_path = Path(input_base_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_base_dir}' not found")
        return

    # Find all recording-level folders (one level down from human/)
    recording_folders = [d for d in input_path.iterdir() if d.is_dir()]

    if not recording_folders:
        print(f"No folders found in '{input_base_dir}'")
        return

    print(f"Found {len(recording_folders)} recording folder(s)")
    print("=" * 60)

    os.makedirs(output_base_dir, exist_ok=True)

    for recording_folder in sorted(recording_folders):
        recording_name = recording_folder.name  # e.g. R02_pre

        # Find all clip folders inside this recording folder
        clip_folders = [d for d in recording_folder.iterdir() if d.is_dir()]

        if not clip_folders:
            print(f"WARNING: No clip folders found in '{recording_folder}'")
            continue

        # Sort clips by numeric ID
        clips_sorted = sorted(clip_folders, key=lambda d: get_clip_number(d.name))

        print(f"\nCombining: {recording_name}  ({len(clips_sorted)} clips)")
        print(f"  Clip order: {[c.name for c in clips_sorted]}")

        # Create output directory for this recording
        output_dir = Path(output_base_dir) / recording_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Combine each file type
        for speaker, file_type in FILE_TYPES:
            combined_lines = []
            found_any = False

            for clip_folder in clips_sorted:
                clip_name = clip_folder.name  # e.g. R02_pre_001

                if speaker:
                    filename = f"{clip_name}_{speaker}_{file_type}.txt"
                else:
                    filename = f"{clip_name}_{file_type}.txt"

                filepath = clip_folder / filename

                if filepath.exists():
                    content = filepath.read_text(encoding='utf-8').strip()
                    if content:
                        combined_lines.append(content)
                        found_any = True

            if found_any:
                if speaker:
                    out_filename = f"{recording_name}_{speaker}_{file_type}.txt"
                else:
                    out_filename = f"{recording_name}_{file_type}.txt"

                out_path = output_dir / out_filename

                # tagged: one entry per line | clean: space-separated
                if file_type == 'tagged':
                    combined_content = '\n'.join(combined_lines)
                else:
                    combined_content = ' '.join(combined_lines)

                out_path.write_text(combined_content, encoding='utf-8')
                print(f"   {out_filename}")
            else:
                if speaker:
                    print(f"  WARNING: No data for {speaker}_{file_type} across any clip")

    print(f"\n{'=' * 60}")
    print(f"Done! Combined files saved in: {output_base_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-clip human transcript files into per-recording files")
    parser.add_argument("-i", "--input",  required=True, help="Input directory containing recording folders with per-clip transcript dirs")
    parser.add_argument("-o", "--output", required=True, help="Output directory for combined recording-level files")
    args = parser.parse_args()
    combine_clips(args.input, args.output)
