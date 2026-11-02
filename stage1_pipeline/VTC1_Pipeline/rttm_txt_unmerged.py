import argparse
from pathlib import Path
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    return str(timedelta(seconds=seconds))

def parse_rttm_line(line):
    """
    Parse an RTTM line and extract relevant information.
    RTTM format: SPEAKER filename channel start_time duration <NA> <NA> speaker_label <NA> <NA>
    Returns: (start_time, end_time, speaker_label) or None if invalid
    """
    parts = line.strip().split()
    
    if len(parts) < 8 or parts[0] != "SPEAKER":
        return None
    
    try:
        start_time = float(parts[3])
        duration = float(parts[4])
        speaker_label = parts[7]
        
        # Skip <NA> labels only
        if speaker_label == "<NA>":
            return None
        
        # Keep CHI and KCHI separate - removed the merge logic
        
        # Calculate end time
        end_time = start_time + duration
        
        return (start_time, end_time, speaker_label)
    
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse line: {line.strip()}")
        return None

def process_rttm_file(rttm_file, output_dir, audio_folder_name):
    """
    Process a single RTTM file and create output files.
    """
    print(f"\n  Processing: {audio_folder_name}")
    
    # Parse all lines
    entries = []
    with open(rttm_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_rttm_line(line)
            if result:
                entries.append(result)
    
    if not entries:
        print(f"    WARNING: No valid speaker entries found!")
        return False
    
    # Sort by start time
    entries.sort(key=lambda x: x[0])
    
    print(f"    Found {len(entries)} speaker segments")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use audio folder name as base name for output files
    base_name = audio_folder_name
    
    # Write main combined file
    main_output_file = output_path / f"{base_name}_all_speakers.txt"
    with open(main_output_file, 'w', encoding='utf-8') as f:
        for start_time, end_time, speaker in entries:
            start_str = format_timestamp(start_time)
            end_str = format_timestamp(end_time)
            line = f"[{start_str} - {end_str}] Speaker: {speaker}\n"
            f.write(line)
    
    print(f"    Main file: {main_output_file.name}")
    
    # Group entries by speaker
    speaker_entries = {}
    for start_time, end_time, speaker in entries:
        if speaker not in speaker_entries:
            speaker_entries[speaker] = []
        speaker_entries[speaker].append((start_time, end_time, speaker))
    
    # Write separate files for each speaker
    for speaker, speaker_data in speaker_entries.items():
        speaker_file = output_path / f"{base_name}_{speaker}.txt"
        
        with open(speaker_file, 'w', encoding='utf-8') as f:
            f.write(f"Speaker: {speaker}\n")
            f.write(f"Total segments: {len(speaker_data)}\n")
            f.write("="*60 + "\n\n")
            
            for start_time, end_time, spk in speaker_data:
                start_str = format_timestamp(start_time)
                end_str = format_timestamp(end_time)
                duration = end_time - start_time
                line = f"[{start_str} - {end_str}] Duration: {duration:.3f}s\n"
                f.write(line)
        
        print(f"    {speaker}: {len(speaker_data)} segments")
    
    # Print statistics
    speakers = sorted(speaker_entries.keys())
    total_duration = sum(end - start for start, end, _ in entries)
    print(f"    Speakers: {', '.join(speakers)} | Total duration: {total_duration:.2f}s")
    
    return True

def find_all_rttm_files(input_base_dir, rttm_filename):
    """
    Recursively find all 'all.rttm' files in the directory structure.
    Returns list of tuples: (rttm_file_path, relative_folder_path)
    """
    input_path = Path(input_base_dir)
    rttm_files = []
    
    # Recursively search for all.rttm files
    for rttm_file in input_path.rglob(rttm_filename):
        # Get the parent folder (the audio folder containing the rttm)
        audio_folder = rttm_file.parent
        
        # Get relative path from input_base_dir to audio_folder
        try:
            relative_path = audio_folder.relative_to(input_path)
            rttm_files.append((rttm_file, relative_path))
        except ValueError:
            # Skip if path is not relative to input_base_dir
            continue
    
    return rttm_files

def process_all_rttm_files(input_base_dir, output_base_dir, rttm_filename):
    """
    Find and process all 'all.rttm' files in the input directory structure.
    Maintains folder structure in output.
    """
    input_path = Path(input_base_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_base_dir}' not found!")
        return
    
    # Find all RTTM files recursively
    rttm_files = find_all_rttm_files(input_base_dir, rttm_filename)
    
    if not rttm_files:
        print(f"No '{rttm_filename}' files found in '{input_base_dir}' or its subdirectories")
        return
    
    print(f"Found {len(rttm_files)} '{rttm_filename}' file(s)")
    print("="*60)
    
    processed_count = 0
    failed_count = 0
    
    for rttm_file, relative_path in sorted(rttm_files):
        # Use the immediate parent folder name as the audio folder name
        audio_folder_name = rttm_file.parent.name
        
        # Create output directory: {output_base}/{relative_path}/{audio_folder_name}/
        # The extra level ensures the speaker tagger finds {recording}/{clip}/ structure
        output_dir = Path(output_base_dir) / relative_path / audio_folder_name
        
        # Process the RTTM file
        success = process_rttm_file(rttm_file, output_dir, audio_folder_name)
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"  Successfully processed: {processed_count} file(s)")
    if failed_count > 0:
        print(f"  Failed/Empty: {failed_count} file(s)")
    print(f"  Output directory: {output_base_dir}")
    print("="*60)

def main():
    print("="*60)
    print("RTTM Batch Processor (Recursive)")
    print("="*60)
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Target file: {RTTM_FILENAME}")
    print("="*60)
    
    process_all_rttm_files(INPUT_BASE_DIR, OUTPUT_BASE_DIR, RTTM_FILENAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VTC1 RTTM files to timestamped text")
    parser.add_argument("-i", "--input",  required=True, help="Input directory containing RTTM files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for timestamped text files")
    parser.add_argument("--rttm-filename", default="all.rttm", help="RTTM filename to look for (default: all.rttm)")
    args = parser.parse_args()
    process_all_rttm_files(args.input, args.output, args.rttm_filename)
