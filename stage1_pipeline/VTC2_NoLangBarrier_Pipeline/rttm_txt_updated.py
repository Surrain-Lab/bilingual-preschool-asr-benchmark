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
        
        # Keep all speaker labels as they are (KCHI, OCH, etc.)
        
        # Calculate end time
        end_time = start_time + duration
        
        return (start_time, end_time, speaker_label)
    
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse line: {line.strip()}")
        return None

def process_rttm_file(rttm_file, output_dir, base_name):
    """
    Process a single RTTM file and create output files in a dedicated directory.
    """
    print(f"\n  Processing: {rttm_file.name}")
    
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
    
    # Create output directory - create a subdirectory for this audio file
    output_path = Path(output_dir) / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write main combined file (with audio name prefix)
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
    
    # Write separate files for each speaker (with audio name prefix)
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
    print(f"    Output directory: {output_path}")
    
    return True

def find_all_rttm_files(input_base_dir):
    """
    Recursively find all .rttm files in the directory structure.
    Returns list of tuples: (rttm_file_path, relative_folder_path)
    """
    input_path = Path(input_base_dir)
    rttm_files = []
    
    # Recursively search for all .rttm files
    for rttm_file in input_path.rglob("*.rttm"):
        # Get the parent folder (the folder containing the rttm)
        parent_folder = rttm_file.parent
        
        # Get relative path from input_base_dir to parent_folder
        try:
            relative_path = parent_folder.relative_to(input_path)
            rttm_files.append((rttm_file, relative_path))
        except ValueError:
            # Skip if path is not relative to input_base_dir
            continue
    
    return rttm_files

def process_all_rttm_files(input_base_dir, output_base_dir):
    """
    Find and process all .rttm files in the input directory structure.
    Creates a separate directory for each audio file.
    """
    input_path = Path(input_base_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_base_dir}' not found!")
        return
    
    # Find all RTTM files recursively
    rttm_files = find_all_rttm_files(input_base_dir)
    
    if not rttm_files:
        print(f"No .rttm files found in '{input_base_dir}' or its subdirectories")
        return
    
    print(f"Found {len(rttm_files)} .rttm file(s)")
    print("="*60)
    
    processed_count = 0
    failed_count = 0
    
    for rttm_file, relative_path in sorted(rttm_files):
        # Extract the base name from the RTTM filename (remove .rttm extension)
        base_name = rttm_file.stem
        
        # Create output directory maintaining the relative path structure
        output_dir = Path(output_base_dir) / relative_path
        
        # Process the RTTM file
        success = process_rttm_file(rttm_file, output_dir, base_name)
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
    
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"  Successfully processed: {processed_count} file(s)")
    if failed_count > 0:
        print(f"  Failed/Empty: {failed_count} file(s)")
    print("="*60)

def main():
    print("="*60)
    print("RTTM Batch Processor (Recursive - All .rttm files)")
    print("="*60)
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Processing: All .rttm files")
    print(f"Output structure: Separate directory per audio file")
    print("="*60)
    
    process_all_rttm_files(INPUT_BASE_DIR, OUTPUT_BASE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VTC2 RTTM files to timestamped text")
    parser.add_argument("-i", "--input",  required=True,
                        help="Input dir containing VTC2 raw .rttm files (from New_VTC model output)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for timestamped text files")
    args = parser.parse_args()
    process_all_rttm_files(args.input, args.output)
