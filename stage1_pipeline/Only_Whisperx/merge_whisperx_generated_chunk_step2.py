import re
import argparse
from pathlib import Path

CHUNK_PATTERN = r"^chunk_\d+\.txt$"

def extract_chunk_number(filename):
    """
    Extract the chunk number from filename like 'chunk_00000000.txt', 'chunk_00000001.txt'
    Returns the number as integer for proper sorting
    """
    match = re.search(r'chunk_(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    return -1

def merge_chunks_for_folder(audio_folder_path, output_dir):
    """
    Merge all chunk files in a single audio folder.
    """
    audio_folder_name = audio_folder_path.name
    
    # Find all chunk files in this folder
    chunk_files = []
    for file in audio_folder_path.iterdir():
        if file.is_file() and re.match(CHUNK_PATTERN, file.name):
            chunk_files.append(file)
    
    if not chunk_files:
        print(f"  WARNING: No chunk files found in: {audio_folder_name}")
        return False
    
    # Sort chunk files by their number (chunk0, chunk1, chunk2, ...)
    chunk_files.sort(key=lambda x: extract_chunk_number(x.name))
    
    print(f"\n  Processing: {audio_folder_name}")
    print(f"    Found {len(chunk_files)} chunk file(s): {', '.join([f.name for f in chunk_files])}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create merged output filename
    merged_filename = f"{audio_folder_name}_all_merged_chunks.txt"
    merged_filepath = output_path / merged_filename
    
    # Merge all chunks into one file
    total_lines = 0
    with open(merged_filepath, 'w', encoding='utf-8') as outfile:
        for i, chunk_file in enumerate(chunk_files):
            # Add separator between chunks (except for the first one)
            if i > 0:
                outfile.write("\n")  # Add blank line between chunks
            
            # Write chunk header
            outfile.write(f"# ===== {chunk_file.name} =====\n")
            
            # Read and write chunk content
            with open(chunk_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                
                # Count lines (excluding empty lines)
                lines = [line for line in content.split('\n') if line.strip()]
                total_lines += len(lines)
            
            # Add newline at end of chunk if not present
            if not content.endswith('\n'):
                outfile.write('\n')
    
    print(f"    Merged into: {merged_filename}")
    print(f"    Total lines: {total_lines}")
    
    return True

def merge_all_chunks(input_base_dir, output_base_dir):
    """
    Find all audio folders and merge their chunk files.
    Maintains folder structure in output.
    """
    input_path = Path(input_base_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_base_dir}' not found!")
        return
    
    # Find all directories that contain chunk files
    audio_folders = []
    
    # Search recursively for directories containing chunk files
    for item in input_path.rglob("*"):
        if item.is_dir():
            # Check if this directory contains any chunk files
            has_chunks = any(
                f.is_file() and re.match(CHUNK_PATTERN, f.name)
                for f in item.iterdir()
            )
            if has_chunks:
                try:
                    relative_path = item.relative_to(input_path)
                    audio_folders.append((item, relative_path))
                except ValueError:
                    continue
    
    if not audio_folders:
        print(f"No folders with chunk files found in '{input_base_dir}'")
        return
    
    print(f"Found {len(audio_folders)} folder(s) with chunk files")
    print("="*60)
    
    processed_count = 0
    skipped_count = 0
    
    for audio_folder, relative_path in sorted(audio_folders):
        # Create output directory maintaining the relative path structure
        output_dir = Path(output_base_dir) / relative_path
        
        # Merge chunks for this folder
        success = merge_chunks_for_folder(audio_folder, output_dir)
        
        if success:
            processed_count += 1
        else:
            skipped_count += 1
    
    print("\n" + "="*60)
    print(f"Merging complete!")
    print(f"  Successfully processed: {processed_count} folder(s)")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} folder(s)")
    print(f"  Output directory: {output_base_dir}")
    print("="*60)

def main():
    print("="*60)
    print("Chunk File Merger")
    print("="*60)
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Chunk pattern: {CHUNK_PATTERN}")
    print("="*60)
    
    merge_all_chunks(INPUT_BASE_DIR, OUTPUT_BASE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge WhisperX chunk transcript files into single file per audio")
    parser.add_argument("-i", "--input",  required=True, help="Input directory containing per-audio chunk folders")
    parser.add_argument("-o", "--output", required=True, help="Output directory for merged transcripts")
    args = parser.parse_args()
    merge_all_chunks(args.input, args.output)
