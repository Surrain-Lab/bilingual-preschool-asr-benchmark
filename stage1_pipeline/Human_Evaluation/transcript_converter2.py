#!/usr/bin/env python3
import re
import os
import unicodedata

def remove_diacritics(text):
    """Remove diacritics from Spanish words"""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

def clean_text_for_comparison(text):
    """Clean text according to preprocessing rules for comparison with automated transcripts"""
    
    # 1. Handle [: formal equivalent] - extract the formal form
    text = re.sub(r'<[^>]*>\s*\[:\s*([^\]]+)\]', r'\1', text)
    
    # 2. Remove content after =! (like [=! laughs])
    text = re.sub(r'\[=![^\]]*\]', '', text)
    
    # 3. Extract words from <> and remove the brackets
    text = re.sub(r'<([^>]+)>', r'\1', text)
    
    # 4. Extract words from {} and remove the braces (English words)
    text = re.sub(r'\{([^}]+)\}', r'\1', text)
    
    # 5. Remove @tags
    text = re.sub(r'@\w+', '', text)
    
    # 6. Remove &=, &~, &+ markers and any words following them
    text = re.sub(r'&[=~+]\w*', '', text)
    text = re.sub(r'&-\w*', '', text)
    
    # 7. Remove xxx (case insensitive, as standalone word)
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)
    
    # 8. Remove punctuation markers like +//, +/, +<, +...
    text = re.sub(r'\+[/\.<]+', '', text)
    
    # 9. Remove all punctuation marks (.?!,;:)
    text = re.sub(r'[.?!,;:]', '', text)
    
    # 10. Remove square brackets and their content (if any remain)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 11. Remove any remaining words starting with @ or &
    text = re.sub(r'[@&][^\s]+', '', text)
    
    # 12. Remove any remaining + and - symbols
    text = text.replace('+', '')
    text = text.replace('-', '')
    
    # 13. Clean up extra spaces
    text = ' '.join(text.split())
    
    # 14. Remove inverted Spanish punctuation
    text = re.sub(r'[¿¡]', '', text)
    
    # 15. Normalize contractions
    text = re.sub(r'\bde el\b', 'del', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba el\b', 'al', text, flags=re.IGNORECASE)
    
    # 16. Remove diacritics from Spanish words
    text = remove_diacritics(text)
    
    # 17. Convert to lowercase for comparison
    text = text.lower()
    
    return text.strip()

def clean_text_for_tagged(text):
    """Clean text for tagged files - same as clean_text_for_comparison but keeps
    original diacritics and original casing (for human readability)"""
    
    # 1. Handle [: formal equivalent] - extract the formal form
    text = re.sub(r'<[^>]*>\s*\[:\s*([^\]]+)\]', r'\1', text)
    # 2. Remove content after =!
    text = re.sub(r'\[=![^\]]*\]', '', text)
    # 3. Extract words from <>
    text = re.sub(r'<([^>]+)>', r'\1', text)
    # 4. Extract words from {}
    text = re.sub(r'\{([^}]+)\}', r'\1', text)
    # 5. Remove @tags
    text = re.sub(r'@\w+', '', text)
    # 6. Remove &=, &~, &+ markers
    text = re.sub(r'&[=~+]\w*', '', text)
    text = re.sub(r'&-\w*', '', text)
    # 7. Remove xxx
    text = re.sub(r'\bxxx\b', '', text, flags=re.IGNORECASE)
    # 8. Remove punctuation markers like +//, +/, +<, +...
    text = re.sub(r'\+[/\.<]+', '', text)
    # 9. Remove all punctuation marks
    text = re.sub(r'[.?!,;:]', '', text)
    # 10. Remove square brackets and their content
    text = re.sub(r'\[.*?\]', '', text)
    # 11. Remove any remaining words starting with @ or &
    text = re.sub(r'[@&][^\s]+', '', text)
    # 12. Remove any remaining + and - symbols
    text = text.replace('+', '')
    text = text.replace('-', '')
    # 13. Clean up extra spaces
    text = ' '.join(text.split())
    # 14. Remove inverted Spanish punctuation
    text = re.sub(r'[¿¡]', '', text)
    # 15. Normalize contractions
    text = re.sub(r'\bde el\b', 'del', text, flags=re.IGNORECASE)
    text = re.sub(r'\ba el\b', 'al', text, flags=re.IGNORECASE)
    # NOTE: diacritics kept, case kept — for human readability in tagged files
    
    return text.strip()


def map_speaker(speaker_code):
    """Map speaker codes to standard format"""
    # New format uses [MAL], [FEM], [KCHI], [OCH] directly
    code = speaker_code.strip('[]')
    if code in ('MAL', 'FEM', 'KCHI', 'OCH'):
        return code
    # Old format fallback
    if code.startswith('FA'):
        return 'FEM'
    elif code.startswith('MA'):
        return 'MAL'
    elif code.startswith('CHI') or code.startswith('Tar'):
        return 'KCHI'
    return None

def parse_timestamped_line(line):
    """
    Parse a line in the new timestamped format:
    [0:00:01.030000  - 0:00:01.670000] (en) [MAL]: { alright } .
    Returns (start_ts, end_ts, lang, speaker_code, text) or None if not matched.
    """
    pattern = r'^\[([^\]]+)\]\s*\(([^)]+)\)\s*\[([^\]]+)\]:\s*(.*)$'
    match = re.match(pattern, line.strip())
    if not match:
        return None
    
    timestamp_raw = match.group(1).strip()
    lang = match.group(2).strip()
    speaker_code = match.group(3).strip()
    text = match.group(4).strip()
    
    # Parse start and end from timestamp (e.g. "0:00:01.030000  - 0:00:01.670000")
    ts_parts = timestamp_raw.split('-')
    if len(ts_parts) == 2:
        start_ts = ts_parts[0].strip()
        end_ts = ts_parts[1].strip()
        # Truncate to seconds (remove sub-second decimals) for cleaner display
        start_ts = re.sub(r'(\d+:\d+:\d+)\.\d+', r'\1', start_ts)
        end_ts = re.sub(r'(\d+:\d+:\d+)\.\d+', r'\1', end_ts)
    else:
        start_ts = timestamp_raw
        end_ts = ''
    
    return start_ts, end_ts, lang, speaker_code, text

def parse_old_format_line(line):
    """
    Parse a line in the old format:
    MA1   { alright } .
    Returns (speaker_code, text) or None if not matched.
    """
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]

def process_transcript(input_file, output_base='five_mins_benchmark'):
    """Process transcript file and create clean text files for comparison with automated transcripts"""

    # Get base filename without extension
    filename = os.path.basename(input_file)
    base_name_full = filename.rsplit('.', 1)[0]  # e.g. R02_pre_001_FINAL_human

    # Extract base name up to clip number (_NNN) by finding the last _NNN or -NNN segment
    clip_match = re.match(r'^(.+_\d+)(?:[-_].+)?$', base_name_full)
    if clip_match:
        base_name = clip_match.group(1)
    else:
        # No clip number — strip trailing _human suffix if present so that
        # the GT naming matches the model-output naming (e.g. P12_W2_Interaction)
        base_name = re.sub(r'_human$', '', base_name_full, flags=re.IGNORECASE)

    # Extract recording name: everything up to the last _NNN segment
    recording_match = re.match(r'^(.+)_(\d+)$', base_name)
    if recording_match:
        recording_name = recording_match.group(1)  # e.g. R02_pre
    else:
        recording_name = base_name

    # Create output directory: output_base/recording_name/base_name/
    os.makedirs(output_base, exist_ok=True)
    output_dir = os.path.join(output_base, recording_name, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store data
    clean_texts_all = []
    clean_texts_by_speaker = {
        'MAL': [], 'FEM': [], 'KCHI': [], 'OCH': [],
        'ADULT': [],    # MAL + FEM
        'CHILDREN': []  # KCHI + OCH
    }
    tagged_texts = []
    tagged_texts_by_speaker = {
        'MAL': [], 'FEM': [], 'KCHI': [], 'OCH': [],
        'ADULT': [],    # MAL + FEM
        'CHILDREN': []  # KCHI + OCH
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try new timestamped format first
        parsed = parse_timestamped_line(line)
        if parsed:
            start_ts, end_ts, lang, speaker_code, text = parsed
            timestamp_str = f"[{start_ts} - {end_ts}]"
            lang_str = f"({lang})"
        else:
            # Fall back to old format
            old_parsed = parse_old_format_line(line)
            if not old_parsed:
                continue
            speaker_code, text = old_parsed
            timestamp_str = None
            lang_str = None
        
        # Map to standard speaker format
        speaker = map_speaker(speaker_code)
        if not speaker:
            continue
        
        # Clean text two ways:
        # - cleaned: diacritics removed + lowercased (for _clean.txt files)
        # - cleaned_tagged: diacritics + casing preserved (for _tagged.txt files)
        cleaned = clean_text_for_comparison(text)
        cleaned_tagged = clean_text_for_tagged(text)
        
        # Skip if text is empty after cleaning
        if not cleaned:
            continue
        
        # Build tagged line (uses original diacritics/casing)
        if timestamp_str and lang_str:
            tagged_line = f"{timestamp_str} {lang_str} [{speaker}]: {cleaned_tagged}"
        else:
            tagged_line = f"[{speaker}]: {cleaned_tagged}"
        
        # Add to overall list
        clean_texts_all.append(cleaned)
        tagged_texts.append(tagged_line)
        
        # Add to speaker-specific list
        clean_texts_by_speaker[speaker].append(cleaned)
        tagged_texts_by_speaker[speaker].append(tagged_line)

        # ADULT = MAL + FEM
        if speaker in ['MAL', 'FEM']:
            clean_texts_by_speaker['ADULT'].append(cleaned)
            tagged_texts_by_speaker['ADULT'].append(tagged_line)

        # CHILDREN = KCHI + OCH
        if speaker in ['KCHI', 'OCH']:
            clean_texts_by_speaker['CHILDREN'].append(cleaned)
            tagged_texts_by_speaker['CHILDREN'].append(tagged_line)
    
    # Write combined clean text file
    with open(os.path.join(output_dir, f'{base_name}_clean.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join(clean_texts_all))
    
    # Write tagged text file (with speaker labels + timestamps)
    with open(os.path.join(output_dir, f'{base_name}_tagged.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tagged_texts))
    
    # Write speaker-specific files
    for speaker in ['MAL', 'FEM', 'KCHI', 'OCH', 'ADULT', 'CHILDREN']:
        with open(os.path.join(output_dir, f'{base_name}_{speaker}_clean.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(clean_texts_by_speaker[speaker]))
        
        with open(os.path.join(output_dir, f'{base_name}_{speaker}_tagged.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(tagged_texts_by_speaker[speaker]))
    
    print("Processing complete!")
    print(f"\nFiles created in folder '{output_dir}/':")
    print(f"  {base_name}_clean.txt")
    print(f"  {base_name}_tagged.txt")
    for sp in ['MAL', 'FEM', 'KCHI', 'OCH', 'ADULT', 'CHILDREN']:
        print(f"  {base_name}_{sp}_clean.txt")
        print(f"  {base_name}_{sp}_tagged.txt")
    
    print(f"\nStatistics:")
    print(f"  Total utterances    : {len(clean_texts_all)}")
    print(f"  MAL utterances      : {len(clean_texts_by_speaker['MAL'])}")
    print(f"  FEM utterances      : {len(clean_texts_by_speaker['FEM'])}")
    print(f"  KCHI utterances     : {len(clean_texts_by_speaker['KCHI'])}")
    print(f"  OCH utterances      : {len(clean_texts_by_speaker['OCH'])}")
    print(f"  ADULT utterances    : {len(clean_texts_by_speaker['ADULT'])}")
    print(f"  CHILDREN utterances : {len(clean_texts_by_speaker['CHILDREN'])}")


def process_folder(input_folder, output_base='five_mins_benchmark'):
    """
    Process all txt files inside recording subfolders.
    Expected structure:
      input_folder/
        R02_pre/
          R02_pre_001_FINAL_human.txt
          R02_pre_002_FINAL_human.txt
        R02_pre2/
          ...
    """
    if not os.path.isdir(input_folder):
        print(f"Folder not found: '{input_folder}'")
        return

    # Collect all .txt files recursively under input_folder
    all_files = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"No .txt files found under '{input_folder}'")
        return

    print(f"Found {len(all_files)} txt file(s) to process")
    print("=" * 50)

    for filepath in all_files:
        print(f"\nProcessing: {filepath}")
        print("-" * 50)
        process_transcript(filepath, output_base=output_base)
        print("=" * 50)

    print(f"\n All files processed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CHAT transcripts to clean reference text")
    parser.add_argument("-i", "--input",  required=True, help="Input folder containing recording subfolders with .txt transcripts")
    parser.add_argument("-o", "--output", required=True, help="Output base directory (e.g. outputs/human_eval/five_mins_benchmark)")
    args = parser.parse_args()
    process_folder(args.input, output_base=args.output)
