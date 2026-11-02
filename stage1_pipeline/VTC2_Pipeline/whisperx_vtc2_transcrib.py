import os
import torch
import whisperx
import pandas as pd
import numpy as np
from datetime import timedelta
import re
from pathlib import Path

# ========== CONFIG ==========
MODEL_SIZE = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Padding / sampling configuration
MIN_CONTEXT_SECONDS = 30  # Whisper's language detector expects ~30s context
SAMPLE_RATE = 16000

# Speaker types to process
SPEAKER_TYPES = ["all_speakers"]  # Only process all_speakers file
# =============================

print(f"Using device: {DEVICE}")

# Load WhisperX model once
print(f"Loading WhisperX model ({MODEL_SIZE})...")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")


def parse_timestamp_file(txt_path, default_speaker=None):
    """Parse timestamp txt file and return list of segments."""
    segments = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip header lines
            if line.startswith('Speaker:') or line.startswith('Total segments:') or line.startswith('==='):
                continue
            
            # Format 1: [timestamp] Speaker: SPEAKER
            match1 = re.match(r'\[(\d+:\d+:\d+\.\d+)\s*-\s*(\d+:\d+:\d+\.\d+)\]\s*Speaker:\s*(\w+)', line)
            if match1:
                start_str, end_str, speaker = match1.groups()
                
                # Convert timestamp to seconds
                start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_str.split(':'))))
                end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_str.split(':'))))
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker_id': speaker
                })
                continue
            
            # Format 2: [timestamp] Duration: Xs (for individual speaker files)
            match2 = re.match(r'\[(\d+:\d+:\d+\.\d+)\s*-\s*(\d+:\d+:\d+\.\d+)\]', line)
            if match2 and default_speaker:
                start_str, end_str = match2.groups()
                
                # Convert timestamp to seconds
                start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_str.split(':'))))
                end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_str.split(':'))))
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker_id': default_speaker
                })
    
    return segments


CHILD_SPEAKERS = {"KCHI", "OCH", "CHI"}
OVERLAP_TOLERANCE = 0.1  # seconds of overlap required to flag a segment

# Step 7: expanded Spanish indicator list for bilingual child recordings
SPANISH_INDICATORS = [
    # Punctuation / diacritics
    '¿', '¡', 'ñ',
    # Accented vowels
    'á', 'é', 'í', 'ó', 'ú',
    # Question / exclamation words
    'qué', 'por qué', 'cómo', 'dónde', 'cuándo', 'quién', 'cuál',
    # High-frequency unambiguous Spanish words
    'porque', 'gracias', 'hola', 'adiós', 'también', 'siempre',
    # Articles (space-bounded to avoid partial matches in English)
    ' el ', ' la ', ' los ', ' las ', ' un ', ' una ', ' unos ', ' unas ',
    # Prepositions
    ' de ', ' del ', ' en ', ' con ', ' por ', ' para ', ' sin ', ' al ',
    # Common verbs
    'está', 'tiene', 'tienes', 'tengo', 'quiero', 'quiere',
    'vamos', 'puedo', 'puede', 'soy', 'eres', 'están',
    'mira', 'dame', 'ven ', 'dime',
    # Pronouns (space-bounded)
    'usted', ' yo ', ' tú ', ' él ', ' ella ', ' nosotros ',
    ' te ', ' se ', ' me ', ' le ', ' lo ', ' mi ', ' tu ',
    # Common child-speech Spanish words
    'mamá', 'papá', 'muy ', 'más ', 'todo ', 'todos',
    'aquí', 'allá', 'ahora', 'bien', 'malo', 'bueno',
]


def looks_like_spanish(text):
    """Heuristic to detect Spanish text using expanded bilingual word list."""
    if not text:
        return False
    low = text.lower()
    for indicator in SPANISH_INDICATORS:
        if indicator in low:
            return True
    return False


def detect_overlaps(segments, tolerance=OVERLAP_TOLERANCE):
    """Flag segments that temporally overlap with a different speaker."""
    for i, seg in enumerate(segments):
        seg['is_overlap'] = False
        for j, other in enumerate(segments):
            if i == j:
                continue
            if other['speaker_id'] != seg['speaker_id']:
                overlap_dur = (min(seg['end_time'], other['end_time'])
                               - max(seg['start_time'], other['start_time']))
                if overlap_dur > tolerance:
                    seg['is_overlap'] = True
                    break
    return segments


def transcribe_segment(audio_segment, model, device, speaker=None):
    """Transcribe a single audio segment and detect language.
    Uses batch_size=1 for child speakers (KCHI/OCH) to reduce errors on short utterances.
    """
    min_samples = int(MIN_CONTEXT_SECONDS * SAMPLE_RATE)

    # Pad if too short
    if len(audio_segment) < min_samples:
        pad_len = min_samples - len(audio_segment)
        audio_segment = np.concatenate([audio_segment, np.zeros(pad_len, dtype=audio_segment.dtype)])

    # Step 3: use smaller batch for child speakers to reduce cross-utterance contamination
    batch = 1 if speaker in CHILD_SPEAKERS else 16

    try:
        # Transcribe
        result = model.transcribe(audio_segment, batch_size=batch, task="transcribe")
        
        if not result.get("segments"):
            return None, None
        
        detected_lang = result.get("language", "unknown")
        
        # Improve language detection
        if detected_lang not in ["en", "es"]:
            try:
                acoustic = model.detect_language(audio_segment)
                if acoustic and str(acoustic).startswith("es"):
                    detected_lang = "es"
                elif acoustic and str(acoustic).startswith("en"):
                    detected_lang = "en"
                else:
                    detected_lang = "unknown"
            except Exception:
                pass
        
        # Try alignment for better quality
        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
            result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_segment, device)
            text = " ".join([w["word"] for seg in result_aligned["segments"] 
                           for w in seg.get("words", [])]).strip()
        except:
            text = " ".join([seg.get("text", "").strip() for seg in result["segments"]]).strip()
        
        # Final heuristic check
        if detected_lang == 'unknown' and looks_like_spanish(text):
            detected_lang = 'es'
        
        return text, detected_lang
    
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None, None


def process_audio_file(audio_path, timestamp_file, output_dir, model, device):
    """Process a single audio file with its timestamp file."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(audio_path).name}")
    print(f"Timestamp file: {Path(timestamp_file).name}")
    
    # Extract speaker from filename if it's a single-speaker file
    filename = Path(timestamp_file).stem
    default_speaker = None
    for speaker in ['FEM', 'MAL', 'KCHI','CHI']:
        if filename.endswith(f'_{speaker}'):
            default_speaker = speaker
            break
    if default_speaker == 'CHI':
        default_speaker = 'OCH'
    # Parse timestamps
    segments = parse_timestamp_file(timestamp_file, default_speaker)
    if not segments:
        print(f"No segments found in {timestamp_file}")
        return

    # Step 4: flag segments that overlap with a different speaker
    segments = detect_overlaps(segments)

    print(f"Found {len(segments)} segments")
    if default_speaker:
        print(f"Speaker: {default_speaker}")
    
    # Load audio
    print("Loading audio...")
    full_audio = whisperx.load_audio(audio_path)
    
    # Pad audio if needed
    min_samples = int(MIN_CONTEXT_SECONDS * SAMPLE_RATE)
    if len(full_audio) < min_samples:
        pad_len = min_samples - len(full_audio)
        full_audio = np.concatenate([full_audio, np.zeros(pad_len, dtype=full_audio.dtype)])
    
    # Process segments
    results = []
    processed = 0
    
    for seg in segments:
        start_time = seg['start_time']
        end_time = seg['end_time']
        speaker = seg['speaker_id']
        duration = end_time - start_time
        
        # Skip very short segments
        if duration < 0.1:
            continue
        
        # Extract audio segment
        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int(end_time * SAMPLE_RATE)
        audio_segment = full_audio[start_sample:end_sample]
        
        if len(audio_segment) < 1600:  # Less than 0.1s
            continue
        
        # Transcribe (pass speaker so child segments use batch_size=1)
        text, lang = transcribe_segment(audio_segment, model, device, speaker=speaker)

        if text:
            if lang not in ["en", "es"]:
                text = ""
            results.append({
                'start_time': start_time,
                'end_time': end_time,
                'speaker': speaker,
                'text': text,
                'language': lang if lang in ["en", "es"] else "unknown",
                'is_overlap': seg.get('is_overlap', False),
            })
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(segments)} segments...")
    
    print(f"Successfully transcribed {len(results)} segments")
    
    # Sort by timestamp
    results.sort(key=lambda x: x['start_time'])
    
    # Save results
    audio_filename = Path(audio_path).stem
    save_transcriptions(results, output_dir, audio_filename)


def format_timestamp(seconds):
    """Convert seconds to H:MM:SS.ffffff format."""
    return str(timedelta(seconds=seconds))


def save_transcriptions(results, output_dir, audio_filename):
    """Save tagged transcription for all speakers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by timestamp (should already be sorted but ensure it)
    results.sort(key=lambda x: x['start_time'])
    
    # Tagged version: [timestamp] (lang) [SPEAKER]: text
    tagged_lines = []
    for r in results:
        start_str = format_timestamp(r['start_time'])
        end_str = format_timestamp(r['end_time'])
        lang_tag = r['language']
        speaker_tag = r['speaker']
        text = r['text']
        
        overlap_tag = " [OVERLAP]" if r.get('is_overlap') else ""
        line = f"[{start_str} - {end_str}] ({lang_tag}) [{speaker_tag}]{overlap_tag}: {text}"
        tagged_lines.append(line)
    
    # Save tagged file for all speakers
    tagged_filename = f"{audio_filename}_tagged.txt"
    tagged_path = os.path.join(output_dir, tagged_filename)
    with open(tagged_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tagged_lines))
    
    print(f"Saved: {tagged_path} ({len(tagged_lines)} lines)")


def main():
    """Main processing loop."""
    import argparse
    parser = argparse.ArgumentParser(description="VTC2 + WhisperX transcription pipeline")
    parser.add_argument("--audio",  required=True, help="Folder containing WAV audio files")
    parser.add_argument("--rttm",   required=True, help="Folder containing VTC2 rttm_txt timestamp files")
    parser.add_argument("--output", required=True, help="Output folder for tagged transcriptions")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        audio_files.extend(Path(args.audio).rglob(ext))

    if not audio_files:
        print(f"No audio files found in {args.audio}")
        return

    print(f"Found {len(audio_files)} audio files to process")

    for audio_path in audio_files:
        audio_name = audio_path.stem

        timestamp_file = None
        for speaker_type in SPEAKER_TYPES:
            matches = list(Path(args.rttm).rglob(f"{audio_name}_{speaker_type}.txt"))
            if matches:
                timestamp_file = matches[0]
                break

        if not timestamp_file:
            print(f"Warning: No timestamp file found for {audio_name}, skipping...")
            print(f"  Searched recursively under: {args.rttm}")
            continue

        recording_name = audio_path.parent.name
        output_dir = os.path.join(args.output, recording_name, audio_name)

        try:
            process_audio_file(str(audio_path), str(timestamp_file), output_dir, model, DEVICE)
        except Exception as e:
            print(f"Error processing {audio_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
