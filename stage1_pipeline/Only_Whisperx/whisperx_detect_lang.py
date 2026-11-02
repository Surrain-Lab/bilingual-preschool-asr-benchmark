import torch
import whisperx
import numpy as np
import argparse
import subprocess
from datetime import timedelta
from pathlib import Path

# ── Fixed constants (not user-configurable) ──────────────────────
CHUNK_DURATION      = 300       # 5-minute chunks
MODEL_SIZE          = "large-v3"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
MIN_CONTEXT_SECONDS = 30
SAMPLE_RATE         = 16000
AUDIO_EXTENSIONS    = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX-Only transcription pipeline: chunk audio and transcribe"
    )
    parser.add_argument("--audio",   required=True, help="Input folder containing WAV audio files")
    parser.add_argument("--chunks",  required=True, help="Folder to store audio chunks")
    parser.add_argument("--output",  required=True, help="Output folder for chunk transcripts")
    args = parser.parse_args()

    INPUT_FOLDER      = args.audio
    CHUNK_BASE_FOLDER = args.chunks
    OUTPUT_BASE_FOLDER = args.output

    print(f"Using device: {DEVICE}")
    Path(CHUNK_BASE_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_BASE_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_FOLDER)
    audio_files_to_process = []
    for ext in AUDIO_EXTENSIONS:
        audio_files_to_process.extend(input_path.rglob(f"*{ext}"))
    audio_files_to_process = sorted(audio_files_to_process)

    if not audio_files_to_process:
        print(f"No audio files found in '{INPUT_FOLDER}/'")
        print(f"Looking for files with extensions: {', '.join(AUDIO_EXTENSIONS)}")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(audio_files_to_process)} audio file(s) to process:")
    for af in audio_files_to_process:
        print(f"  - {af.name}")
    print(f"{'='*60}")

    print(f"\nLoading WhisperX model ({MODEL_SIZE})...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")

    for file_idx, input_audio_path in enumerate(audio_files_to_process, 1):
        print(f"\n{'#'*60}")
        print(f"PROCESSING AUDIO {file_idx}/{len(audio_files_to_process)}: {input_audio_path.name}")
        print(f"{'#'*60}")
    
        # Create subfolder based on audio filename (without extension)
        relative_dir  = input_audio_path.parent.relative_to(input_path)
        clip_name     = input_audio_path.stem
        chunk_folder  = Path(CHUNK_BASE_FOLDER)  / relative_dir / clip_name
        output_folder = Path(OUTPUT_BASE_FOLDER) / relative_dir / clip_name

        # Create specific folders for this audio file
        chunk_folder.mkdir(parents=True, exist_ok=True)
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Chunk folder: {chunk_folder}")
        print(f"Output folder: {output_folder}")

        # ========== STEP 1: CHUNK THE AUDIO ==========
        print(f"\n{'='*60}")
        print("STEP 1: Chunking audio file")
        print(f"{'='*60}")
        print(f"Chunk duration: {CHUNK_DURATION} seconds ({CHUNK_DURATION/60} minutes)")

        # Get audio duration using ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", str(input_audio_path)],
                capture_output=True,
                text=True,
                check=True
            )
            total_duration = float(result.stdout.strip())
            print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        except subprocess.CalledProcessError as e:
            print(f"Error getting audio duration: {e}")
            print(f"Skipping {input_audio_path.name}")
            continue
        except FileNotFoundError:
            print("Error: ffmpeg/ffprobe not found. Please install ffmpeg.")
            exit()

        # Calculate number of chunks
        num_chunks = int(np.ceil(total_duration / CHUNK_DURATION))
        print(f"Will create {num_chunks} chunk(s)")

        # Clear existing chunks in the folder
        existing_chunks = list(chunk_folder.glob("chunk_*.wav"))
        if existing_chunks:
            print(f"Removing {len(existing_chunks)} existing chunk(s)...")
            for chunk in existing_chunks:
                chunk.unlink()

        # Create chunks using ffmpeg
        print("\nCreating chunks...")
        for i in range(num_chunks):
            start_time = i * CHUNK_DURATION
            chunk_filename = chunk_folder / f"chunk_{i:08d}.wav"
        
            cmd = [
                "ffmpeg",
                "-i", str(input_audio_path),
                "-ss", str(start_time),
                "-t", str(CHUNK_DURATION),
                "-ar", str(SAMPLE_RATE),  # Resample to 16kHz for WhisperX
                "-ac", "1",  # Convert to mono
                "-y",  # Overwrite output file
                str(chunk_filename)
            ]
        
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"Created: {chunk_filename.name}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating {chunk_filename.name}: {e}")
                continue

        # ========== STEP 2: TRANSCRIBE ALL CHUNKS ==========
        print(f"\n{'='*60}")
        print("STEP 2: Transcribing audio chunks")
        print(f"{'='*60}")

        # Get all .wav files from the chunk folder
        chunk_files = sorted(chunk_folder.glob("chunk_*.wav"))

        if not chunk_files:
            print(f"No chunk files found in {chunk_folder}")
            continue

        print(f"Found {len(chunk_files)} chunk(s) to transcribe")

        # Process each chunk
        for idx, audio_file in enumerate(chunk_files, 1):
            print(f"\n{'-'*60}")
            print(f"Processing chunk {idx}/{len(chunk_files)}: {audio_file.name}")
            print(f"{'-'*60}")
        
            # Define output transcript path
            transcript_file = output_folder / f"{audio_file.stem}.txt"
        
            try:
                # 1. Load audio
                print("Loading audio...")
                audio = whisperx.load_audio(str(audio_file))
            
                # If audio shorter than MIN_CONTEXT_SECONDS, pad with silence
                min_samples = int(MIN_CONTEXT_SECONDS * SAMPLE_RATE)
                if len(audio) < min_samples:
                    pad_len = min_samples - len(audio)
                    audio = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])

                # 2. Transcribe with WhisperX
                print("Transcribing...")
                result = model.transcribe(audio, batch_size=16, task="transcribe")

                detected_language = result.get("language", "unknown")
                print(f"Detected language: {detected_language}")

                # Language detection refinement
                try:
                    if detected_language not in ["en", "es"]:
                        try:
                            acoustic_lang = model.detect_language(audio)
                        except Exception:
                            acoustic_lang = None

                        if acoustic_lang and acoustic_lang.startswith("es"):
                            detected_language = "es"
                        elif acoustic_lang and acoustic_lang.startswith("en"):
                            detected_language = "en"
                        else:
                            print(f"Global language detection ambiguous ({detected_language}), falling back to 'unknown'. Acoustic check: {acoustic_lang}")
                            detected_language = "unknown"
                except Exception:
                    pass

                # 3. Align whisper output
                print("Aligning transcript...")
                result_aligned = None
                alignment_used = False
                try:
                    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
                    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
                    alignment_used = True
                except ValueError as e:
                    # No default alignment model for this language (common when detected_language is 'unknown').
                    # Fall back to using the raw segments from the transcribe result so we can continue.
                    print(f"No align model available for language '{detected_language}': {e}")
                    print("Falling back to raw transcribe segments (no alignment). Continuing to process later segments...")
                    result_aligned = {"segments": result.get("segments", [])}
                except Exception as e:
                    # Any other unexpected error when attempting to align - fall back but log the problem
                    print(f"Alignment failed with error: {e}. Falling back to raw segments.")
                    result_aligned = {"segments": result.get("segments", [])}

                # 4. Format and save transcript
                transcript_lines = []

                for seg in result_aligned["segments"]:
                    start_time = str(timedelta(seconds=seg["start"]))
                    end_time = str(timedelta(seconds=seg["end"]))
                    text = seg["text"].strip()
                
                    # Per-segment language detection and correction
                    try:
                        s = max(0, int(seg["start"] * SAMPLE_RATE))
                        e = min(len(audio), int(seg["end"] * SAMPLE_RATE))
                        audio_slice = audio[s:e]
                    
                        if len(audio_slice) < min_samples:
                            pad_len = min_samples - len(audio_slice)
                            audio_slice = np.concatenate([audio_slice, np.zeros(pad_len, dtype=audio.dtype)])

                        if len(audio_slice) > 100:
                            try:
                                seg_lang_det = model.detect_language(audio_slice)
                            except Exception:
                                seg_lang_det = None

                            if seg_lang_det and str(seg_lang_det).startswith("es"):
                                try:
                                    sub_res = model.transcribe(audio_slice, batch_size=1, task="transcribe", language="es")
                                    if sub_res and isinstance(sub_res, dict) and sub_res.get("segments"):
                                        text = sub_res["segments"][0].get("text", text).strip()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                
                    # Language tag assignment
                    seg_lang = seg.get("language") if isinstance(seg, dict) else None
                    lang_tag = None
                    if seg_lang:
                        lang_tag = 'es' if str(seg_lang).startswith('es') else ('en' if str(seg_lang).startswith('en') else seg_lang)
                    else:
                        lang_tag = detected_language if detected_language in ["en", "es"] else detected_language

                    # Heuristic fallback for Spanish detection
                    # Step 7: expanded Spanish indicators for bilingual child recordings
                    _SPANISH_INDICATORS = [
                        '¿', '¡', 'ñ', 'á', 'é', 'í', 'ó', 'ú',
                        'qué', 'por qué', 'cómo', 'dónde', 'cuándo', 'quién', 'cuál',
                        'porque', 'gracias', 'hola', 'adiós', 'también', 'siempre',
                        ' el ', ' la ', ' los ', ' las ', ' un ', ' una ', ' unos ', ' unas ',
                        ' de ', ' del ', ' en ', ' con ', ' por ', ' para ', ' sin ', ' al ',
                        'está', 'tiene', 'tienes', 'tengo', 'quiero', 'quiere',
                        'vamos', 'puedo', 'puede', 'soy', 'eres', 'están',
                        'mira', 'dame', 'ven ', 'dime',
                        'usted', ' yo ', ' tú ', ' él ', ' ella ', ' nosotros ',
                        ' te ', ' se ', ' me ', ' le ', ' lo ', ' mi ', ' tu ',
                        'mamá', 'papá', 'muy ', 'más ', 'todo ', 'todos',
                        'aquí', 'allá', 'ahora', 'bien', 'malo', 'bueno',
                    ]

                    def _looks_like_spanish(t: str) -> bool:
                        if not t:
                            return False
                        low = t.lower()
                        for w in _SPANISH_INDICATORS:
                            if w in low:
                                return True
                        return False

                    try:
                        if isinstance(lang_tag, str) and (str(lang_tag).startswith('en') or str(lang_tag) == 'unknown'):
                            if _looks_like_spanish(text):
                                lang_tag = 'es'
                    except Exception:
                        pass
                
                    # If language is unknown (not en or es), set text to empty
                    if str(lang_tag) not in ['en', 'es']:
                        text = ""  # Blank text for unknown languages
                        lang_tag = "unknown"
                
                    line = f"[{start_time} - {end_time}] ({lang_tag}): {text}"
                    transcript_lines.append(line)

                # Save transcript to file
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(transcript_lines))

                print(f"Transcription saved to {transcript_file}")
                print(f"Total segments: {len(result_aligned['segments'])}")
            
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")
                continue

        # Summary for this audio file
        print(f"\n{'='*60}")
        print(f"Completed processing: {input_audio_path.name}")
        print(f"Created {num_chunks} chunk(s) in '{chunk_folder}/'")
        print(f"Transcribed {len(chunk_files)} file(s) to '{output_folder}/'")
        print(f"{'='*60}")

    print(f"\n{'#'*60}")
    print("ALL PROCESSING COMPLETE!")
    print(f"{'#'*60}")
    print(f"Processed {len(audio_files_to_process)} audio file(s)")
    print(f"Chunks saved to '{CHUNK_BASE_FOLDER}/'")
    print(f"Transcripts saved to '{OUTPUT_BASE_FOLDER}/'")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
