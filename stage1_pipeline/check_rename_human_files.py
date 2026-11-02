#!/usr/bin/env python3
"""
check_rename_human_files.py
============================
Checks that human-annotated transcript filenames match their corresponding
WAV file names (same clip ID, no extra suffixes). Renames them if needed.

Expected project structure:

  audio/                          ← --audio-dir (default)
    ID-0002_pre/            ← recording folder
      ID-0002_pre_001.wav
      ID-0002_pre_002.wav
    ID-0004_pre/
      ID-0004_pre_010.wav

  annotated-text/                 ← --human-dir (default)
    ID-0002_pre/            ← same recording folder names
      ID-0002_pre_001_FINAL_human.txt
      ID-0002_pre_002_FINAL_human.txt
    ID-0004_pre/
      ID-0004_pre_010_FINAL_human.txt

Matching logic (per recording folder):
  WAV stem : ID-0002_pre_001
  Human txt: ID-0002_pre_001_FINAL_human.txt → renamed to ID-0002_pre_001.txt

Run (dry run — shows what WOULD happen, no changes):
    python check_rename_human_files.py --dry-run

Run (actually rename):
    python check_rename_human_files.py

Override default directories:
    python check_rename_human_files.py --audio-dir /path/to/audio --human-dir /path/to/annotated-text
"""

import os
import re
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIG — default directories relative to this script's location
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DEFAULT_AUDIO = SCRIPT_DIR / "audio"
DEFAULT_HUMAN = SCRIPT_DIR / "annotated-text"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def extract_clip_id(filename_stem):
    """
    Extract the base clip ID from a filename stem by stripping extra suffixes.

    Examples:
      ID-0002_pre_001_FINAL_human  →  ID-0002_pre_001
      ID-0002_pre_001              →  ID-0002_pre_001  (unchanged)
      P01_W2_Interaction_5min_book       →  P01_W2_Interaction
      P02_Interaction                    →  P02_Interaction         (unchanged)
    """
    # Study-style: PREFIX-DIGITS_word_DIGITS e.g. ID-0002_pre_001 (stop at next _ that leads to non-numeric)
    m = re.match(r'^([A-Z]+-\d+_\w+_\d+)', filename_stem)
    if m:
        return m.group(1)

    # P-style with W-suffix: P01_W2_Interaction → P01_W2_Interaction
    m = re.match(r'^(P\d+(?:_W\d+)?_Interaction)', filename_stem)
    if m:
        return m.group(1)

    # P-style simple: P02_Interaction
    m = re.match(r'^(P\d+_Interaction)', filename_stem)
    if m:
        return m.group(1)

    # Fallback — return stem unchanged
    return filename_stem


def get_recording_folders(base_dir):
    """
    Return all immediate subdirectories of base_dir as recording folders.
    Also includes base_dir itself if it directly contains audio/txt files.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    subdirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    return subdirs if subdirs else [base]


def find_wavs_in_folder(folder):
    """Find all WAV files directly inside a recording folder (non-recursive)."""
    wavs = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            wavs[p.stem] = p
    return wavs


def find_txts_in_folder(folder):
    """Find all .txt files directly inside a recording folder (non-recursive)."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt")


# ─────────────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────────────

def match_recording(wav_stems, human_txts):
    """
    Match WAV stems to human txt files within the same recording folder.

    Returns:
      matched        : list of (wav_stem, human_path, needs_rename)
      unmatched_wav  : list of wav_stems with no human file found
      unmatched_human: list of human paths with no WAV match
    """
    # Build clip_id → txt_path lookup, preferring exact matches
    clip_to_human = {}
    for txt_path in human_txts:
        clip_id = extract_clip_id(txt_path.stem)
        if clip_id not in clip_to_human:
            clip_to_human[clip_id] = txt_path
        elif txt_path.stem == clip_id:
            # Exact match — prefer it over a suffixed version
            clip_to_human[clip_id] = txt_path

    matched        = []
    unmatched_wav  = []
    used_human     = set()

    for wav_stem in sorted(wav_stems):
        wav_path = wav_stems[wav_stem]
        clip_id  = extract_clip_id(wav_stem)

        human_path = clip_to_human.get(clip_id)

        # Fallback: direct stem match
        if human_path is None:
            for txt_path in human_txts:
                if txt_path.stem == wav_stem:
                    human_path = txt_path
                    break

        if human_path:
            used_human.add(human_path)
            needs_rename = human_path.stem != wav_stem
            matched.append((wav_stem, human_path, needs_rename))
        else:
            unmatched_wav.append((wav_stem, wav_path))

    unmatched_human = [p for p in human_txts if p not in used_human]
    return matched, unmatched_wav, unmatched_human


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check and rename human transcript files to match WAV names.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio-dir", default=str(DEFAULT_AUDIO),
        help=f"Base audio directory (default: {DEFAULT_AUDIO})"
    )
    parser.add_argument(
        "--human-dir", default=str(DEFAULT_HUMAN),
        help=f"Base annotated-text directory (default: {DEFAULT_HUMAN})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only — do not rename anything"
    )
    args = parser.parse_args()

    dry  = args.dry_run
    mode = "DRY RUN — no files will be changed" if dry else "LIVE RUN — files will be renamed"

    print("=" * 70)
    print("Human Transcript File Checker / Renamer")
    print(f"Mode      : {mode}")
    print(f"Audio dir : {args.audio_dir}")
    print(f"Human dir : {args.human_dir}")
    print("=" * 70)

    audio_base = Path(args.audio_dir)
    human_base = Path(args.human_dir)

    if not audio_base.exists():
        print(f"\nERROR: Audio directory not found: {audio_base}")
        return
    if not human_base.exists():
        print(f"\nERROR: Human transcript directory not found: {human_base}")
        return

    # Get recording folders from audio side
    audio_recording_folders = get_recording_folders(audio_base)

    if not audio_recording_folders:
        print("\nERROR: No recording folders found in audio directory.")
        return

    print(f"\nFound {len(audio_recording_folders)} recording folder(s) in audio dir\n")

    # Totals across all recordings
    total_wav        = 0
    total_correct    = 0
    total_renamed    = 0
    total_no_human   = 0
    total_orphan     = 0
    total_error      = 0

    for audio_rec_folder in audio_recording_folders:
        rec_name = audio_rec_folder.name

        # Find matching human folder (same recording name)
        human_rec_folder = human_base / rec_name

        print(f"{'─'*70}")
        print(f"Recording: {rec_name}")

        # Find WAVs
        wav_stems = find_wavs_in_folder(audio_rec_folder)
        if not wav_stems:
            print(f"  WARNING: No audio files found — skipping")
            print()
            continue

        # Find human txts
        if not human_rec_folder.exists():
            print(f"  ERROR: No matching human folder: {human_rec_folder}")
            print(f"     Expected: annotated-text/{rec_name}/")
            print(f"     Missing transcripts for {len(wav_stems)} WAV file(s)")
            total_wav      += len(wav_stems)
            total_no_human += len(wav_stems)
            print()
            continue

        human_txts = find_txts_in_folder(human_rec_folder)

        print(f"  WAV files  : {len(wav_stems)}")
        print(f"  TXT files  : {len(human_txts)}")

        # Match within this recording
        matched, unmatched_wav, unmatched_human = match_recording(wav_stems, human_txts)

        total_wav += len(wav_stems)

        # Already correct
        already_correct = [(s, p) for s, p, n in matched if not n]
        if already_correct:
            print(f"\n  Already matching ({len(already_correct)}):")
            for wav_stem, p in already_correct:
                print(f"     {p.name}")
            total_correct += len(already_correct)

        # Needs rename
        to_rename = [(s, p) for s, p, n in matched if n]
        if to_rename:
            label = "Would rename" if dry else "Renaming"
            print(f"\n  {'' if dry else ''} {label} ({len(to_rename)}):")
            for wav_stem, human_path in to_rename:
                new_name = wav_stem + ".txt"
                new_path = human_path.parent / new_name
                print(f"     {human_path.name}")
                print(f"     → {new_name}")

                if new_path.exists() and new_path != human_path:
                    print(f"     WARNING: Target already exists — skipping")
                    total_error += 1
                elif not dry:
                    try:
                        human_path.rename(new_path)
                        print(f"     Done")
                        total_renamed += 1
                    except Exception as e:
                        print(f"     ERROR: Error: {e}")
                        total_error += 1
                else:
                    total_renamed += 1

        # WAVs with no transcript
        if unmatched_wav:
            print(f"\n  WARNING: WAVs with no matching transcript ({len(unmatched_wav)}):")
            for wav_stem, wav_path in unmatched_wav:
                print(f"     {wav_path.name}  →  expected: {wav_stem}.txt")
            total_no_human += len(unmatched_wav)

        # Orphan transcripts
        if unmatched_human:
            print(f"\n  WARNING: Transcripts with no matching WAV ({len(unmatched_human)}):")
            for p in unmatched_human:
                print(f"     {p.name}")
            total_orphan += len(unmatched_human)

        print()

    # ── Final summary ─────────────────────────────────────────────
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total WAV files           : {total_wav}")
    print(f"  Already correctly named   : {total_correct}")
    print(f"  {'Would rename' if dry else 'Renamed successfully'}  : {total_renamed}")
    print(f"  WAVs with no transcript   : {total_no_human}")
    print(f"  Orphan transcripts        : {total_orphan}")
    if total_error:
        print(f"  Errors / skipped          : {total_error}")
    print("=" * 70)

    if dry and total_renamed > 0:
        print("\n  Run without --dry-run to apply the renames.")
    if total_no_human > 0:
        print(f"\n  WARNING: {total_no_human} WAV file(s) have no human transcript.")
        print(f"     These clips cannot be used for evaluation.")


if __name__ == "__main__":
    main()
