#!/usr/bin/env python3
"""
split_rttm_by_uri.py
--------------------
VTC1's apply.sh is run per recording FOLDER, producing one all.rttm per recording
whose rows are tagged with a per-clip URI (column 2). The downstream
whisperx_vtc_transcribe step, however, expects ONE timestamp file PER CLIP.

This script splits each recording's all.rttm by URI into per-clip all.rttm files:

    <dst>/<clip_uri>/all.rttm

Only clips that actually exist under the audio dir are emitted, which also drops
stale recordings left over in the shared VTC1 model output folder.

Usage:
    split_rttm_by_uri.py <src_output_voice_type_classifier> <dst_per_clip_dir> <audio_dir>
"""
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)

    src_base  = Path(sys.argv[1])
    dst_base  = Path(sys.argv[2])
    audio_dir = Path(sys.argv[3])

    # Set of clip stems we actually have audio for (e.g. P01_pre_002)
    our_clips = {p.stem for p in audio_dir.rglob("*.wav")}
    if not our_clips:
        print(f"ERROR: no .wav files found under {audio_dir}")
        sys.exit(1)
    print(f"Audio clips found: {len(our_clips)}")

    rttm_files = sorted(src_base.rglob("all.rttm"))
    if not rttm_files:
        print(f"ERROR: no all.rttm found under {src_base}")
        sys.exit(1)
    print(f"Source recording RTTMs: {len(rttm_files)}")

    written, skipped_uris = 0, set()
    for rttm in rttm_files:
        groups = {}
        for line in rttm.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            uri = parts[1]
            if uri not in our_clips:
                skipped_uris.add(uri)
                continue
            groups.setdefault(uri, []).append(line)

        for uri, lines in groups.items():
            d = dst_base / uri
            d.mkdir(parents=True, exist_ok=True)
            (d / "all.rttm").write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1

    print(f"Per-clip RTTMs written: {written}")
    missing = our_clips - {p.parent.name for p in dst_base.rglob("all.rttm")}
    if missing:
        print(f"WARNING: {len(missing)} audio clips had no RTTM rows: {sorted(missing)[:5]}...")
    if skipped_uris:
        print(f"Note: skipped {len(skipped_uris)} URI(s) with no matching audio (stale/extra).")


if __name__ == "__main__":
    main()
