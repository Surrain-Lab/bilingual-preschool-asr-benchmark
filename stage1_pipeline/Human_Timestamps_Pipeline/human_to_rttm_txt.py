"""
human_to_rttm_txt.py
====================
Build rttm_txt-style timestamp files from HUMAN-annotated transcripts, so the
standard WhisperX segment transcriber (whisperx_vtc2_transcrib.py) can run on
human ("oracle") segmentation + speaker tags instead of VTC1/VTC2 output.

Input  (annotated-text/human_transcripts_timestamps):
    <recording>/<clip>.txt with lines:
        [0:00:03.270000  - 0:00:03.960000] (es) [FEM]: text...

Output (same layout the VTC pipelines consume):
    <out>/<clip>/<clip>_all_speakers.txt   lines: [start - end] Speaker: FEM
    <out>/<clip>/<clip>_FEM.txt            (per-speaker, for parity)
    ...

Notes
-----
* Some files carry duplicate annotation lines (same timestamps + speaker,
  body is a language note like "spa"/"unsure"). We dedup on
  (start, end, speaker) keeping the first occurrence.
* Timestamps are always written with a 6-digit fractional part: the
  transcriber's regex requires `H:MM:SS.ffffff` and silently drops
  fraction-less times like `0:00:30`.
* Empty transcript files (e.g. a clip awaiting annotation) are skipped with
  a warning.

Usage:
    python human_to_rttm_txt.py -i annotated-text/human_transcripts_timestamps \
                                -o output_14clips/human_timestamps/rttm_txt
"""

import argparse
import re
from pathlib import Path

LINE_RE = re.compile(
    r'^\[(\d+:\d+:\d+(?:\.\d+)?)\s*-\s*(\d+:\d+:\d+(?:\.\d+)?)\]'
    r'\s*\([a-z]+\)\s*\[([A-Z]+)\]:'
)

SPEAKERS = ["FEM", "MAL", "KCHI", "OCH"]


def to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fmt(ts_seconds: float) -> str:
    """Format seconds as H:MM:SS.ffffff (always with 6-digit fraction)."""
    h = int(ts_seconds // 3600)
    m = int((ts_seconds % 3600) // 60)
    s = ts_seconds % 60
    return f"{h}:{m:02d}:{s:09.6f}"


def parse_clip(txt_path: Path):
    """Return deduped [(start_sec, end_sec, speaker)] in file order."""
    segments = []
    seen = set()
    bad = 0
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            bad += 1
            continue
        start, end, speaker = m.groups()
        key = (start, end, speaker)
        if key in seen:          # duplicate annotation line (e.g. "spa")
            continue
        seen.add(key)
        segments.append((to_seconds(start), to_seconds(end), speaker))
    return segments, bad


def write_clip(out_root: Path, clip_stem: str, segments):
    clip_dir = out_root / clip_stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"[{fmt(s)} - {fmt(e)}] Speaker: {spk}" for s, e, spk in segments]
    (clip_dir / f"{clip_stem}_all_speakers.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    for spk in SPEAKERS:
        spk_lines = [f"[{fmt(s)} - {fmt(e)}] Speaker: {spk}"
                     for s, e, sp in segments if sp == spk]
        (clip_dir / f"{clip_stem}_{spk}.txt").write_text(
            ("\n".join(spk_lines) + "\n") if spk_lines else "",
            encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", required=True,
                    help="human_transcripts_timestamps dir (recording subfolders)")
    ap.add_argument("-o", "--output", required=True,
                    help="output rttm_txt root (per-clip subfolders)")
    args = ap.parse_args()

    in_root, out_root = Path(args.input), Path(args.output)
    total_clips = total_segs = skipped_empty = total_bad = 0
    spk_counts = {s: 0 for s in SPEAKERS}

    for rec_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        for txt in sorted(rec_dir.glob("*.txt")):
            segments, bad = parse_clip(txt)
            total_bad += bad
            if not segments:
                print(f"  WARNING: SKIP (no segments): {rec_dir.name}/{txt.name}")
                skipped_empty += 1
                continue
            write_clip(out_root, txt.stem, segments)
            total_clips += 1
            total_segs += len(segments)
            for _, _, spk in segments:
                spk_counts[spk] = spk_counts.get(spk, 0) + 1
            print(f"   {txt.stem}: {len(segments)} segments")

    print("\n" + "=" * 60)
    print(f"Clips written : {total_clips}")
    print(f"Clips skipped : {skipped_empty} (empty)")
    print(f"Segments      : {total_segs}  {spk_counts}")
    if total_bad:
        print(f"WARNING: Unparsed lines: {total_bad}")
    print(f"Output        : {out_root}")


if __name__ == "__main__":
    main()
