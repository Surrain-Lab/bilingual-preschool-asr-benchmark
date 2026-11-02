#!/usr/bin/env python3
"""
Build per-clip and per-recording Excel rollups from count_eng_esp_final
output CSVs. Fixes the Audio_File label for cases where extract_clip_id
truncated at a mid-stem `_<digit>` segment (e.g. R01_pre_1_NNN).
"""
import os
import re
import csv
import argparse
from pathlib import Path

import pandas as pd


def true_clip_id(stem: str) -> str:
    """Derive clip ID from the renamed transcript stem (already canonical)."""
    return stem


def fixed_combined_id(recording: str) -> str:
    return f"{recording}_COMBINED"


def gather(human_dir: Path, csv_dir: Path):
    per_clip_rows = []
    per_recording_rows = []
    for rec_dir in sorted(p for p in human_dir.iterdir() if p.is_dir()):
        recording = rec_dir.name
        clip_files = sorted(rec_dir.glob("*.txt"))
        if not clip_files:
            continue

        csv_path = csv_dir / recording / "combined_statistics.csv"
        if not csv_path.exists():
            print(f"  WARN: missing CSV for {recording}")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != len(clip_files) + 1:
            print(f"  WARN {recording}: csv has {len(rows)} rows, expected {len(clip_files)+1}")

        # First N rows are per-clip in same alphabetical order as Path.glob/sort.
        # Last row is the COMBINED row.
        for clip_file, row in zip(clip_files, rows[:len(clip_files)]):
            row = dict(row)
            row["Audio_File"] = clip_file.stem
            row["Recording"] = recording
            per_clip_rows.append(row)

        if rows:
            combined = dict(rows[-1])
            combined["Audio_File"] = fixed_combined_id(recording)
            combined["Recording"] = recording
            per_recording_rows.append(combined)

    return per_clip_rows, per_recording_rows


def to_excel(rows, path: Path):
    if not rows:
        print(f"  no rows for {path}")
        return
    df = pd.DataFrame(rows)
    cols = ["Recording"] + [c for c in df.columns if c != "Recording"]
    df = df[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    print(f"  wrote {path}  rows={len(df)}  cols={len(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--csv-dir",  required=True)
    ap.add_argument("--out-dir",  required=True)
    args = ap.parse_args()

    human_dir = Path(args.human_dir)
    csv_dir   = Path(args.csv_dir)
    out_dir   = Path(args.out_dir)

    per_clip, per_rec = gather(human_dir, csv_dir)
    to_excel(per_clip, out_dir / "human_per_clip.xlsx")
    to_excel(per_rec,  out_dir / "human_per_recording.xlsx")


if __name__ == "__main__":
    main()
