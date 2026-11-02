"""
word_count_comparison_excel.py
==============================
Reads combined_statistics.csv files produced by count_words*.py scripts
across all pipelines (Human Eval, VTC1, VTC2, WhisperX-Only) and builds
a formatted Excel workbook for language distribution comparison.

Expected folder structure per pipeline:
    pipeline_word_count_dir/
        {recording_name}/
            combined_statistics.csv   ← one row per clip + one COMBINED row

Run:
    python word_count_comparison_excel.py

Output:
    word_count_comparison.xlsx
"""
import os
from pathlib import Path
# Pipeline output root: $ASR_PROJECT/output_14clips, or pass --project.
_OUT = Path(os.environ.get("ASR_PROJECT", Path(__file__).resolve().parents[1] / "work")) / "output_14clips"


import csv
import os
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────────
# CONFIG — update paths to match your word count output folders
# ─────────────────────────────────────────────────────────────────
PIPELINES = [
    ("Human Eval",          str(_OUT / "human_eval/word_count")),
    ("Human-TS + Whisper",  str(_OUT / "human_timestamps/word_count")),
    ("VTC1 + Whisper",      str(_OUT / "vtc1/word_count")),
    ("VTC2 + Whisper",      str(_OUT / "vtc2/word_count")),
    ("Only Whisper",        str(_OUT / "whisperx_only/word_count")),
]

OUTPUT_FILE = "word_count_comparison_14clips.xlsx"

# ─────────────────────────────────────────────────────────────────
# SPEAKER → CSV column mapping
# Each tuple: (total, english, spanish, english_pct, spanish_pct)
# ─────────────────────────────────────────────────────────────────
SPEAKER_COLS = {
    "OVERALL":  (
        "Total_Words",
        "Total_English_Words",
        "Total_Spanish_Words",
        "English_Percentage",
        "Spanish_Percentage",
    ),
    "KCHI": (
        "KCHI_Total_Words",
        "KCHI_English_Words",
        "KCHI_Spanish_Words",
        "KCHI_English_Percentage",
        "KCHI_Spanish_Percentage",
    ),
    "OCH": (
        "OCH_Total_Words",
        "OCH_English_Words",
        "OCH_Spanish_Words",
        "OCH_English_Percentage",
        "OCH_Spanish_Percentage",
    ),
    "CHILDREN": (
        "Children_Combined_Total_Words",
        "Children_Combined_English_Words",
        "Children_Combined_Spanish_Words",
        "Children_Combined_English_Percentage",
        "Children_Combined_Spanish_Percentage",
    ),
    "MAL": (
        "MAL_Total_Words",
        "MAL_English_Words",
        "MAL_Spanish_Words",
        "MAL_English_Percentage",
        "MAL_Spanish_Percentage",
    ),
    "FEM": (
        "FEM_Total_Words",
        "FEM_English_Words",
        "FEM_Spanish_Words",
        "FEM_English_Percentage",
        "FEM_Spanish_Percentage",
    ),
    "ADULT": (
        "Adults_Combined_Total_Words",
        "Adults_Combined_English_Words",
        "Adults_Combined_Spanish_Words",
        "Adults_Combined_English_Percentage",
        "Adults_Combined_Spanish_Percentage",
    ),
}

SPEAKERS = ["KCHI", "OCH", "CHILDREN", "MAL", "FEM", "ADULT", "OVERALL"]

# Metrics shown on per-speaker sheets (label, col_index_in_SPEAKER_COLS, is_pct)
METRICS = [
    ("Total Words",   0, False),
    ("English Words", 1, False),
    ("Spanish Words", 2, False),
    ("English %",     3, True),
    ("Spanish %",     4, True),
]

# Metrics shown on Summary sheet
KEY_METRICS = [
    ("Total Words", 0, False),
    ("English %",   3, True),
    ("Spanish %",   4, True),
]

# ─────────────────────────────────────────────────────────────────
# STYLE CONSTANTS  (matches pipeline_comparision_eval_final.py)
# ─────────────────────────────────────────────────────────────────
HDR_DARK  = "1F3864"
HDR_MED   = "2E75B6"
HDR_LIGHT = "D6E4F0"
PIPE_COLS = ["D9EAD3", "FFF2CC", "FCE5CD", "EAD1DC"]
WHITE     = "FFFFFF"
ALT_ROW   = "F5F8FF"
COMB_ROW  = "FFE0B2"   # orange tint for COMBINED rows
SPK_COLS  = ["1F3864", "2E75B6", "2F9C6E", "7B3F7A", "C0392B", "6E4B2E", "555555"]


def _font(color="FFFFFF", sz=11, bold=True):
    return Font(name="Arial", bold=bold, color=color, size=sz)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _center(wrap=True):
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)

def _left():
    return Alignment(horizontal="left", vertical="center")

def _thin():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def _sep():
    return Border(
        left=Side(style="thin",   color="AAAAAA"),
        right=Side(style="medium", color="444444"),
        top=Side(style="thin",    color="AAAAAA"),
        bottom=Side(style="thin", color="AAAAAA"),
    )


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_pipeline_data(base_dir):
    """
    Load all combined_statistics.csv files under base_dir.
    Returns: dict of { recording_name: { clip_id: {col: value} } }
    """
    data = {}
    base = Path(base_dir)
    if not base.exists():
        print(f"  WARNING: Directory not found: {base_dir}")
        return data

    for rec_dir in sorted(base.iterdir()):
        if not rec_dir.is_dir():
            continue
        csv_file = rec_dir / "combined_statistics.csv"
        if not csv_file.exists():
            continue

        recording_name = rec_dir.name
        data[recording_name] = {}

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id = row.get("Audio_File", "").strip()
                if clip_id:
                    data[recording_name][clip_id] = row

    return data


def collect_all_data(pipelines):
    """Load data from all pipelines and find common recordings/clips."""
    pipeline_data = {}
    all_recordings = set()

    for name, path in pipelines:
        print(f"  Loading: {name}")
        pd = load_pipeline_data(path)
        pipeline_data[name] = pd
        all_recordings.update(pd.keys())

    # Build ordered clip list per recording (union across pipelines)
    recording_clips = {}
    for rec in sorted(all_recordings):
        clips_seen = set()
        combined_clips = []
        regular_clips = []

        for name, _ in pipelines:
            for clip_id in pipeline_data.get(name, {}).get(rec, {}).keys():
                if clip_id not in clips_seen:
                    clips_seen.add(clip_id)
                    if clip_id.endswith("_COMBINED"):
                        combined_clips.append(clip_id)
                    else:
                        regular_clips.append(clip_id)

        # Sort regular clips, put COMBINED at end
        recording_clips[rec] = sorted(regular_clips) + sorted(combined_clips)

    return pipeline_data, sorted(all_recordings), recording_clips


def get_value(pipeline_data, pipeline_name, recording, clip_id, speaker, metric_idx):
    """Safely get a value from loaded data."""
    col_name = SPEAKER_COLS.get(speaker, (None,) * 5)[metric_idx]
    if col_name is None:
        return None
    row = pipeline_data.get(pipeline_name, {}).get(recording, {}).get(clip_id, {})
    val = row.get(col_name, "")
    if val == "" or val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────
# EXCEL WRITING
# ─────────────────────────────────────────────────────────────────

def write_cell_value(cell, value, is_pct):
    """Write a numeric value with appropriate formatting."""
    if value is None:
        cell.value = "N/A"
        cell.font  = Font(name="Arial", color="999999", size=10)
        return
    if is_pct:
        cell.value         = value / 100.0
        cell.number_format = "0.00%"
    else:
        cell.value         = int(value) if value == int(value) else value
        cell.number_format = "#,##0"
    cell.font = Font(name="Arial", size=10)


def write_3row_header(ws, title_text, pipeline_names, metric_list):
    """Write the 3-row header: title / pipeline groups / metric sub-headers."""
    n_metrics   = len(metric_list)
    n_pipelines = len(pipeline_names)
    total_cols  = 2 + n_pipelines * n_metrics   # col A=Recording, col B=Clip

    # Row 1 — title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    c = ws.cell(1, 1, title_text)
    c.font = _font(sz=13); c.fill = _fill(HDR_DARK); c.alignment = _center()

    # Row 2 — Recording / Clip headers + pipeline group headers
    for col, label in [(1, "Recording"), (2, "Clip")]:
        c = ws.cell(2, col, label)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _center()

    for p_idx, p_name in enumerate(pipeline_names):
        cs = 3 + p_idx * n_metrics
        ce = cs + n_metrics - 1
        ws.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=ce)
        c = ws.cell(2, cs, p_name)
        c.font = _font(); c.alignment = _center()
        c.fill = _fill(HDR_DARK if p_idx % 2 == 0 else HDR_MED)

    # Row 3 — metric sub-headers
    for col, label in [(1, "Recording"), (2, "Clip")]:
        c = ws.cell(3, col, label)
        c.font = _font("000000", sz=9, bold=True)
        c.fill = _fill(HDR_LIGHT); c.alignment = _center(); c.border = _sep()

    for p_idx in range(n_pipelines):
        for m_idx, (m_label, _, _) in enumerate(metric_list):
            col = 3 + p_idx * n_metrics + m_idx
            c = ws.cell(3, col, m_label)
            c.font = Font(name="Arial", bold=True, size=9)
            c.fill = _fill(PIPE_COLS[p_idx % len(PIPE_COLS)])
            c.alignment = _center(); c.border = _thin()

    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 20
    ws.row_dimensions[3].height = 30


def build_per_speaker_sheet(ws, speaker, all_recordings, recording_clips,
                             pipeline_data, pipeline_names):
    """Build one per-speaker sheet."""
    n_metrics   = len(METRICS)
    n_pipelines = len(pipeline_names)
    total_cols  = 2 + n_pipelines * n_metrics

    write_3row_header(ws, f"Language Distribution — Speaker: {speaker}",
                      pipeline_names, METRICS)

    row = 4
    prev_rec = None
    for rec in all_recordings:
        clips = recording_clips.get(rec, [])
        for clip_id in clips:
            is_combined = clip_id.endswith("_COMBINED")
            bg = COMB_ROW if is_combined else (WHITE if row % 2 == 0 else ALT_ROW)

            # Recording column — only show on first clip of each recording
            rec_label = rec if rec != prev_rec else ""
            c = ws.cell(row, 1, rec_label)
            c.font = Font(name="Arial", bold=(rec_label != ""), size=10)
            c.fill = _fill("E8EEF7" if rec_label else "F5F5F5")
            c.alignment = _left(); c.border = _sep()
            prev_rec = rec

            # Clip column
            c = ws.cell(row, 2, clip_id)
            c.font = Font(name="Arial", bold=is_combined, size=10,
                          color="8B4513" if is_combined else "000000")
            c.fill = _fill(bg); c.alignment = _left(); c.border = _sep()

            # Data columns
            for p_idx, p_name in enumerate(pipeline_names):
                for m_idx, (_, col_idx, is_pct) in enumerate(METRICS):
                    col = 3 + p_idx * n_metrics + m_idx
                    c = ws.cell(row, col)
                    c.fill = _fill(bg); c.alignment = _center(wrap=False)
                    c.border = _thin()
                    val = get_value(pipeline_data, p_name, rec, clip_id, speaker, col_idx)
                    write_cell_value(c, val, is_pct)

            row += 1

        row += 1  # blank row between recordings

    # Column widths
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 32
    for p_idx in range(n_pipelines):
        for m_idx in range(n_metrics):
            ws.column_dimensions[
                get_column_letter(3 + p_idx * n_metrics + m_idx)
            ].width = 14


def build_summary_sheet(ws, all_recordings, recording_clips,
                         pipeline_data, pipeline_names):
    """Build summary sheet — key metrics across all speakers × pipelines."""
    n_key      = len(KEY_METRICS)
    n_speakers = len(SPEAKERS)
    n_pipelines = len(pipeline_names)
    total_cols = 2 + n_speakers * n_pipelines * n_key

    # Row 1 — title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    c = ws.cell(1, 1, "Language Distribution Summary — All Speakers × All Pipelines")
    c.font = _font(sz=13); c.fill = _fill(HDR_DARK); c.alignment = _center()

    # Row 2 — Recording / Clip + speaker group headers
    for col, label in [(1, "Recording"), (2, "Clip")]:
        c = ws.cell(2, col, label)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _center()

    for s_idx, spk in enumerate(SPEAKERS):
        cs = 3 + s_idx * len(pipeline_names) * n_key
        ce = cs + len(pipeline_names) * n_key - 1
        ws.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=ce)
        c = ws.cell(2, cs, spk)
        c.font = _font(); c.fill = _fill(SPK_COLS[s_idx % len(SPK_COLS)])
        c.alignment = _center()

    # Row 3 — pipeline sub-headers per speaker
    for col, label in [(1, "Recording"), (2, "Clip")]:
        c = ws.cell(3, col, label)
        c.font = _font("000000", sz=9, bold=True)
        c.fill = _fill(HDR_LIGHT); c.alignment = _center(); c.border = _sep()

    for s_idx in range(len(SPEAKERS)):
        for p_idx, p_name in enumerate(pipeline_names):
            cs = 3 + s_idx * len(pipeline_names) * n_key + p_idx * n_key
            ce = cs + n_key - 1
            ws.merge_cells(start_row=3, start_column=cs, end_row=3, end_column=ce)
            c = ws.cell(3, cs, p_name)
            c.font = Font(name="Arial", bold=True, size=8)
            c.fill = _fill(PIPE_COLS[p_idx % len(PIPE_COLS)])
            c.alignment = _center(); c.border = _thin()

    # Row 4 — metric sub-headers
    for col, label in [(1, "Recording"), (2, "Clip")]:
        c = ws.cell(4, col, label)
        c.font = _font("000000", sz=9, bold=True)
        c.fill = _fill(HDR_LIGHT); c.alignment = _center(); c.border = _sep()

    for s_idx in range(len(SPEAKERS)):
        for p_idx in range(len(pipeline_names)):
            for m_idx, (m_label, _, _) in enumerate(KEY_METRICS):
                col = 3 + s_idx * len(pipeline_names) * n_key + p_idx * n_key + m_idx
                c = ws.cell(4, col, m_label)
                c.font = Font(name="Arial", bold=True, size=8)
                c.fill = _fill(PIPE_COLS[p_idx % len(PIPE_COLS)])
                c.alignment = _center(); c.border = _thin()

    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 20
    ws.row_dimensions[3].height = 20
    ws.row_dimensions[4].height = 30

    # Data rows
    row = 5
    prev_rec = None
    for rec in all_recordings:
        clips = recording_clips.get(rec, [])
        for clip_id in clips:
            is_combined = clip_id.endswith("_COMBINED")
            bg = COMB_ROW if is_combined else (WHITE if row % 2 == 0 else ALT_ROW)

            rec_label = rec if rec != prev_rec else ""
            c = ws.cell(row, 1, rec_label)
            c.font = Font(name="Arial", bold=(rec_label != ""), size=10)
            c.fill = _fill("E8EEF7" if rec_label else "F5F5F5")
            c.alignment = _left(); c.border = _sep()
            prev_rec = rec

            c = ws.cell(row, 2, clip_id)
            c.font = Font(name="Arial", bold=is_combined, size=10,
                          color="8B4513" if is_combined else "000000")
            c.fill = _fill(bg); c.alignment = _left(); c.border = _sep()

            for s_idx, spk in enumerate(SPEAKERS):
                for p_idx, p_name in enumerate(pipeline_names):
                    for m_idx, (_, col_idx, is_pct) in enumerate(KEY_METRICS):
                        col = 3 + s_idx * len(pipeline_names) * n_key + p_idx * n_key + m_idx
                        c = ws.cell(row, col)
                        c.fill = _fill(bg); c.alignment = _center(wrap=False)
                        c.border = _thin()
                        val = get_value(pipeline_data, p_name, rec, clip_id, spk, col_idx)
                        write_cell_value(c, val, is_pct)

            row += 1
        row += 1

    # Column widths
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 32
    for col_idx in range(3, total_cols + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 12


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    pipeline_names = [p[0] for p in PIPELINES]

    print("Loading word count data...")
    print("=" * 60)
    pipeline_data, all_recordings, recording_clips = collect_all_data(PIPELINES)

    if not all_recordings:
        print("No data found. Check PIPELINES paths in CONFIG.")
        return

    print(f"\nFound {len(all_recordings)} recording(s)")
    total_clips = sum(len(v) for v in recording_clips.values())
    print(f"Total clips (including COMBINED rows): {total_clips}")
    print("\nBuilding Excel workbook...")

    wb = Workbook()

    # Summary sheet first (index=0)
    ws_summary = wb.active
    ws_summary.title = "Summary"
    build_summary_sheet(ws_summary, all_recordings, recording_clips,
                        pipeline_data, pipeline_names)
    print("   Summary sheet")

    # Per-speaker sheets
    for speaker in SPEAKERS:
        ws = wb.create_sheet(title=speaker)
        build_per_speaker_sheet(ws, speaker, all_recordings, recording_clips,
                                pipeline_data, pipeline_names)
        print(f"   {speaker} sheet")

    wb.save(OUTPUT_FILE)
    print(f"\n Excel saved: {OUTPUT_FILE}")
    print(f"   Sheets: Summary + {', '.join(SPEAKERS)}")


if __name__ == "__main__":
    main()
