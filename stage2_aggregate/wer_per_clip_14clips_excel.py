import re
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ============================================================
# CONFIG — paths relative to project root (outputs/ from run_pipeline.py)
# ============================================================
_PROJ = Path(__file__).parent
_OUT  = _PROJ / "output_14clips"

PIPELINES = [
    ("Human-TS + Whisper",     str(_OUT / "human_timestamps"    / "eval_report")),
    ("VTC1 + Whisper",         str(_OUT / "vtc1"                / "eval_report")),
    ("VTC2 + Whisper",         str(_OUT / "vtc2"                / "eval_report")),
    ("VTC2-NoLang + Whisper",  str(_OUT / "vtc2_nolang_barrier" / "eval_report")),
    ("Only Whisper",           str(_OUT / "whisperx_only"       / "eval_report")),
]

OUTPUT_FILE = str(_PROJ / "wer_per_clip_14clips.xlsx")
# ============================================================

SPEAKER_MAP = {
    "fem": "FEM", "mal": "MAL", "kchi": "KCHI", "och": "OCH",
    "adult": "ADULT", "children": "CHILDREN",
    "overall": "OVERALL", "combined": "COMBINED",
}

METRICS = [
    ("WER",             "WER",             True),
    ("CER",             "CER",             True),
    ("Accuracy",        "ACC_PCT",         False),
    ("Insertions %",    "INS_PCT",         False),
    ("Deletions %",     "DEL_PCT",         False),
    ("Substitutions %", "SUB_PCT",         False),
    ("MER",             "MER",             True),
    ("WIL",             "WIL",             True),
    ("WIP",             "WIP",             True),
    ("Ref Words",       "ref_words_count", False),
    ("Hits",            "hits",            False),
]

KEY = [
    ("WER",             "WER",     True),
    ("Accuracy",        "ACC_PCT", False),
    ("Insertions %",    "INS_PCT", False),
    ("Deletions %",     "DEL_PCT", False),
    ("Substitutions %", "SUB_PCT", False),
    ("CER",             "CER",     True),
]

SPEAKER_ORDER = ["MAL", "FEM", "KCHI", "OCH", "ADULT", "CHILDREN", "OVERALL", "COMBINED"]

# ─── Colours ──────────────────────────────────────────────────
HDR_DARK  = "1F3864"
HDR_MED   = "2E75B6"
HDR_LIGHT = "D6E4F0"
PIPE_COLS = ["D9EAD3", "FFF2CC", "FCE5CD", "EAD1DC"]
WHITE     = "FFFFFF"
ALT_ROW   = "F5F8FF"
SPK_COLS  = ["1F3864", "2E75B6", "2F9C6E", "7B3F7A", "C0392B", "6E4B2E", "555555"]


# ─── Parsing ──────────────────────────────────────────────────
def parse_evaluation_file(filepath):
    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Could not read {filepath}: {e}")
        return None

    data = {}

    float_patterns = {
        "WER": r"Word Error Rate \(WER\):\s+([\d.]+)",
        "MER": r"Match Error Rate \(MER\):\s+([\d.]+)",
        "WIL": r"Word Information Lost \(WIL\):\s+([\d.]+)",
        "WIP": r"Word Information Preserved \(WIP\):\s+([\d.]+)",
        "CER": r"Character Error Rate \(CER\):\s+([\d.]+)",
    }
    int_patterns = {
        "ref_words_count": r"Total Reference Words:\s+(\d+)",
        "hits":            r"Hits \(Correct\):\s+(\d+)",
        "substitutions":   r"Substitutions.*?Total:\s+(\d+)",
        "deletions":       r"Deletions.*?Total:\s+(\d+)",
        "insertions":      r"Insertions.*?Total:\s+(\d+)",
    }

    for key, pat in float_patterns.items():
        m = re.search(pat, text)
        if m:
            data[key] = float(m.group(1))

    for key, pat in int_patterns.items():
        m = re.search(pat, text)
        if m:
            data[key] = int(m.group(1))

    if not data:
        return None

    ref = data.get("ref_words_count", 0)
    wer = data.get("WER", 0)
    data["ACC_PCT"] = (1 - wer) * 100
    data["INS_PCT"] = (data.get("insertions",    0) / ref * 100) if ref > 0 else 0.0
    data["DEL_PCT"] = (data.get("deletions",     0) / ref * 100) if ref > 0 else 0.0
    data["SUB_PCT"] = (data.get("substitutions", 0) / ref * 100) if ref > 0 else 0.0

    return data


# ─── Data collection ──────────────────────────────────────────
def collect_all_data(pipelines):
    """
    Walks each pipeline base directory recursively.
    Clip name = immediate parent folder of each *_evaluation.txt file.

    Returns
    -------
    all_recordings : sorted list of recording names
    clip_order     : sorted list of (recording, clip) tuples  — ONE ROW EACH
    all_speakers   : ordered list of speaker labels
    results        : results[recording][clip][pipeline][speaker] = parsed_dict
    """
    all_recordings = set()
    all_clips      = set()   # (recording, clip) tuples
    all_speakers   = set()
    results        = {}

    for pipeline_name, base_path in pipelines:
        base = Path(base_path)
        if not base.exists():
            print(f"  WARNING: Directory not found: {base}")
            continue

        for recording_dir in sorted(base.iterdir()):
            if not recording_dir.is_dir():
                continue

            recording_name = recording_dir.name

            # Recursively find every *_evaluation.txt at any depth
            speaker_files = sorted(recording_dir.rglob("*_evaluation.txt"))

            if not speaker_files:
                print(f"  WARNING: No evaluation files under {recording_dir}")
                continue

            all_recordings.add(recording_name)

            for spk_file in speaker_files:
                # clip = immediate parent folder of the txt file
                clip_name = spk_file.parent.name

                parts   = spk_file.stem.split("_")
                spk_raw = parts[-2].lower()
                speaker = SPEAKER_MAP.get(spk_raw, spk_raw.upper())

                parsed = parse_evaluation_file(str(spk_file))
                if parsed:
                    all_speakers.add(speaker)
                    all_clips.add((recording_name, clip_name))
                    (results
                     .setdefault(recording_name, {})
                     .setdefault(clip_name, {})
                     .setdefault(pipeline_name, {})[speaker]) = parsed
                    print(f"   {pipeline_name} | {recording_name} | {clip_name} | {speaker}")

    clip_order = sorted(all_clips, key=lambda x: (x[0], x[1]))

    ordered_speakers = [s for s in SPEAKER_ORDER if s in all_speakers]
    for s in sorted(all_speakers):
        if s not in ordered_speakers:
            ordered_speakers.append(s)

    return sorted(all_recordings), clip_order, ordered_speakers, results


# ─── Style helpers ────────────────────────────────────────────
def _font(color="FFFFFF", sz=11, bold=True):
    return Font(name="Arial", bold=bold, color=color, size=sz)

def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def _ctr(wrap=True):
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)

def _thin():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def _sep():
    return Border(
        left=Side(style="thin",    color="AAAAAA"),
        right=Side(style="medium", color="444444"),
        top=Side(style="thin",     color="AAAAAA"),
        bottom=Side(style="thin",  color="AAAAAA"),
    )

def _rec_border():
    """Thicker top border to visually separate recordings."""
    return Border(
        left=Side(style="thin",   color="AAAAAA"),
        right=Side(style="medium",color="444444"),
        top=Side(style="medium",  color="2E75B6"),
        bottom=Side(style="thin", color="AAAAAA"),
    )

def write_value(cell, value, is_rate, m_key):
    if value is None:
        cell.value = "N/A"
        cell.font  = Font(name="Arial", color="999999", size=10)
        return
    if is_rate:
        cell.value         = value
        cell.number_format = "0.00%"
    elif m_key in ("INS_PCT", "DEL_PCT", "SUB_PCT", "ACC_PCT"):
        cell.value         = value / 100
        cell.number_format = "0.00%"
    else:
        cell.value         = value
        cell.number_format = "#,##0"
    cell.font = Font(name="Arial", size=10)


# ─── Header writer ────────────────────────────────────────────
def write_headers(ws, title_text, pipeline_names, metric_list, label_cols):
    """
    label_cols: list of column labels before metric data, e.g. ["Recording","Clip"]
    """
    n_lbl      = len(label_cols)
    n_metrics  = len(metric_list)
    total_cols = n_lbl + len(pipeline_names) * n_metrics

    # Row 1 – title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    c = ws.cell(1, 1, title_text)
    c.font = _font(sz=13); c.fill = _fill(HDR_DARK); c.alignment = _ctr()

    # Row 2 – label + pipeline group headers
    for col, label in enumerate(label_cols, start=1):
        c = ws.cell(2, col, label)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _ctr()

    for p_idx, p_name in enumerate(pipeline_names):
        cs = n_lbl + 1 + p_idx * n_metrics
        ws.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=cs + n_metrics - 1)
        c = ws.cell(2, cs, p_name)
        c.font = _font(); c.alignment = _ctr()
        c.fill = _fill(HDR_DARK if p_idx % 2 == 0 else HDR_MED)

    # Row 3 – sub-headers
    for col, label in enumerate(label_cols, start=1):
        c = ws.cell(3, col, label)
        c.font = _font("000000", bold=True, sz=9)
        c.fill = _fill(HDR_LIGHT); c.alignment = _ctr(); c.border = _sep()

    for p_idx in range(len(pipeline_names)):
        for m_idx, (m_label, _, _) in enumerate(metric_list):
            col = n_lbl + 1 + p_idx * n_metrics + m_idx
            c = ws.cell(3, col, m_label)
            c.font = Font(name="Arial", bold=True, size=9)
            c.fill = _fill(PIPE_COLS[p_idx % len(PIPE_COLS)])
            c.alignment = _ctr(); c.border = _thin()

    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 20
    ws.row_dimensions[3].height = 32


# ─── Excel builder ────────────────────────────────────────────
def build_excel(all_recordings, clip_order, all_speakers, results, pipelines, output_file):
    wb = Workbook()
    pipeline_names = [p[0] for p in pipelines]

    # ══════════════════════════════════════════════════════════
    # Per-speaker sheets  — columns: Recording | Clip | <pipelines…>
    # Each row = one clip
    # ══════════════════════════════════════════════════════════
    first = True
    for speaker in all_speakers:
        ws = wb.active if first else wb.create_sheet(title=speaker)
        if first:
            ws.title = speaker
            first = False

        write_headers(
            ws,
            f"Transcription Pipeline Comparison — Speaker: {speaker}",
            pipeline_names, METRICS,
            label_cols=["Recording", "Clip"]
        )

        n_metrics = len(METRICS)
        prev_rec  = None

        for row_idx, (rec, clip) in enumerate(clip_order):
            row     = 4 + row_idx
            new_rec = (rec != prev_rec)
            bg      = WHITE if row_idx % 2 == 0 else ALT_ROW
            bdr     = _rec_border() if new_rec else _thin()

            # Col A – Recording name (only on first clip of that recording)
            c = ws.cell(row, 1, rec if new_rec else "")
            c.font = Font(name="Arial", bold=new_rec, size=10)
            c.fill = _fill("E8EEF7")
            c.alignment = Alignment(horizontal="left", vertical="center")
            c.border = bdr

            # Col B – Clip name
            c = ws.cell(row, 2, clip)
            c.font = Font(name="Arial", size=10)
            c.fill = _fill("F0F4FF")
            c.alignment = Alignment(horizontal="left", vertical="center")
            c.border = _thin()

            # Metric columns for each pipeline
            for p_idx, p_name in enumerate(pipeline_names):
                spk_data = (results
                            .get(rec,  {})
                            .get(clip, {})
                            .get(p_name, {})
                            .get(speaker, {}))
                for m_idx, (_, m_key, is_rate) in enumerate(METRICS):
                    col = 3 + p_idx * n_metrics + m_idx
                    c   = ws.cell(row, col)
                    c.fill = _fill(bg)
                    c.alignment = _ctr(wrap=False)
                    c.border = _thin()
                    write_value(c, spk_data.get(m_key), is_rate, m_key)

            prev_rec = rec

        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 32
        for p_idx in range(len(pipeline_names)):
            for m_idx in range(n_metrics):
                ws.column_dimensions[get_column_letter(3 + p_idx * n_metrics + m_idx)].width = 13

    # ══════════════════════════════════════════════════════════
    # Summary sheet — columns: Recording | Clip | Pipeline | <speakers…>
    # Each block = one clip × all pipelines
    # ══════════════════════════════════════════════════════════
    ws_s = wb.create_sheet(title="Summary", index=0)
    n_key      = len(KEY)
    total_cols = 3 + len(all_speakers) * n_key

    ws_s.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    c = ws_s.cell(1, 1, "Pipeline Comparison Summary — All Speakers (Per Clip)")
    c.font = _font(sz=13); c.fill = _fill(HDR_DARK); c.alignment = _ctr()

    for col, label in [(1, "Recording"), (2, "Clip"), (3, "Pipeline")]:
        c = ws_s.cell(2, col, label)
        c.font = _font(); c.fill = _fill(HDR_DARK); c.alignment = _ctr()

    for s_idx, spk in enumerate(all_speakers):
        cs = 4 + s_idx * n_key
        ws_s.merge_cells(start_row=2, start_column=cs, end_row=2, end_column=cs + n_key - 1)
        c = ws_s.cell(2, cs, spk)
        c.font = _font(); c.fill = _fill(SPK_COLS[s_idx % len(SPK_COLS)]); c.alignment = _ctr()

    for col, label in [(1, "Recording"), (2, "Clip"), (3, "Pipeline")]:
        c = ws_s.cell(3, col, label)
        c.font = Font(name="Arial", bold=True, size=9)
        c.fill = _fill(HDR_LIGHT); c.alignment = _ctr(); c.border = _sep()

    for s_idx in range(len(all_speakers)):
        for m_idx, (m_label, _, _) in enumerate(KEY):
            col = 4 + s_idx * n_key + m_idx
            c = ws_s.cell(3, col, m_label)
            c.font = Font(name="Arial", bold=True, size=9)
            c.fill = _fill(PIPE_COLS[s_idx % len(PIPE_COLS)])
            c.alignment = _ctr(); c.border = _thin()

    ws_s.row_dimensions[1].height = 22
    ws_s.row_dimensions[2].height = 20
    ws_s.row_dimensions[3].height = 32

    row      = 4
    prev_rec = None

    for rec, clip in clip_order:
        new_rec = (rec != prev_rec)

        for p_idx, p_name in enumerate(pipeline_names):
            bg  = "F0F4FF" if p_idx % 2 == 0 else WHITE
            bdr = _rec_border() if (new_rec and p_idx == 0) else _thin()

            # Col A – Recording
            ca = ws_s.cell(row, 1, rec if (new_rec and p_idx == 0) else "")
            ca.font = Font(name="Arial", bold=(new_rec and p_idx == 0), size=10)
            ca.fill = _fill("E8EEF7" if new_rec else "F5F5F5")
            ca.alignment = Alignment(horizontal="left", vertical="center")
            ca.border = bdr

            # Col B – Clip
            cb = ws_s.cell(row, 2, clip if p_idx == 0 else "")
            cb.font = Font(name="Arial", bold=(p_idx == 0), size=10)
            cb.fill = _fill("F0F4FF")
            cb.alignment = Alignment(horizontal="left", vertical="center")
            cb.border = _thin()

            # Col C – Pipeline
            cc = ws_s.cell(row, 3, p_name)
            cc.font = Font(name="Arial", size=10)
            cc.fill = _fill(bg)
            cc.alignment = Alignment(horizontal="left", vertical="center")
            cc.border = _thin()

            # Metric columns per speaker
            for s_idx, spk in enumerate(all_speakers):
                spk_data = (results
                            .get(rec,  {})
                            .get(clip, {})
                            .get(p_name, {})
                            .get(spk, {}))
                for m_idx, (_, m_key, is_rate) in enumerate(KEY):
                    col_num = 4 + s_idx * n_key + m_idx
                    c = ws_s.cell(row, col_num)
                    c.fill = _fill(bg); c.alignment = _ctr(wrap=False); c.border = _thin()
                    write_value(c, spk_data.get(m_key), is_rate, m_key)

            row += 1

        prev_rec = rec
        row += 1  # blank row between clips

    ws_s.column_dimensions["A"].width = 28
    ws_s.column_dimensions["B"].width = 32
    ws_s.column_dimensions["C"].width = 20
    for s_idx in range(len(all_speakers)):
        for m_idx in range(n_key):
            ws_s.column_dimensions[get_column_letter(4 + s_idx * n_key + m_idx)].width = 13

    wb.save(output_file)
    print(f"\n Excel saved → {output_file}")


# ─── Entry point ──────────────────────────────────────────────
def main():
    print("Collecting evaluation data...")
    print("=" * 60)
    all_recordings, clip_order, all_speakers, results = collect_all_data(PIPELINES)

    if not results:
        print("\nERROR: No data found. Check your PIPELINES paths in the CONFIG section.")
        return

    print(f"\nFound {len(all_recordings)} recording(s), {len(clip_order)} clip(s) total")
    print(f"Speakers detected: {all_speakers}")
    print("\nBuilding Excel workbook...")
    build_excel(all_recordings, clip_order, all_speakers, results, PIPELINES, OUTPUT_FILE)


if __name__ == "__main__":
    main()
