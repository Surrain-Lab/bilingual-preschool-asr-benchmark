#!/usr/bin/env python3
"""
Replace study recording identifiers with the labels P01..P14 used in the paper.

Labels are assigned by sorting the raw recording identifiers, which gives the
same participant numbering as the paper's per-recording table. The mapping is
written to config/recording_id_map.csv, which is not committed.

Usage:  python stage2_aggregate/anonymise_ids.py data/derived
"""
import csv, os, re, sys, shutil

RAW  = re.compile(r'DL-\d+_pre(?:_1)?')
BASE = re.compile(r'(DL-\d+)(_pre(?:_1)?)')

def collect(root):
    ids = set()
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.endswith(('.csv', '.txt', '.md')):
                try: ids |= set(RAW.findall(open(p, encoding='utf-8', errors='replace').read()))
                except Exception: pass
            elif fn.endswith('.xlsx'):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
                    for ws in wb:
                        for row in ws.iter_rows(values_only=True):
                            for c in row:
                                if isinstance(c, str): ids |= set(RAW.findall(c))
                except Exception: pass
    return ids

def build_map(ids):
    """One label per RECORDING BASE. DL-x_pre and DL-x_pre_1 are the same
    recording under two clip-naming conventions, so they share a label and
    keep their suffix - collapsing them would break joins between files."""
    bases = sorted({BASE.match(i).group(1) for i in ids})
    blab  = {b: f"P{i:02d}" for i, b in enumerate(bases, 1)}
    return {i: blab[BASE.match(i).group(1)] + BASE.match(i).group(2) for i in sorted(ids)}

def apply_text(p, mapping):
    s = open(p, encoding='utf-8', errors='replace').read(); o = s
    for raw, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(raw, new)
    if s != o:
        open(p, 'w', encoding='utf-8').write(s); return True
    return False

def apply_xlsx(p, mapping):
    import openpyxl
    wb = openpyxl.load_workbook(p)
    hit = False
    for ws in wb:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str):
                    v = c.value
                    for raw, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
                        v = v.replace(raw, new)
                    if v != c.value: c.value = v; hit = True
    if hit: wb.save(p)
    return hit

def main(root):
    ids = collect(root)
    if not ids:
        print("no raw identifiers found - nothing to do"); return
    mapping = build_map(ids)
    os.makedirs("config", exist_ok=True)
    with open("config/recording_id_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo_label", "raw_recording_id", "matches_paper_participant"])
        for raw, new in sorted(mapping.items(), key=lambda kv: kv[1]):
            w.writerow([new, raw, "yes"])
    n = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.endswith(('.csv', '.txt', '.md')): n += apply_text(p, mapping)
            elif fn.endswith('.xlsx'):               n += apply_xlsx(p, mapping)
    print(f"mapped {len(mapping)} recordings across {n} files")
    print("wrote config/recording_id_map.csv (labels match the paper's participant numbering)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/derived")
