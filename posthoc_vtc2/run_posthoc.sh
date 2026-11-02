#!/usr/bin/env bash
# WhisperX-VAD + VTC2 (post-hoc): relabel the existing WhisperX transcript with VTC2
# speaker regions and score it with the same evaluation as the other pipelines.
# No transcription and no GPU needed.
#
#   bash posthoc_vtc2/run_posthoc.sh <stage1_output_dir> <whisperx_merged_dir>
#
# <stage1_output_dir>   the stage-1 output folder (contains vtc2/ and human_eval/)
# <whisperx_merged_dir> the WhisperX-only transcripts with chunks merged
#                       (<recording>/<clip>/<clip>_all_merged_chunks.txt)
#
# Writes <stage1_output_dir>/whisperx_vtc2_posthoc/
#   {tagged_transcripts,eval_view,eval_report,langdetect,word_count}
# Activate the WhisperX environment first, or set PYTHON.
set -euo pipefail
OUT="$(cd "${1:?usage: $0 <stage1_output_dir> <whisperx_merged_dir>}" && pwd)"
MERGED="$(cd "${2:?usage: $0 <stage1_output_dir> <whisperx_merged_dir>}" && pwd)"
PY="${PYTHON:-python3}"
STAGE1="$(cd "$(dirname "$0")/../stage1_pipeline" && pwd)"
O="$OUT/whisperx_vtc2_posthoc"
SPK="$O/speakerdir_vtc2"
rm -rf "$O"; mkdir -p "$SPK"

# 1. VTC2 RTTMs -> the tagger's *_all_speakers.txt format. Recording folders are taken
#    from the WhisperX tree, because VTC2 names three recordings _pre_1 where WhisperX
#    uses _pre.
"$PY" - "$MERGED" "$OUT/vtc2/rttm_raw/rttm" "$SPK" <<'PYX'
import glob, os, sys
merged, rttm_dir, out = sys.argv[1:4]
rec_of = {os.path.basename(p.rstrip("/")): os.path.basename(os.path.dirname(p.rstrip("/")))
          for p in glob.glob(os.path.join(merged, "*", "*", ""))}
def fmt(t):
    h = int(t // 3600); m = int((t % 3600) // 60)
    return f"{h}:{m:02d}:{t - h*3600 - m*60:09.6f}"
for p in glob.glob(os.path.join(rttm_dir, "*.rttm")):
    clip = os.path.basename(p)[:-5]
    if clip not in rec_of:
        continue
    segs = sorted((float(f[3]), float(f[3]) + float(f[4]), f[7])
                  for f in (l.split() for l in open(p)) if len(f) >= 8 and f[0] == "SPEAKER")
    d = os.path.join(out, rec_of[clip], clip); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{clip}_all_speakers.txt"), "w") as o:
        for a, b, s in segs:
            o.write(f"[{fmt(a)} - {fmt(b)}] Speaker: {s}\n")
PYX

# 2. Same tagger, evaluator and word counter as the VTC1 post-hoc pipeline.
cd "$STAGE1"
"$PY" Only_Whisperx/transcript_based_speaker_tagger_step3.py --speaker-dir "$SPK" \
   --source-dir "$MERGED" --output-dir "$O/tagged_transcripts"
eval "$(sed -n '/^build_eval_view()/,/^}/p' run_analysis_only.sh)"
build_eval_view "$O/tagged_transcripts" "$O/eval_view"
"$PY" Only_Whisperx/evaluation_step4.py --ground-truth-dir "$OUT/human_eval/five_mins_benchmark" \
   --whisperx-dir "$O/eval_view" --output-dir "$O/eval_report" --language Spanish
"$PY" Only_Whisperx/combine_calculation_overall_final_eval.py --folder "$O/eval_report"
"$PY" Only_Whisperx/langdetect_fallback/langdetect_tags.py -i "$O/tagged_transcripts" -o "$O/langdetect"
"$PY" Only_Whisperx/langdetect_fallback/count_words.py -i "$O/langdetect" -o "$O/word_count"
