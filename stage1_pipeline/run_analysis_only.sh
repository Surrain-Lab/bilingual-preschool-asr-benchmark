#!/usr/bin/env bash

# --- portability -------------------------------------------------------------
# Set ASR_PROJECT to your pipeline working directory (where outputs go), and
# ASR_VTC1_MODEL / ASR_VTC2_MODEL to your model checkouts. See config/config.example.yaml.
: "${ASR_PROJECT:?set ASR_PROJECT to your working directory (see README)}"

# ============================================================================
# run_analysis_only.sh — Re-run all analysis on EXISTING transcripts
#
# Usage:
#   bash run_analysis_only.sh [output_dir]
#
# Default output_dir: $ASR_PROJECT/output_14clips
#
# Assumes transcription is already done. Runs:
#   1. Backup existing analysis (eval_report, langdetect, word_count)
#      → {output}/previous_{TIMESTAMP}/
#   2. Language detection + word count  (vtc1, vtc2, whisperx, human_timestamps)
#   3. VTC1 WER without [SPEECH] umbrella
#   4. VTC2 WER
#   5. WhisperX WER
#   6. Human Timestamps WER
#   7. Excel reports (wer_per_clip, wer_per_recording, word_count_comparison)
#
# Skipped: all transcription, vtc2_nolang_barrier (already done)
# ============================================================================
set -uo pipefail

# ─────────────────────────────── CONFIG ──────────────────────────────────────
PROJECT="${ASR_PROJECT}"
OUTPUT="${1:-$PROJECT/output_14clips}"
AUDIO="$PROJECT/audio"
ANNOT="$PROJECT/annotated-text"
CONDA_BASE="${CONDA_BASE:-/opt/conda}"
WHISPERX_ENV="whisperx"
SHIM_DIR="$PROJECT/bin_shims"

# Ground truth dir — regenerated fresh by step 0; also used for all WER steps
GT="$OUTPUT/human_eval/five_mins_benchmark"
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT/logs"
MASTER_LOG="$LOG_DIR/analysis_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/analysis_${RUN_ID}_status.txt"
PID_FILE="$LOG_DIR/analysis_${RUN_ID}.pid"
PREV_DIR="$OUTPUT/previous_${RUN_ID}"

# ──────────────────────────── self-detach ────────────────────────────────────
if [[ "${_DETACHED:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  echo "Launching analysis-only run in background..."
  _DETACHED=1 RUN_ID="$RUN_ID" setsid bash "$0" "$OUTPUT" \
    >>"$MASTER_LOG" 2>&1 < /dev/null &
  sleep 1
  echo "  Output   : $OUTPUT"
  echo "  Backup   : $PREV_DIR"
  echo "  Log      : $MASTER_LOG"
  echo "  Status   : $STATUS_FILE"
  echo ""
  echo "  Watch    :  tail -f \"$MASTER_LOG\""
  echo "  Status   :  cat \"$STATUS_FILE\""
  echo "  Stop     :  kill -- -\$(cat \"$PID_FILE\")"
  exit 0
fi
echo "$$" > "$PID_FILE"
mkdir -p "$LOG_DIR"

# ─────────────────────── environment ─────────────────────────────────────────
export PATH="$SHIM_DIR:$PATH"
_NV="${WHISPERX_ENV_PREFIX:-$CONDA_BASE/envs/$WHISPERX_ENV}/lib/python3.10/site-packages/nvidia"
if compgen -G "$_NV/*/lib" > /dev/null 2>&1; then
  _LDP=$(printf '%s:' "$_NV"/*/lib); export LD_LIBRARY_PATH="${_LDP%:}"
else
  unset LD_LIBRARY_PATH 2>/dev/null || true
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ─────────────────────── helpers ─────────────────────────────────────────────
log() {
  local m="[$(date '+%F %T')] $*"
  echo "$m"
  echo "$m" >> "$STATUS_FILE"
}

PY() {
  ( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output python "$@" )
}

RP() {
  ( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
      python "$PROJECT/run_pipeline.py" \
      --audio "$AUDIO" --annotated "$ANNOT" --output "$OUTPUT" --device cpu "$@" )
}

# Build a GT-aligned eval view: symlinks recording/clip → actual clip dir
# Needed so evaluators can match hypothesis clips to ground-truth by recording name
build_eval_view() {
  local hyp_dir="$1" view_dir="$2"
  rm -rf "$view_dir"; mkdir -p "$view_dir"
  local n=0 clipdir clip rec
  for clipdir in "$hyp_dir"/*/*/; do
    [[ -d "$clipdir" ]] || continue
    clip=$(basename "$clipdir"); rec="${clip%_*}"
    mkdir -p "$view_dir/$rec"
    ln -sfn "$(realpath "$clipdir")" "$view_dir/$rec/$clip"
    n=$((n+1))
  done
  echo "$n"
}

# ─────────────────────── preflight ───────────────────────────────────────────
: > "$STATUS_FILE"
log "=== ANALYSIS-ONLY RUN $RUN_ID START ==="
log "    output=$OUTPUT  GT=$GT"

fail=0
[[ -d "$OUTPUT" ]]    || { log "OUTPUT dir not found: $OUTPUT"; fail=1; }
[[ -d "$AUDIO" ]]     || { log "Audio dir not found: $AUDIO"; fail=1; }
[[ -d "$ANNOT" ]]     || { log "Annotated text dir not found: $ANNOT"; fail=1; }
conda env list | grep -qw "$WHISPERX_ENV" || { log "MISSING conda env: $WHISPERX_ENV"; fail=1; }
[[ "$fail" -ne 0 ]] && { log "Preflight FAILED — aborting."; exit 1; }

# ─────── PRE-STEP: rename human transcripts to match WAV clip names ──────────
# human_transcripts_timestamps/ files have suffixes like -AM-VOR_human that
# need stripping so transcript_converter2.py can derive the correct clip ID.
# e.g. P07_pre_067-AM-VOR_human.txt  →  P07_pre_067.txt
log "PRE-STEP — Renaming human_transcripts_timestamps/ files to match WAV clip names"
PY "$PROJECT/check_rename_human_files.py" \
   --audio-dir "$AUDIO" \
   --human-dir "$ANNOT/human_transcripts_timestamps" \
   >>"$LOG_DIR/analysis_${RUN_ID}_rename.log" 2>&1 \
  && log "  rename done" \
  || log "  WARNING: rename had issues (check analysis_${RUN_ID}_rename.log)"

# ─── STEP 0: regenerate human eval reference (clean WER ground truth) ────────
# Reads renamed files from human_transcripts_timestamps/, strips CHAT markers,
# outputs per-speaker clean reference text → used as GT for all WER evaluations.
# Output: $OUTPUT/human_eval/five_mins_benchmark/
log "STEP 0 — Human eval reference generation (GT for WER)"
( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
    python "$PROJECT/Human_Evaluation/transcript_converter2.py" \
    -i "$ANNOT/human_transcripts_timestamps" \
    -o "$GT" ) \
  >>"$LOG_DIR/analysis_${RUN_ID}_step0.log" 2>&1 \
  && log "  STEP 0 done — GT at $GT" \
  || log "  WARNING: STEP 0 had issues (check analysis_${RUN_ID}_step0.log)"

[[ -d "$GT" ]] || { log "GT dir still missing after step 0: $GT — aborting."; exit 1; }

# ── STEP 0b: human transcript word count (English/Spanish from CHAT source) ──
# Counts words directly from the original human transcripts (curly braces = English,
# rest = Spanish). Runs independently of the pipeline transcription word counts.
log "STEP 0b — Human transcript word count"
( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
    python "$PROJECT/Human_Evaluation/count_eng_esp_final.py" \
    -i "$ANNOT/human_transcripts_timestamps" \
    -o "$OUTPUT/human_eval/word_count" ) \
  >>"$LOG_DIR/analysis_${RUN_ID}_human_wordcount.log" 2>&1 \
  && log "  human word count done" \
  || log "  WARNING: human word count had issues (check analysis_${RUN_ID}_human_wordcount.log)"

log "STEP 0b — Human word count Excel"
( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
    python "$PROJECT/Human_Evaluation/build_human_excel.py" \
    --human-dir "$ANNOT/human_transcripts_timestamps" \
    --csv-dir   "$OUTPUT/human_eval/word_count" \
    --out-dir   "$OUTPUT/human_eval" ) \
  >>"$LOG_DIR/analysis_${RUN_ID}_human_wordcount.log" 2>&1 \
  && log "  human word count Excel done" \
  || log "  WARNING: human word count Excel had issues"

# ─────────────────────── STEP 1: backup existing analysis ────────────────────
log "STEP 1 — Backing up existing analysis → $PREV_DIR"
mkdir -p "$PREV_DIR"

for pipe in vtc1 vtc2 whisperx_only human_timestamps vtc2_nolang_barrier; do
  src="$OUTPUT/$pipe"
  [[ -d "$src" ]] || continue
  dest="$PREV_DIR/$pipe"
  mkdir -p "$dest"
  for d in eval_report eval_report.bak_withspeech langdetect langdetect.bak_clipid word_count word_count.bak_clipid speaker_separated_nospeech; do
    [[ -d "$src/$d" ]] && cp -r "$src/$d" "$dest/$d" && log "  backed up $pipe/$d"
  done
done

# Backup Excel files from project root
mkdir -p "$PREV_DIR/excel"
for f in "$PROJECT"/*.xlsx; do
  [[ -f "$f" ]] && cp "$f" "$PREV_DIR/excel/" && log "  backed up $(basename "$f")"
done
log "  Backup complete → $PREV_DIR"

# ─────────────── STEP 2: clean old analysis (fresh start) ────────────────────
log "STEP 2 — Removing old analysis dirs (keeping transcripts)"
for pipe in vtc1 vtc2 whisperx_only human_timestamps; do
  for d in eval_report langdetect word_count speaker_separated_nospeech eval_view; do
    [[ -d "$OUTPUT/$pipe/$d" ]] && rm -rf "$OUTPUT/$pipe/$d" && log "  removed $pipe/$d"
  done
done

# ──────────── STEP 3: language detection + word count ────────────────────────
log "STEP 3 — Language detection + word count (vtc1, vtc2, whisperx)"
RP --pipelines vtc1 vtc2 whisperx --steps 4 \
  >>"$LOG_DIR/analysis_${RUN_ID}_step4.log" 2>&1 \
  && log "  langdetect+word_count (vtc1/vtc2/whisperx) done" \
  || log "  WARNING: langdetect+word_count had issues (check analysis_${RUN_ID}_step4.log)"

log "STEP 3 — Language detection + word count (human_timestamps)"
PY "$PROJECT/Human_Timestamps_Pipeline/Langdetect_fallback/langdetect_tags.py" \
   -i "$OUTPUT/human_timestamps/speaker_separated" \
   -o "$OUTPUT/human_timestamps/langdetect" \
   >>"$LOG_DIR/analysis_${RUN_ID}_step4_humants.log" 2>&1 \
  && log "  langdetect (human_timestamps) done" \
  || log "  WARNING: langdetect (human_timestamps) issue"

PY "$PROJECT/Human_Timestamps_Pipeline/Langdetect_fallback/count_words.py" \
   -i "$OUTPUT/human_timestamps/langdetect" \
   -o "$OUTPUT/human_timestamps/word_count" \
   >>"$LOG_DIR/analysis_${RUN_ID}_step4_humants.log" 2>&1 \
  && log "  word_count (human_timestamps) done" \
  || log "  WARNING: word_count (human_timestamps) issue"

# ────────── STEP 4: VTC1 WER (no [SPEECH] umbrella — avoids inflation) ───────
log "STEP 4 — VTC1 WER (without [SPEECH] umbrella)"

TAGGED_VTC1="$OUTPUT/vtc1/tagged_transcripts"
if [[ ! -d "$TAGGED_VTC1" ]]; then
  log "  WARNING: vtc1/tagged_transcripts not found — skipping VTC1 WER"
else
  # Generate nospeech splitter if not already present
  PY - >>"$LOG_DIR/analysis_${RUN_ID}_vtc1_wer.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path
P = Path(os.environ["ASR_PROJECT"])
out = P / "VTC1_Pipeline/labels_seperator_nospeech.py"
if out.exists():
    print(f"already exists: {out}")
else:
    src = (P / "VTC2_Pipeline/labels_seperator.py").read_text(encoding="utf-8")
    old = "            by_speaker['all'].append(parsed)"
    new = ("            if speaker != 'SPEECH':  # exclude [SPEECH] umbrella from combined\n"
           "                by_speaker['all'].append(parsed)")
    assert old in src, "anchor line not found in labels_seperator.py"
    out.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"created {out}")
PYEOF
  [[ $? -ne 0 ]] && log "  WARNING: nospeech script generation failed — skipping VTC1 WER"

  if [[ $? -eq 0 ]]; then
    log "  [VTC1 WER] re-splitting tagged_transcripts without [SPEECH]"
    PY "$PROJECT/VTC1_Pipeline/labels_seperator_nospeech.py" \
       -i "$TAGGED_VTC1" -o "$OUTPUT/vtc1/speaker_separated_nospeech" \
       >>"$LOG_DIR/analysis_${RUN_ID}_vtc1_wer.log" 2>&1 \
      && log "  [VTC1 WER] split done" || { log "  WARNING: [VTC1 WER] split failed"; }

    log "  [VTC1 WER] building eval view"
    n=$(build_eval_view "$OUTPUT/vtc1/speaker_separated_nospeech" "$OUTPUT/vtc1/eval_view")
    log "  [VTC1 WER] eval view: $n clips"

    PY "$PROJECT/VTC1_Pipeline/evaluation_unmerged_not_applicable.py" \
       --ground-truth-dir "$GT" \
       --whisperx-dir     "$OUTPUT/vtc1/eval_view" \
       --output-dir       "$OUTPUT/vtc1/eval_report" \
       --language         Spanish \
       >>"$LOG_DIR/analysis_${RUN_ID}_vtc1_wer.log" 2>&1 \
      && log "  [VTC1 WER] evaluation done" || log "  WARNING: [VTC1 WER] evaluation issue"

    PY "$PROJECT/VTC1_Pipeline/combine_calculation_unmerged_overall_final_eval.py" \
       --folder "$OUTPUT/vtc1/eval_report" \
       >>"$LOG_DIR/analysis_${RUN_ID}_vtc1_wer.log" 2>&1 \
      && log "  [VTC1 WER] aggregation done" || log "  WARNING: [VTC1 WER] aggregation issue"
  fi
fi

# ─────────────────────── STEP 5: VTC2 WER ────────────────────────────────────
log "STEP 5 — VTC2 WER"
n=$(build_eval_view "$OUTPUT/vtc2/speaker_separated" "$OUTPUT/vtc2/eval_view")
log "  [VTC2 WER] eval view: $n clips"

PY "$PROJECT/VTC2_Pipeline/evaluation.py" \
   --ground-truth-dir "$GT" \
   --whisperx-dir     "$OUTPUT/vtc2/eval_view" \
   --output-dir       "$OUTPUT/vtc2/eval_report" \
   --language         Spanish \
   >>"$LOG_DIR/analysis_${RUN_ID}_vtc2_wer.log" 2>&1 \
  && log "  [VTC2 WER] evaluation done" || log "  WARNING: [VTC2 WER] evaluation issue"

PY "$PROJECT/VTC2_Pipeline/combine_calculation_overall_final_eval.py" \
   --folder "$OUTPUT/vtc2/eval_report" \
   >>"$LOG_DIR/analysis_${RUN_ID}_vtc2_wer.log" 2>&1 \
  && log "  [VTC2 WER] aggregation done" || log "  WARNING: [VTC2 WER] aggregation issue"

# ─────────────────────── STEP 6: WhisperX WER ───────────────────────────────
log "STEP 6 — WhisperX WER"
n=$(build_eval_view "$OUTPUT/whisperx_only/tagged_transcripts" "$OUTPUT/whisperx_only/eval_view")
log "  [WhisperX WER] eval view: $n clips"

PY "$PROJECT/Only_Whisperx/evaluation_step4.py" \
   --ground-truth-dir "$GT" \
   --whisperx-dir     "$OUTPUT/whisperx_only/eval_view" \
   --output-dir       "$OUTPUT/whisperx_only/eval_report" \
   --language         Spanish \
   >>"$LOG_DIR/analysis_${RUN_ID}_whisperx_wer.log" 2>&1 \
  && log "  [WhisperX WER] evaluation done" || log "  WARNING: [WhisperX WER] evaluation issue"

PY "$PROJECT/Only_Whisperx/combine_calculation_overall_final_eval.py" \
   --folder "$OUTPUT/whisperx_only/eval_report" \
   >>"$LOG_DIR/analysis_${RUN_ID}_whisperx_wer.log" 2>&1 \
  && log "  [WhisperX WER] aggregation done" || log "  WARNING: [WhisperX WER] aggregation issue"

# ─────────────────────── STEP 7: Human Timestamps WER ───────────────────────
log "STEP 7 — Human Timestamps WER"
n=$(build_eval_view "$OUTPUT/human_timestamps/speaker_separated" "$OUTPUT/human_timestamps/eval_view")
log "  [HumanTS WER] eval view: $n clips"

PY "$PROJECT/VTC2_Pipeline/evaluation.py" \
   --ground-truth-dir "$GT" \
   --whisperx-dir     "$OUTPUT/human_timestamps/eval_view" \
   --output-dir       "$OUTPUT/human_timestamps/eval_report" \
   --language         Spanish \
   >>"$LOG_DIR/analysis_${RUN_ID}_humants_wer.log" 2>&1 \
  && log "  [HumanTS WER] evaluation done" || log "  WARNING: [HumanTS WER] evaluation issue"

PY "$PROJECT/VTC2_Pipeline/combine_calculation_overall_final_eval.py" \
   --folder "$OUTPUT/human_timestamps/eval_report" \
   >>"$LOG_DIR/analysis_${RUN_ID}_humants_wer.log" 2>&1 \
  && log "  [HumanTS WER] aggregation done" || log "  WARNING: [HumanTS WER] aggregation issue"

# ─────────────────────── STEP 8: Excel reports ───────────────────────────────
log "STEP 8 — Excel reports"
for script in wer_per_clip_14clips_excel.py wer_per_recording_14clips_excel.py word_count_comparison_14clips_excel.py; do
  [[ -f "$PROJECT/$script" ]] || { log "  WARNING: $script not found — skipping"; continue; }
  PY "$PROJECT/$script" >>"$LOG_DIR/analysis_${RUN_ID}_excel.log" 2>&1 \
    && log "  $script done" || log "  WARNING: $script issue"
done

# ─────────────────────── final summary ───────────────────────────────────────
log "=== ANALYSIS-ONLY RUN $RUN_ID COMPLETE ==="
log ""
log "Previous results backed up to : $PREV_DIR"
log "Updated results in             : $OUTPUT"
log ""
log "Analysis status:"
for pipe in vtc1 vtc2 whisperx_only human_timestamps; do
  eval_ok=$([ -d "$OUTPUT/$pipe/eval_report" ] && echo "yes" || echo "MISSING")
  lang_ok=$([ -d "$OUTPUT/$pipe/langdetect"  ] && echo "yes" || echo "MISSING")
  wc_ok=$([ -d "$OUTPUT/$pipe/word_count"    ] && echo "yes" || echo "MISSING")
  log "  %-20s  eval_report=%-7s  langdetect=%-7s  word_count=%s" \
      "$pipe" "$eval_ok" "$lang_ok" "$wc_ok"
done
log ""
log "Excel files:"
for f in "$PROJECT"/*.xlsx; do [[ -f "$f" ]] && log "  $f"; done
log ""
log "Logs: $LOG_DIR/analysis_${RUN_ID}_*.log"
