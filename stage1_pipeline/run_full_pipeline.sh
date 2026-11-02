#!/usr/bin/env bash

# --- portability -------------------------------------------------------------
# Set ASR_PROJECT to your pipeline working directory (where outputs go), and
# ASR_VTC1_MODEL / ASR_VTC2_MODEL to your model checkouts. See config/config.example.yaml.
: "${ASR_PROJECT:?set ASR_PROJECT to your working directory (see README)}"

# ============================================================================
# run_full_pipeline.sh — COMPLETE ASR pipeline: audio + annotations → Excel
#
# Usage:
#   bash run_full_pipeline.sh <audio_dir> <annotated_text_dir> [output_dir]
#
# Runs all 4 pipelines end-to-end:
#   VTC1 + WhisperX  |  VTC2 + WhisperX  |  WhisperX-only  |  Human Timestamps
#
# Stages:
#   0.  Human eval reference generation (converts annotated transcripts)
#   pre. Normalize WAVs to 16kHz mono
#   1.  VTC1 speaker model (pyannote env)  → VTC1 transcription (GPU_MAIN)
#   1.  WhisperX-only transcription        → (GPU_MAIN, after VTC1)
#   1.  Human Timestamps transcription     → (GPU_MAIN, after WhisperX)
#   1.  VTC2 transcription                 → (GPU_VTC2, parallel)
#   4.  Language detection + word count    (all 4 pipelines)
#   2+3. WER evaluation + aggregation      (all 4 pipelines)
#   5.  Excel report generation
#
# Features:
#   • Self-detaches (survives SSH logout)
#   • 2-GPU parallel (VTC1+WhisperX+HumanTS=GPU_MAIN; VTC2=GPU_VTC2)
#   • Auto-converts WAVs to 16kHz mono (in-place, already-16k files untouched)
#   • Retry logic for each transcription stage
#   • Per-stage logs under logs/
#   • FINAL_SUMMARY.txt written to output dir when done
# ============================================================================
set -uo pipefail

# ─────────────────────────────── ARGS ────────────────────────────────────────
AUDIO_IN="${1:-}"
ANNOT_IN="${2:-}"
if [[ -z "$AUDIO_IN" || ! -d "$AUDIO_IN" || -z "$ANNOT_IN" || ! -d "$ANNOT_IN" ]]; then
  echo "Usage: bash run_full_pipeline.sh <audio_dir> <annotated_text_dir> [output_dir]"
  echo ""
  echo "  <audio_dir>           must exist; layout: <recording>/<clip>.wav"
  echo "  <annotated_text_dir>  must exist; contains human CHAT transcripts"
  echo "                        and optionally human_transcripts_timestamps/"
  echo "                        for the Human Timestamps pipeline"
  exit 2
fi
AUDIO="$(cd "$AUDIO_IN" && pwd)"
ANNOT="$(cd "$ANNOT_IN" && pwd)"

# ─────────────────────────────── CONFIG ──────────────────────────────────────
PROJECT="${ASR_PROJECT}"
CONDA_BASE="${CONDA_BASE:-/opt/conda}"
WHISPERX_ENV="whisperx"
PYANNOTE_ENV="pyannote"
VTC1_MODEL="${ASR_VTC1_MODEL:?set ASR_VTC1_MODEL to your voice_type_classifier checkout}"
SHIM_DIR="$PROJECT/bin_shims"
GPU_MAIN="1"    # VTC1 + WhisperX-only + Human Timestamps
GPU_VTC2="0"    # VTC2 (runs in parallel with GPU_MAIN)
DEVICE="cuda"
MAX_RETRIES=4
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${3:-$PROJECT/outputs_$RUN_ID}"
LOG_DIR="$PROJECT/logs"
MASTER_LOG="$LOG_DIR/pipeline_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/pipeline_${RUN_ID}_status.txt"
PID_FILE="$LOG_DIR/pipeline_${RUN_ID}.pid"
SUMMARY="$OUTPUT/FINAL_SUMMARY.txt"

# ──────────────────────────── self-detach ────────────────────────────────────
if [[ "${_DETACHED:-0}" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  echo "Launching full pipeline in background (GPU$GPU_MAIN + GPU$GPU_VTC2 parallel)..."
  _DETACHED=1 RUN_ID="$RUN_ID" setsid bash "$0" "$AUDIO" "$ANNOT" "$OUTPUT" \
    >>"$MASTER_LOG" 2>&1 < /dev/null &
  sleep 1
  echo "  Audio    : $AUDIO"
  echo "  Annotated: $ANNOT"
  echo "  Output   : $OUTPUT"
  echo "  Log      : $MASTER_LOG"
  echo "  Status   : $STATUS_FILE"
  echo "  Summary  : $SUMMARY  (written when done)"
  echo ""
  echo "  Watch    :  tail -f \"$MASTER_LOG\""
  echo "  Status   :  cat \"$STATUS_FILE\""
  echo "  Stop     :  kill -- -\$(cat \"$PID_FILE\")"
  exit 0
fi
echo "$$" > "$PID_FILE"
mkdir -p "$LOG_DIR" "$OUTPUT"

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
EXPECTED=$(find "$AUDIO" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
[[ "$EXPECTED" -eq 0 ]] && EXPECTED=1

log() {
  local m="[$(date '+%F %T')] $*"
  echo "$m"
  echo "$m" >> "$STATUS_FILE"
}

count_done() {
  local b="$1" n=0 d
  [[ -d "$b" ]] || { echo 0; return; }
  for d in "$b"/*/; do
    [[ -d "$d" ]] || continue
    find "$d" -type f -print -quit 2>/dev/null | grep -q . && n=$((n+1))
  done
  echo "$n"
}

clips_in() {
  find "$1" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l | tr -d ' '
}

PY() {
  ( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output python "$@" )
}

RP() {
  ( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
      python "$PROJECT/run_pipeline.py" \
      --audio "$AUDIO" --annotated "$ANNOT" --output "$OUTPUT" --device "$DEVICE" "$@" )
}

# ─────────────────────── preflight ───────────────────────────────────────────
: > "$STATUS_FILE"
log "=== FULL PIPELINE $RUN_ID START ==="
log "    audio=$AUDIO  annotated=$ANNOT  output=$OUTPUT"

fail=0
for p in \
    "$PROJECT/run_pipeline.py" \
    "$PROJECT/split_rttm_by_uri.py" \
    "$VTC1_MODEL/apply.sh" \
    "$SHIM_DIR/ffprobe" \
    "$PROJECT/Human_Timestamps_Pipeline/human_to_rttm_txt.py" \
    "$PROJECT/Human_Timestamps_Pipeline/whisperx_human_transcrib.py"; do
  [[ -e "$p" ]] || { log "MISSING: $p"; fail=1; }
done
conda env list | grep -qw "$WHISPERX_ENV" || { log "MISSING conda env: $WHISPERX_ENV"; fail=1; }
conda env list | grep -qw "$PYANNOTE_ENV" || { log "MISSING conda env: $PYANNOTE_ENV"; fail=1; }
command -v uv >/dev/null 2>&1 || log "WARN: 'uv' not on PATH — VTC2 model inference needs it"
[[ "$fail" -ne 0 ]] && { log "Preflight FAILED — aborting."; exit 1; }
rm -f "$PROJECT/.vtc2_handled_externally"

# ─────────────────────── step 0: human eval reference ────────────────────────
log "STEP 0 — Human eval reference generation"
RP --steps 0 >>"$LOG_DIR/pipeline_${RUN_ID}_step0.log" 2>&1 \
  && log "  STEP 0 done" \
  || log "  WARNING: STEP 0 had issues (check pipeline_${RUN_ID}_step0.log)"

# Count English/Spanish words from original human CHAT transcripts
log "STEP 0b — Human transcript word count (English/Spanish)"
PY "$PROJECT/Human_Evaluation/count_eng_esp_final.py" \
   -i "$ANNOT/human_transcripts_timestamps" \
   -o "$OUTPUT/human_eval/word_count" \
   >>"$LOG_DIR/pipeline_${RUN_ID}_step0.log" 2>&1 \
  && log "  human word count done" \
  || log "  WARNING: human word count had issues"

log "STEP 0b — Human word count Excel"
PY "$PROJECT/Human_Evaluation/build_human_excel.py" \
   --human-dir "$ANNOT/human_transcripts_timestamps" \
   --csv-dir   "$OUTPUT/human_eval/word_count" \
   --out-dir   "$OUTPUT/human_eval" \
   >>"$LOG_DIR/pipeline_${RUN_ID}_step0.log" 2>&1 \
  && log "  human word count Excel done" \
  || log "  WARNING: human word count Excel had issues"

# ─────────────────────── pre-step 1: normalize audio ─────────────────────────
log "PRE-STEP 1 — Normalize WAVs to 16kHz mono"
( cd "$PROJECT" && conda run -n "$WHISPERX_ENV" --no-capture-output \
    python -c "import run_pipeline as r; r.ensure_16k_mono('$AUDIO')" ) \
  >>"$LOG_DIR/pipeline_${RUN_ID}_normalize.log" 2>&1 \
  && log "  normalize done" \
  || log "  WARNING: normalize had issues"

# ─────────────────── restructure VTC1 rttm_txt for WhisperX tagger ───────────
# After VTC1 transcription, rttm_txt layout is <clip>/<clip>/*.txt but the
# WhisperX speaker-tagger expects <recording>/<clip>/*.txt. This function
# rebuilds the layout so the two match.
reorg_vtc1_rttm_txt() {
  local RT="$OUTPUT/vtc1/rttm_txt"
  local TMP="$OUTPUT/vtc1/rttm_txt_byrec"
  local n=0 recdir rec wav clip src
  [[ -d "$RT" ]] || return 0
  rm -rf "$TMP"
  for recdir in "$AUDIO"/*/; do
    [[ -d "$recdir" ]] || continue
    rec=$(basename "$recdir")
    for wav in "$recdir"*.wav; do
      [[ -e "$wav" ]] || continue
      clip=$(basename "$wav" .wav)
      src="$RT/$clip/$clip"
      [[ -d "$src" ]] || continue
      mkdir -p "$TMP/$rec/$clip"
      cp "$src"/*.txt "$TMP/$rec/$clip/" 2>/dev/null
      n=$((n+1))
    done
  done
  if [[ "$n" -gt 0 ]]; then
    rm -rf "$RT"; mv "$TMP" "$RT"
    log "  rttm_txt restructured: $n clip(s) moved to <recording>/<clip>/ layout"
  else
    rm -rf "$TMP"
  fi
}

# ─── build GT-aligned eval view (symlinks: <recording>/<clip> → hyp clipdir) ──
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

# ───────── GPU_MAIN stream: VTC1 → WhisperX-only → Human Timestamps ──────────
gpu_main_stream() {
  export CUDA_VISIBLE_DEVICES="$GPU_MAIN"
  local RTTM_BASE="$VTC1_MODEL/output_voice_type_classifier"
  local a rec bn rt

  # ── VTC1 speaker model (pyannote env) ────────────────────────────────────
  log "[GPU$GPU_MAIN] PHASE A — VTC1 speaker model (pyannote env)"
  for rec in "$AUDIO"/*/; do
    [[ -d "$rec" ]] || continue
    bn=$(basename "$rec"); rt="$RTTM_BASE/$bn/all.rttm"
    if [[ -s "$rt" ]]; then log "  [VTC1-model] $bn exists — skip"; continue; fi
    a=1
    while (( a <= MAX_RETRIES )); do
      log "  [VTC1-model] $bn attempt $a/$MAX_RETRIES"
      ( cd "$VTC1_MODEL" && conda run -n "$PYANNOTE_ENV" --no-capture-output \
          bash apply.sh "$rec" --device=gpu ) \
        >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1model.log" 2>&1
      [[ -s "$rt" ]] && { log "  [VTC1-model] $bn done"; break; }
      a=$((a+1)); sleep 10
    done
    [[ -s "$rt" ]] || log "  WARNING: [VTC1-model] $bn INCOMPLETE"
  done
  log "  [VTC1-model] $(find "$RTTM_BASE" -name all.rttm -size +0c 2>/dev/null | wc -l | tr -d ' ') RTTM(s) present"

  # ── VTC1 transcription ───────────────────────────────────────────────────
  log "[GPU$GPU_MAIN] PHASE B — VTC1 transcription (whisperx env)"
  a=1
  while (( a <= MAX_RETRIES )); do
    log "  [VTC1-transcribe] attempt $a/$MAX_RETRIES (done: $(count_done "$OUTPUT/vtc1/speaker_separated")/$EXPECTED)"
    rm -rf "$OUTPUT/vtc1/rttm_per_clip" "$OUTPUT/vtc1/rttm_txt"
    ( cd "$PROJECT" &&
      conda run -n "$WHISPERX_ENV" --no-capture-output \
        python split_rttm_by_uri.py "$RTTM_BASE" "$OUTPUT/vtc1/rttm_per_clip" "$AUDIO" &&
      conda run -n "$WHISPERX_ENV" --no-capture-output \
        python VTC1_Pipeline/rttm_txt_unmerged.py \
          -i "$OUTPUT/vtc1/rttm_per_clip" -o "$OUTPUT/vtc1/rttm_txt" &&
      conda run -n "$WHISPERX_ENV" --no-capture-output \
        python VTC1_Pipeline/whisperx_vtc_transcribe_unmerged.py \
          --audio "$AUDIO" --rttm "$OUTPUT/vtc1/rttm_txt" \
          --output "$OUTPUT/vtc1/tagged_transcripts" &&
      conda run -n "$WHISPERX_ENV" --no-capture-output \
        python VTC1_Pipeline/split_transcript_step2_unmerged.py \
          -i "$OUTPUT/vtc1/tagged_transcripts" -o "$OUTPUT/vtc1/speaker_separated"
    ) >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1.log" 2>&1
    [[ "$(count_done "$OUTPUT/vtc1/speaker_separated")" -ge "$EXPECTED" ]] \
      && { log "  [VTC1-transcribe] DONE ($(count_done "$OUTPUT/vtc1/speaker_separated")/$EXPECTED)"; break; }
    log "  [VTC1-transcribe] incomplete — retry in 15s"; a=$((a+1)); sleep 15
  done

  # Restructure rttm_txt so WhisperX tagger can match files by recording
  reorg_vtc1_rttm_txt

  # ── WhisperX-only transcription ──────────────────────────────────────────
  log "[GPU$GPU_MAIN] WhisperX-only transcription"
  a=1
  while (( a <= MAX_RETRIES )); do
    log "  [WhisperX] attempt $a/$MAX_RETRIES (done: $(count_done "$OUTPUT/whisperx_only/tagged_transcripts")/$EXPECTED)"
    RP --pipelines whisperx --steps 1 >>"$LOG_DIR/pipeline_${RUN_ID}_whisperx.log" 2>&1
    [[ "$(count_done "$OUTPUT/whisperx_only/tagged_transcripts")" -ge "$EXPECTED" ]] \
      && { log "  [WhisperX] DONE"; break; }
    log "  [WhisperX] incomplete — retry in 15s"; a=$((a+1)); sleep 15
  done

  # ── Human Timestamps pipeline ─────────────────────────────────────────────
  log "[GPU$GPU_MAIN] Human Timestamps pipeline"
  local HTS_OUT="$OUTPUT/human_timestamps"
  local HUMAN_TS="$ANNOT/human_transcripts_timestamps"
  mkdir -p "$HTS_OUT"

  if [[ ! -d "$HUMAN_TS" ]]; then
    log "  WARNING: $HUMAN_TS not found — Human Timestamps pipeline skipped"
    log "      (create annotated_text_dir/human_transcripts_timestamps/ to enable)"
  else
    # Convert human timestamp annotations to rttm_txt format
    PY "$PROJECT/Human_Timestamps_Pipeline/human_to_rttm_txt.py" \
       -i "$HUMAN_TS" -o "$HTS_OUT/rttm_txt" \
       >>"$LOG_DIR/pipeline_${RUN_ID}_humants_convert.log" 2>&1 \
      && log "  [HumanTS] convert done" || { log "  WARNING: [HumanTS] convert FAILED"; return; }

    local N_CLIPS
    N_CLIPS=$(find "$HTS_OUT/rttm_txt" -name "*_all_speakers.txt" 2>/dev/null | wc -l | tr -d ' ')
    log "  [HumanTS] $N_CLIPS timestamp clip files found"

    # Transcribe using WhisperX on human-defined segment boundaries
    a=1
    while (( a <= MAX_RETRIES )); do
      log "  [HumanTS-transcribe] attempt $a/$MAX_RETRIES (clips: $(clips_in "$HTS_OUT/tagged_transcripts"))"
      PY "$PROJECT/Human_Timestamps_Pipeline/whisperx_human_transcrib.py" \
         --audio "$AUDIO" --rttm "$HTS_OUT/rttm_txt" --output "$HTS_OUT/tagged_transcripts" \
         >>"$LOG_DIR/pipeline_${RUN_ID}_humants_transcribe.log" 2>&1
      [[ "$(clips_in "$HTS_OUT/tagged_transcripts")" -ge "$N_CLIPS" ]] \
        && { log "  [HumanTS-transcribe] DONE ($(clips_in "$HTS_OUT/tagged_transcripts") clips)"; break; }
      log "  [HumanTS-transcribe] incomplete — retry in 15s"; a=$((a+1)); sleep 15
    done

    # Split by speaker
    PY "$PROJECT/VTC2_Pipeline/labels_seperator.py" \
       -i "$HTS_OUT/tagged_transcripts" -o "$HTS_OUT/speaker_separated" \
       >>"$LOG_DIR/pipeline_${RUN_ID}_humants_split.log" 2>&1 \
      && log "  [HumanTS] speaker split done" \
      || log "  WARNING: [HumanTS] speaker split issue"

    # Language detection + word count
    PY "$PROJECT/Human_Timestamps_Pipeline/Langdetect_fallback/langdetect_tags.py" \
       -i "$HTS_OUT/speaker_separated" -o "$HTS_OUT/langdetect" \
       >>"$LOG_DIR/pipeline_${RUN_ID}_humants_langdetect.log" 2>&1 \
      && log "  [HumanTS] langdetect done" \
      || log "  WARNING: [HumanTS] langdetect issue"

    PY "$PROJECT/Human_Timestamps_Pipeline/Langdetect_fallback/count_words.py" \
       -i "$HTS_OUT/langdetect" -o "$HTS_OUT/word_count" \
       >>"$LOG_DIR/pipeline_${RUN_ID}_humants_count.log" 2>&1 \
      && log "  [HumanTS] word count done" \
      || log "  WARNING: [HumanTS] word count issue"
  fi
}

# ───────────────── GPU_VTC2 stream: VTC2 (runs in parallel) ──────────────────
vtc2_stream() {
  export CUDA_VISIBLE_DEVICES="$GPU_VTC2"
  local TARGET=$(( $(find "$AUDIO" -iname '*.wav' | wc -l) * 9 / 10 ))
  local a
  log "[GPU$GPU_VTC2] VTC2 transcription (clip target >= $TARGET)"
  a=1
  while (( a <= MAX_RETRIES )); do
    log "  [VTC2] attempt $a/$MAX_RETRIES (clips: $(clips_in "$OUTPUT/vtc2/tagged_transcripts"))"
    RP --pipelines vtc2 --steps 1 >>"$LOG_DIR/pipeline_${RUN_ID}_vtc2.log" 2>&1
    [[ "$(clips_in "$OUTPUT/vtc2/tagged_transcripts")" -ge "$TARGET" ]] \
      && { log "  [VTC2] DONE ($(clips_in "$OUTPUT/vtc2/tagged_transcripts") clips)"; break; }
    log "  [VTC2] incomplete — retry in 15s"; a=$((a+1)); sleep 15
  done
}

# ─── VTC2 no-language-barrier (runs on GPU_MAIN after VTC2 stream finishes) ───
# Reuses VTC2 RTTM timestamps (diarization is language-independent) and
# re-transcribes keeping ALL detected languages. The standard VTC2 run filters
# to en/es only; this variant preserves every language for word-count analysis.
vtc2_nolang_barrier() {
  local PIPE="$PROJECT/VTC2_NoLangBarrier_Pipeline"
  local RTTM_TXT="$OUTPUT/vtc2/rttm_txt"
  local OUT="$OUTPUT/vtc2_nolang_barrier"
  local a N_CLIPS TARGET
  export CUDA_VISIBLE_DEVICES="$GPU_MAIN"

  if ! find "$RTTM_TXT" -name "*.txt" -quit 2>/dev/null | grep -q .; then
    log "[VTC2-nolang] VTC2 rttm_txt not ready — skipping"
    return
  fi

  N_CLIPS=$(find "$RTTM_TXT" -name "*_all_speakers.txt" 2>/dev/null | wc -l | tr -d ' ')
  TARGET=$(( N_CLIPS * 9 / 10 ))
  log "[VTC2-nolang] starting (reusing VTC2 timestamps: $N_CLIPS clips, target>=$TARGET)"
  mkdir -p "$OUT"

  # Transcribe keeping all detected languages (no en/es filter)
  a=1
  while (( a <= MAX_RETRIES )); do
    log "  [VTC2-nolang transcribe] attempt $a/$MAX_RETRIES (clips: $(clips_in "$OUT/tagged_transcripts"))"
    PY "$PIPE/whisperx_vtc2_transcrib.py" \
       --audio "$AUDIO" --rttm "$RTTM_TXT" --output "$OUT/tagged_transcripts" \
       >>"$LOG_DIR/pipeline_${RUN_ID}_vtc2nolang.log" 2>&1
    [[ "$(clips_in "$OUT/tagged_transcripts")" -ge "$TARGET" ]] \
      && { log "  [VTC2-nolang transcribe] DONE ($(clips_in "$OUT/tagged_transcripts") clips)"; break; }
    log "  [VTC2-nolang transcribe] incomplete — retry in 15s"; a=$((a+1)); sleep 15
  done

  # Split by speaker
  PY "$PIPE/labels_seperator.py" \
     -i "$OUT/tagged_transcripts" -o "$OUT/speaker_separated" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc2nolang.log" 2>&1 \
    && log "  [VTC2-nolang] split done" || log "  WARNING: [VTC2-nolang] split issue"

  # Language detection (real detected language, any language)
  PY "$PIPE/Langdetect_fallback/langdetect_tags.py" \
     -i "$OUT/speaker_separated" -o "$OUT/langdetect" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc2nolang.log" 2>&1 \
    && log "  [VTC2-nolang] langdetect done" || log "  WARNING: [VTC2-nolang] langdetect issue"

  # Word count (per speaker + per language)
  PY "$PIPE/Langdetect_fallback/count_words.py" \
     -i "$OUT/langdetect" -o "$OUT/word_count" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc2nolang.log" 2>&1 \
    && log "  [VTC2-nolang] word count done" || log "  WARNING: [VTC2-nolang] word count issue"

  log "[VTC2-nolang] complete — results: $OUT/{tagged_transcripts,speaker_separated,langdetect,word_count}"
}

# ─────────────── launch both GPU streams in parallel ─────────────────────────
log "Launching parallel streams: VTC2 on GPU$GPU_VTC2 (background), VTC1+WhisperX+HumanTS on GPU$GPU_MAIN (foreground)"
vtc2_stream &
VTC2_BG=$!
gpu_main_stream
log "GPU$GPU_MAIN stream finished; waiting for VTC2 (GPU$GPU_VTC2)..."
wait "$VTC2_BG"
log "Both GPU streams complete."

# VTC2 no-language-barrier runs on GPU_MAIN after VTC2 timestamps are ready
vtc2_nolang_barrier

# ─────────────────────── step 4: language + word count ───────────────────────
log "STEP 4 — Language detection + word count (VTC1, VTC2, WhisperX)"
a=1
while (( a <= MAX_RETRIES )); do
  RP --pipelines vtc1 vtc2 whisperx --steps 4 >>"$LOG_DIR/pipeline_${RUN_ID}_step4.log" 2>&1
  [[ -d "$OUTPUT/vtc1/word_count" && -d "$OUTPUT/vtc2/word_count" && -d "$OUTPUT/whisperx_only/word_count" ]] \
    && { log "  STEP 4 done"; break; }
  a=$((a+1))
  [[ "$a" -gt "$MAX_RETRIES" ]] && { log "  WARNING: STEP 4 gave up after $MAX_RETRIES attempts"; break; }
  sleep 15
done

# ─────────────────────── step 2+3: WER evaluation ────────────────────────────
log "STEP 2+3 — WER evaluation + aggregation (VTC2, WhisperX)"
# VTC1 WER is handled separately by vtc1_nospeech_wer() below, which first
# removes the [SPEECH] umbrella segment before evaluating — avoiding the
# double-counting inflation that pushes VTC1 WER above 1.0.
RP --pipelines vtc2 whisperx --steps 2 3 >>"$LOG_DIR/pipeline_${RUN_ID}_wer.log" 2>&1 \
  && log "  WER (vtc2/whisperx) done" \
  || log "  WARNING: WER had issues (check pipeline_${RUN_ID}_wer.log)"

# WER for Human Timestamps (not handled by run_pipeline.py)
HTS_OUT="$OUTPUT/human_timestamps"
HTS_VIEW="$HTS_OUT/eval_view"
GT_DIR="$OUTPUT/human_eval/five_mins_benchmark"

if [[ -d "$HTS_OUT/speaker_separated" && -d "$GT_DIR" ]]; then
  log "STEP 2+3 — WER evaluation (Human Timestamps)"
  n=$(build_eval_view "$HTS_OUT/speaker_separated" "$HTS_VIEW")
  log "  [HumanTS WER] eval view built: $n clips"

  PY "$PROJECT/VTC2_Pipeline/evaluation.py" \
     --ground-truth-dir "$GT_DIR" \
     --whisperx-dir     "$HTS_VIEW" \
     --output-dir       "$HTS_OUT/eval_report" \
     --language         Spanish \
     >>"$LOG_DIR/pipeline_${RUN_ID}_humants_wer.log" 2>&1 \
    && log "  [HumanTS] WER evaluation done" \
    || log "  WARNING: [HumanTS] WER evaluation issue"

  PY "$PROJECT/VTC2_Pipeline/combine_calculation_overall_final_eval.py" \
     --folder "$HTS_OUT/eval_report" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_humants_wer.log" 2>&1 \
    && log "  [HumanTS] WER aggregation done" \
    || log "  WARNING: [HumanTS] WER aggregation issue"
else
  log "  [HumanTS WER] skipped — speaker_separated or GT dir not found"
  log "      speaker_separated: $([ -d "$HTS_OUT/speaker_separated" ] && echo found || echo MISSING)"
  log "      GT dir           : $([ -d "$GT_DIR" ] && echo found || echo MISSING)"
fi

# ── VTC1 WER without [SPEECH] umbrella (uses already-generated tagged_transcripts) ──
# VTC1 emits a [SPEECH] umbrella segment that duplicates every word in the
# combined hypothesis, pushing WER above 1.0. This step re-splits the existing
# tagged_transcripts excluding [SPEECH] from the combined file, then re-runs WER.
# No re-transcription — pure CPU post-processing on already-generated files.
vtc1_nospeech_wer() {
  local TAGGED="$OUTPUT/vtc1/tagged_transcripts"
  local GT_DIR="$OUTPUT/human_eval/five_mins_benchmark"

  if [[ ! -d "$TAGGED" ]]; then
    log "[VTC1-nospeech] vtc1/tagged_transcripts not found — skipping"
    return
  fi
  if [[ ! -d "$GT_DIR" ]]; then
    log "[VTC1-nospeech] GT dir not found — skipping"
    return
  fi

  log "[VTC1-nospeech] generating labels_seperator_nospeech.py"
  PY - <<'PYEOF' >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1nospeech.log" 2>&1
import os
from pathlib import Path
P = Path(os.environ["ASR_PROJECT"])
src = (P / "VTC2_Pipeline/labels_seperator.py").read_text(encoding="utf-8")
old = "            by_speaker['all'].append(parsed)"
new = ("            if speaker != 'SPEECH':  # exclude [SPEECH] umbrella from combined\n"
       "                by_speaker['all'].append(parsed)")
assert old in src, "anchor line not found in labels_seperator.py"
out = P / "VTC1_Pipeline/labels_seperator_nospeech.py"
out.write_text(src.replace(old, new, 1), encoding="utf-8")
print(f"created {out}")
PYEOF
  [[ $? -ne 0 ]] && { log "  WARNING: [VTC1-nospeech] script generation failed — skipping"; return; }

  log "[VTC1-nospeech] re-splitting tagged_transcripts (no [SPEECH] in combined)"
  rm -rf "$OUTPUT/vtc1/speaker_separated_nospeech"
  PY "$PROJECT/VTC1_Pipeline/labels_seperator_nospeech.py" \
     -i "$TAGGED" -o "$OUTPUT/vtc1/speaker_separated_nospeech" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1nospeech.log" 2>&1 \
    && log "  [VTC1-nospeech] split done" || { log "  WARNING: [VTC1-nospeech] split failed — skipping"; return; }

  log "[VTC1-nospeech] building eval view"
  local VIEW="$OUTPUT/vtc1/eval_view_nospeech"
  local n
  n=$(build_eval_view "$OUTPUT/vtc1/speaker_separated_nospeech" "$VIEW")
  log "  [VTC1-nospeech] eval view: $n clips"

  log "[VTC1-nospeech] WER evaluation"
  [[ -d "$OUTPUT/vtc1/eval_report" && ! -d "$OUTPUT/vtc1/eval_report.bak_withspeech" ]] \
    && mv "$OUTPUT/vtc1/eval_report" "$OUTPUT/vtc1/eval_report.bak_withspeech"
  rm -rf "$OUTPUT/vtc1/eval_report"
  PY "$PROJECT/VTC1_Pipeline/evaluation_unmerged_not_applicable.py" \
     --ground-truth-dir "$GT_DIR" \
     --whisperx-dir     "$VIEW" \
     --output-dir       "$OUTPUT/vtc1/eval_report" \
     --language         Spanish \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1nospeech.log" 2>&1 \
    && log "  [VTC1-nospeech] WER evaluation done" || log "  WARNING: [VTC1-nospeech] WER evaluation issue"

  PY "$PROJECT/VTC1_Pipeline/combine_calculation_unmerged_overall_final_eval.py" \
     --folder "$OUTPUT/vtc1/eval_report" \
     >>"$LOG_DIR/pipeline_${RUN_ID}_vtc1nospeech.log" 2>&1 \
    && log "  [VTC1-nospeech] WER aggregation done" || log "  WARNING: [VTC1-nospeech] WER aggregation issue"

  log "[VTC1-nospeech] complete — eval_report now uses no-[SPEECH] hypothesis"
  log "  (original eval_report backed up as eval_report.bak_withspeech)"
}

log "VTC1 WER (no-[SPEECH] umbrella)"
vtc1_nospeech_wer

# ─────────────────────── step 5: Excel reports ───────────────────────────────
log "STEP 5 — Excel report generation"
RP --steps 5 >>"$LOG_DIR/pipeline_${RUN_ID}_excel.log" 2>&1 \
  && log "  STEP 5 done" \
  || log "  WARNING: STEP 5 had issues (check pipeline_${RUN_ID}_excel.log)"

for script in wer_per_clip_14clips_excel.py wer_per_recording_14clips_excel.py word_count_comparison_14clips_excel.py; do
  [[ -f "$PROJECT/$script" ]] || { log "  WARNING: $script not found — skipping"; continue; }
  PY "$PROJECT/$script" >>"$LOG_DIR/pipeline_${RUN_ID}_excel.log" 2>&1 \
    && log "  $script done" || log "  WARNING: $script issue"
done

# ─────────────────────── final summary ───────────────────────────────────────
recs() { find "$OUTPUT/$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; }
{
  echo "================  FINAL SUMMARY  —  $(date '+%F %T')  ================"
  echo "Audio    : $AUDIO"
  echo "Annotated: $ANNOT"
  echo "Output   : $OUTPUT"
  echo "Input    : $EXPECTED recordings, $(find "$AUDIO" -iname '*.wav' | wc -l) clips"
  echo ""
  echo "TRANSCRIPTS (recordings / clips):"
  printf "  %-24s : %s recordings, %s clips\n" \
    "VTC1"                  "$(recs vtc1/speaker_separated)"                   "$(clips_in "$OUTPUT/vtc1/speaker_separated")"
  printf "  %-24s : %s recordings, %s clips\n" \
    "VTC2"                  "$(recs vtc2/speaker_separated)"                   "$(clips_in "$OUTPUT/vtc2/speaker_separated")"
  printf "  %-24s : %s recordings, %s clips\n" \
    "VTC2 (no-lang-barrier)" "$(recs vtc2_nolang_barrier/speaker_separated)"  "$(clips_in "$OUTPUT/vtc2_nolang_barrier/speaker_separated")"
  printf "  %-24s : %s recordings, %s clips\n" \
    "WhisperX-only"         "$(recs whisperx_only/tagged_transcripts)"         "$(clips_in "$OUTPUT/whisperx_only/tagged_transcripts")"
  printf "  %-24s : %s recordings, %s clips\n" \
    "Human Timestamps"      "$(recs human_timestamps/speaker_separated)"       "$(clips_in "$OUTPUT/human_timestamps/speaker_separated")"
  echo ""
  echo "WER eval_report present:"
  for p in vtc1 vtc2 whisperx_only human_timestamps; do
    printf "  %-24s : %s\n" "$p" "$([ -d "$OUTPUT/$p/eval_report" ] && echo yes || echo NO)"
  done
  echo "  (vtc2_nolang_barrier: word-count analysis only, no WER)"
  echo ""
  echo "Word count present:"
  for p in vtc1 vtc2 vtc2_nolang_barrier whisperx_only human_timestamps; do
    printf "  %-24s : %s\n" "$p" "$([ -d "$OUTPUT/$p/word_count" ] && echo yes || echo NO)"
  done
  echo ""
  echo "Excel outputs:"
  for f in "$PROJECT"/*.xlsx "$OUTPUT"/*.xlsx; do
    [[ -f "$f" ]] && echo "  $f"
  done
  echo ""
  echo "Logs: $LOG_DIR/pipeline_${RUN_ID}_*.log"
  echo "==========================================================================="
} > "$SUMMARY"

log "=== FULL PIPELINE $RUN_ID COMPLETE — summary: $SUMMARY ==="
cat "$SUMMARY"
