#!/usr/bin/env bash
# One-clip smoke test: checks that VTC2 and WhisperX run, and, if you pass a
# reference RTTM, that VTC2 reproduces it exactly.
#
# It does not run run_full_pipeline.sh end to end.
#
#   ASR_VTC2_DIR=... ASR_WAV=... [ASR_REF_RTTM=...] bash stage1_pipeline/smoke_test.sh
set -euo pipefail
: "${ASR_VTC2_DIR:?path to the VTC checkout that CONTAINS VTC-2.0/model/best.ckpt}"
: "${ASR_WAV:?path to one 16 kHz mono wav}"
WORK="${ASR_WORK:-$(cd "$(dirname "$0")/.." && pwd)/work/smoke}"
mkdir -p "$WORK/in" "$WORK/out"; cp "$ASR_WAV" "$WORK/in/"

echo "== 1/2  VTC2 segmentation =="
# run from the directory holding VTC-2.0/, the checkpoint path is relative
( cd "$ASR_VTC2_DIR" && uv run scripts/infer.py --wavs "$WORK/in" --output "$WORK/out" --device "${ASR_DEVICE:-cuda}" )
RTTM=$(find "$WORK/out/rttm" -name '*.rttm' | head -1)
echo "   $(wc -l <"$RTTM") segments -> $RTTM"
if [ -n "${ASR_REF_RTTM:-}" ] && [ -f "$ASR_REF_RTTM" ]; then
  if diff -q "$RTTM" "$ASR_REF_RTTM" >/dev/null; then echo "   MATCHES the archived RTTM byte-for-byte"
  else echo "   DIFFERS from the archived RTTM:"; diff "$RTTM" "$ASR_REF_RTTM" | head -5; fi
fi

echo "== 2/2  WhisperX transcription =="
# ctranslate2 needs cuDNN 8; torch >= 2.9 ships cuDNN 9, so point at the v8 runtime
CUDNN=$(find "$(dirname "$(command -v python)")/.." -name 'libcudnn_ops_infer.so.8' 2>/dev/null | head -1 || true)
[ -n "$CUDNN" ] && export LD_LIBRARY_PATH="$(dirname "$CUDNN"):${LD_LIBRARY_PATH:-}"
python - "$ASR_WAV" <<'PY'
import sys, time, whisperx
m = whisperx.load_model("large-v3", device="cuda", compute_type="float16",
                        language="es", vad_method="silero")
a = whisperx.load_audio(sys.argv[1]); t = time.time()
r = m.transcribe(a, batch_size=16)
print(f"   {len(r['segments'])} segments, "
      f"{sum(len(s['text'].split()) for s in r['segments'])} words, "
      f"lang={r['language']}, {time.time()-t:.0f}s for {len(a)/16000:.0f}s audio")
PY
echo "smoke test complete."
