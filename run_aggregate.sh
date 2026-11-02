#!/usr/bin/env bash
# Stage 2: per-clip transcripts from stage 1 -> the spreadsheets in data/derived/.
#
# data/derived/ already holds the outputs of the paper's run, so this is only
# needed after you have run stage 1 on your own recordings. Some scripts expect
# the folder layout described in stage1_pipeline/README.md.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
WORK="${1:-./work}"

if [ ! -d "$WORK" ]; then
  echo "usage: $0 <stage1_output_dir>" >&2
  echo "  (no such directory: $WORK)" >&2
  exit 1
fi

echo "== aggregating from $WORK =="
for s in word_count_comparison_14clips_excel.py wer_per_clip_14clips_excel.py \
         wer_per_recording_14clips_excel.py vocab_by_speaker.py wer_by_language.py; do
  echo "-- $s"
  "$PYTHON" "stage2_aggregate/$s" "$WORK" || echo "   (failed - see README)"
done
echo
echo "then anonymise before sharing:"
echo "   $PYTHON stage2_aggregate/anonymise_ids.py data/derived"
