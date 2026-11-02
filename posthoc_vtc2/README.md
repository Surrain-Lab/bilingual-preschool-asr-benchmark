# WhisperX-VAD + VTC2 (post-hoc)

This pipeline takes the transcript of WhisperX-VAD + VTC1 and gives it speaker
labels from VTC2 instead of VTC1, using the same rule (overlap, with priority
KCHI > OCH > FEM > MAL). It is then scored with the same evaluation as the other
pipelines. Nothing is transcribed again and no GPU is needed.

## Running it

```bash
bash posthoc_vtc2/run_posthoc.sh <stage1_output_dir> <whisperx_merged_dir>
```

This writes `<stage1_output_dir>/whisperx_vtc2_posthoc/`, which the stage 2
scripts list as `WhisperX-VAD + VTC2`. Run with VTC1 regions instead, the same
steps give back the WhisperX-VAD + VTC1 transcripts exactly.

## Overall WER in time order

The speaker tagger writes each clip's overall transcript (`{clip}_clean.txt`)
grouped by speaker: all key-child words, then other children, women and men. The
human transcript and the other pipelines are in time order, so the overall
transcripts of both post-hoc pipelines are rebuilt in time order before scoring:

```bash
python posthoc_vtc2/rescore_time_order_build.py SRC_EVAL_VIEW DST_VIEW
python stage1_pipeline/Only_Whisperx/evaluation_step4.py --whisperx-dir DST_VIEW ...
python posthoc_vtc2/rescore_time_order_per_recording.py LABEL=DST_EVAL_REPORT > pooled.csv
```

`rescore_time_order_build.py` first checks that grouping the lines of
`{clip}_tagged.txt` by speaker gives back the existing `_clean.txt` exactly, then
writes the time-ordered version and links all other files. Key-child and adult
WER are scored on single-speaker files and are not affected.

The evaluation scores only words assigned to a speaker
(`transcript_based_speaker_tagger_step3.py` drops `UNKNOWN` and `SPEECH`).

## Result

157 scored clips, averaged over the 14 recordings.

| speaker labels | overall | key child | adult |
|---|---|---|---|
| VTC1 | 73.3% | 102.2% | 89.7% |
| VTC2 | 69.5% | 103.4% | 87.1% |
| difference | −3.7 (p = .013) | +1.2 (p = .52) | −2.6 (p = .21) |
