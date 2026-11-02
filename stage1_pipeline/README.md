# Stage 1: audio and human transcripts to per-clip transcripts

These are the scripts used to produce the transcripts in the paper.

`run_full_pipeline.sh` runs everything. `run_analysis_only.sh` re-runs the
scoring on transcripts that already exist. Both take the output folder as their
first argument. `smoke_test.sh` runs VTC2 and WhisperX on a single clip, which
is a quick way to check an installation.

`Human_Evaluation/` turns the ELAN exports into the cleaned reference text and
counts English and Spanish words in it (step 0). `check_rename_human_files.py`
renames transcript files so they match the audio file names.

The drivers expect this folder's layout under `$ASR_PROJECT`, so copy or link
`stage1_pipeline/` there.

## Environments

VTC1 needs `pyannote-audio` 1.x, which does not install alongside a current
PyTorch, so three environments are used.

| environment | contains | used by |
|---|---|---|
| `venv-whisperx` | PyTorch and torchaudio for your CUDA version, `whisperx==3.7.4` | all transcription |
| `venv-vtc1` | `pyannote-audio` 1.1.x and the voice-type-classifier checkpoints | `VTC1_Pipeline/` |
| `venv-vtc2` | the VTC2 distribution from LAAC-LSCP (managed with uv) | `VTC2_Pipeline/` |

Configuration used for the paper: CUDA 12.8, PyTorch 2.8.0, WhisperX 3.7.4
(faster-whisper 1.2.0, CTranslate2 4.6.0), Whisper `large-v3` in float16,
`batch_size=16`, default VAD. Audio shorter than 30 s is zero-padded before
language detection.

The models are gated on Hugging Face. Accept the terms, create a read token and
export `HF_TOKEN`.

Installation notes:

- `ctranslate2` needs `libcudnn_ops_infer.so.8`, but PyTorch 2.9 and later ship
  cuDNN 9. If WhisperX fails with *Could not load library
  libcudnn_ops_infer.so.8*, install `nvidia-cudnn-cu12==8.9.7.29` and add its
  `lib/` folder to `LD_LIBRARY_PATH`. `smoke_test.sh` does this.
- The VTC2 pipelines do not need a Hugging Face token: VTC2 provides the speech
  segments, so WhisperX's pyannote VAD is not loaded (`vad_method="silero"`).
- Run VTC2's `infer.py` from the folder that contains `VTC-2.0/`, because the
  checkpoint path is relative.
- If the conda environment is not in `$CONDA_BASE/envs/`, set
  `WHISPERX_ENV_PREFIX` before running the driver scripts.

## Pipelines

| folder | pipeline | speech regions | speaker labels |
|---|---|---|---|
| `VTC1_Pipeline/` | VTC1 + WhisperX | VTC1 | VTC1 |
| `VTC2_Pipeline/` | VTC2 + WhisperX | VTC2 | VTC2 |
| `Only_Whisperx/` | WhisperX-VAD + VTC1 (post-hoc) | WhisperX VAD | VTC1, applied after transcription |
| `Human_Timestamps_Pipeline/` | Human-TS + WhisperX (control) | expert boundaries | expert labels |
| `VTC2_NoLangBarrier_Pipeline/` | VTC2-NoLang (control) | VTC2 | VTC2, output in other languages kept |

Each one segments the audio, transcribes it, splits the transcript by speaker,
tags the language of each utterance (`Langdetect_fallback/`), counts words and
scores against the human transcript.

The fourth pipeline in the paper, WhisperX-VAD + VTC2, relabels the
`Only_Whisperx/` transcript with VTC2 speaker regions; see `../posthoc_vtc2/`.

## English and Spanish only

Whisper is not given a language. `model.transcribe(...)` detects it
automatically. If the detected language is not `en` or `es`, the scripts run
`model.detect_language()` once more, and if that is still not `en` or `es`, the
segment's text is discarded (`text = ""`; for example
`VTC2_Pipeline/whisperx_vtc2_transcrib.py`, lines 253–254). The same rule is
used in `VTC1_Pipeline/`, `Human_Timestamps_Pipeline/` and
`Only_Whisperx/whisperx_detect_lang.py`. `VTC2_NoLangBarrier_Pipeline/` keeps
that text.

For VTC2 + WhisperX, 2,824 of the 2,847 segments that end up with no words were
discarded this way, mostly output in Chinese, Korean or Japanese (8,559 words).
The scoring step's `clean_text()` also removes characters outside the Latin
alphabet (`[^a-zA-Z0-9\s]`), so output in non-Latin scripts is never scored.
See `../stage2_aggregate/failure_analysis/`.

## Outputs read by stage 2

```
<output_dir>/<pipeline>/
  eval_report/<recording>/<clip>/<clip>_{overall,kchi,adult,mal,fem}_evaluation.txt
  langdetect/<recording>/<clip>/<clip>_tagged_langdetect.txt
  word_count/...
  rttm_raw/rttm/<clip>.rttm             (VTC2)
  rttm_txt/<clip>/..._all_speakers.txt  (VTC1)
```
