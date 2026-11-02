# bilingual-preschool-asr-benchmark

Code and derived data for *Measuring language environments in Spanish–English
bilingual preschoolers: Benchmarking automatic speech recognition on daylong
recordings*.

The repository has three stages:

1. `stage1_pipeline/`: runs the speech pipelines (VTC1, VTC2 and WhisperX) on
   audio and writes a transcript for each clip. Needs a GPU.
2. `stage2_aggregate/`: turns those transcripts into the spreadsheets and CSV
   files in `data/derived/`.
3. `stage3_tables_figures/`: builds the tables and figures of the paper from
   `data/derived/`.

The audio and the transcripts are not included, because they contain speech
from the participating families. `data/derived/` has the counts and statistics
the paper is built on, so stage 3 runs without them.

## Reproducing the tables and figures

Python 3.9 or later. No GPU needed; it takes about a minute.

```bash
./setup.sh              # creates .venv, installs requirements.txt, runs verify.py
. .venv/bin/activate
./run_analysis.sh       # verify.py, then figures_out/ and tables_out/
```

`verify.py` recomputes the numbers reported in the paper from `data/derived/`
and compares them with `expected/paper_values.json`. It prints PASS or FAIL for
each value and ends with `ALL CHECKS PASSED` when everything matches.

To build only the tables:

```bash
python verify.py
python stage3_tables_figures/make_tables.py        # table_agreement.tex, table_pipeline_wer.tex
python stage3_tables_figures/make_supp_tables.py   # supp_A1 ... supp_A9
```

## Running on your own recordings

### Inputs

```
audio_dir/                 human_dir/
  R01/                       R01/
    R01_002.wav                R01_002.txt
    R01_007.wav                R01_007.txt
```

Audio is converted to 16 kHz mono WAV. Human transcripts are one file per clip,
one utterance per line, exported from ELAN:

```
[0:00:07.880000 - 0:00:11.593000] (es) [KCHI]: mira, {look} el perro.
```

The speaker is one of `KCHI`, `OCH`, `MAL`, `FEM`. Curly braces mark English
inside a Spanish utterance. `xxx` (unintelligible) and `&=...` (non-speech
sounds) are removed before scoring.

### Models and token

The VTC and pyannote models are gated on Hugging Face. Accept the model terms,
create a read token, and put it in `config/config.yaml` or export `HF_TOKEN`.

### Environments

Stage 1 needs three environments, because VTC1 depends on an old version of
`pyannote-audio` that does not install alongside a current PyTorch. See
`stage1_pipeline/README.md`.

### Run

```bash
cp config/config.example.yaml config/config.yaml   # then edit the paths
bash stage1_pipeline/run_full_pipeline.sh          # stage 1 (GPU, several hours)
bash run_aggregate.sh <stage1_output_dir>          # stage 2, writes data/derived/
./run_analysis.sh                                  # stage 3
```

## Layout

```
config/                 example configuration
data/derived/           counts and statistics used by the paper
expected/               paper_values.json, the values verify.py checks against
stage1_pipeline/        transcription pipelines
stage2_aggregate/       aggregation, inter-coder agreement, failure analyses
stage3_tables_figures/  tables and figures
posthoc_vtc2/           the WhisperX-VAD + VTC2 (post-hoc) pipeline
verify.py               checks data/derived/ against the paper
```

## Data

`data/derived/` contains per-clip word counts, error counts and agreement
statistics. It contains no audio and no transcript text. Recordings are
labelled `P01` to `P14`, the same participant numbers used in the paper.

## Notes

- Error rates are averaged over recordings: errors and reference words are
  summed within each of the 14 recordings, and the 14 rates are averaged.
  Confidence intervals and pairwise tests resample recordings, not clips
  (`numpy.random.default_rng(42)`; 5,000 resamples for intervals, 10,000 for
  each pairwise test). `stage3_tables_figures/make_supp_stats.py` reproduces
  them exactly.
- Two values fall exactly on a rounding boundary at three decimals (Human-TS
  overall 0.7155, VTC2 adult 0.7215). `verify.py` compares unrounded values
  within half of the last printed digit.
- For the two post-hoc pipelines, the overall transcript is scored in time
  order. See `posthoc_vtc2/README.md`.

## Software

WhisperX (Bain et al., 2023), VTC1 (Lavechin et al., 2020), VTC2 / BabyHuBERT
(Charlot et al., 2026), jiwer and langdetect.
