# Stage 2: per-clip transcripts to spreadsheets

Turns the per-clip outputs of stage 1 into the `.xlsx` and `.csv` files in
`data/derived/` that stage 3 reads.

| script | output |
|---|---|
| `wer_per_clip_14clips_excel.py` | WER per clip for every pipeline |
| `wer_per_recording_14clips_excel.py` | `wer_per_recording_14clips.xlsx`, WER per recording |
| `word_count_comparison_14clips_excel.py` | `word_count_comparison_14clips.xlsx`, word counts per clip by speaker and language, human and automatic |
| `vocab_by_speaker.py` | `vocab_by_speaker.csv`, word types and tokens per clip, speaker and language |
| `make_vocab_table.py` | `vocab_by_clip.csv`, the same per clip and side |
| `wer_by_language.py` | `wer_by_language.csv`, errors per language |
| `recompute_supp_wer.py` | `supp_wer_by_pipeline.csv`, macro-averaged WER per pipeline |
| `irr_symmetric_wer.py` | `irr_symmetric_wer.csv`, WER between coders; also prints the word-count ICCs stored in `irr_icc.csv` |
| `irr_interannotator.py`, `krippendorff_alpha.py` | `irr_alpha/`, Krippendorff's α for speech, speaker and intelligibility |
| `irr_language_agreement.py` | `irr_alpha/language_alpha.csv`, α for language tags |
| `failure_analysis/failure_analysis.py` | `failure_analysis/`, see the README there |
| `anonymise_ids.py` | replaces study recording IDs with `P01` to `P14` |

`transcript_tags.py` holds the transcript parsing shared by several scripts.

## Word counts and different words

Total word counts come from `word_count_comparison_14clips.xlsx` (all 168
clips). Counts of different words come from `vocab_by_speaker.csv`, which only
has rows where the speaker talks in the clip (158 clips with adult speech, 161
with key-child speech). The paper uses each file for its own measure, and
`verify.py` checks both.

## Other-child speech in coder agreement

In `irr_symmetric_wer.csv`, speech from other children is left out of the
key-child and adult columns and kept in the overall column, as in the pipeline
comparison. This gives the pooled overall coder WER of 46.0% reported in the
paper. The key-child (35.4%) and adult (53.9%) values do not depend on this.
