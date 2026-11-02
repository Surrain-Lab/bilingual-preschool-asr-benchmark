# Stage 3: tables and figures

Run everything with `../run_analysis.sh`, or each script on its own.

| script | output |
|---|---|
| `make_tables.py` | `tables_out/table_agreement.tex`, `tables_out/table_pipeline_wer.tex` |
| `make_supp_tables.py` | `tables_out/supp_A1_*.tex` to `supp_A9_*.tex`, the supplementary tables |
| `make_supp_stats.py` | `data/derived/supp_bootstrap.csv`, `supp_pairwise.csv` (confidence intervals and pairwise tests) |
| `make_paper_figures.py` | `figures_out/fig_total_words.png`, `fig_lexical_diversity.png` |
| `make_fig_error_counts.py` | `figures_out/fig_error_counts_by_speaker_language.png` |
| `make_fig_corr_by_model.py` | `figures_out/corr_by_model.png` |

`make_supp_tables.py` prints WER as percentages. Set `PERCENT = False` at the
top of the file to print proportions instead.

`make_paper_figures.py` and `make_fig_corr_by_model.py` first check the
correlations, slopes and clip counts printed in the paper and stop without
drawing if any differ.

```bash
python make_paper_figures.py --validate-only   # check only
python make_paper_figures.py --scale 1.35      # font size used in the paper
```
