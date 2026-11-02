# Failure analyses

`failure_analysis.py` computes the numbers in the Results on English insertions
and empty segments, the words the post-hoc pipelines leave unassigned, the
comparison of VTC1 and VTC2 adult segments, and the counts that describe the
reference transcripts.

It reads stage-1 outputs and the expert transcripts (the docstring lists the
nine input folders) and writes counts only, with no transcript text and no study
IDs, to `data/derived/failure_analysis/`. `verify.py` checks these files against
`expected/paper_values.json`.

The inputs contain speech from the families and are not included here.
