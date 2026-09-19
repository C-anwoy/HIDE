# Revision deliverables — 19 September 2026

- `pdf/COLI_template.pdf`: compiled revised manuscript, 47 pages.
- `pdf/response_letter.pdf`: point-by-point response, 6 pages.
- `pdf/llama3-8b_Mechanistic_Flow_updated.pdf` and
  `pdf/gemma-2-9b_Mechanistic_Flow_updated.pdf`: replacement Figure 2 panels.
- `pdf/computation_time_plot_updated.pdf`: replacement Figure 3, verified against the supplied timing CSV.
- `pdf/Scalability_Analysis_updated.pdf`: replacement Figure 4.
- `HIDE_final_revision_source.zip`: complete LaTeX source with figures and tables;
  manuscript and response are separate compilation targets.

`SHA256.json` records the hashes of these files. The new analysis is in
`results/answer-2000-analysis-01`; paper edits and numerical qualifications are in
`docs/FINAL_MANUSCRIPT_CHANGES.md` and `docs/SUBMISSION_NUMERICAL_AUDIT.md`.

The uploaded results are used without further GPU runs. The new baseline results
do not show a general HIDE advantage over attention; the paper and response report
this and the selected-token-count limitation. Historical tables are kept distinct
from the corrected protocol. The retained timing suite has its own stated protocol.

Review the revised claims before submission. The journal's bundle also requires
the current original decision letter, which is not generated or sent here.

The paper uses magenta `\revisionr3{}` markup. The timing headline is verified as 50.8% reduction relative to EigenScore, with dataset/model averaging defined. The new detector tables reproduce Appendix B values in a reference panel and retain the separate matched comparison.

Final editorial review: simplified wording, verified response locations, checked revision markup and rendered PDFs. See `../docs/OVERLEAF_UPLOAD.md` for the exact upload list.
