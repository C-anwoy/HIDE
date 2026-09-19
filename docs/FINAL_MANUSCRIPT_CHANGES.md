# Completed manuscript changes

Main source: `paper/COLI_template.tex`. Response: `paper/response_letter.tex`.
No further GPU runs were performed for this revision.

## Section-by-section changes

1. **Abstract / introduction:** scope the approximately 29% benchmark improvement
   to the evaluated uncertainty baselines; report the new attention comparison;
   remove universal nonlinear/causal and production-speedup conclusions. Keep
   three populated contribution bullets. Use 50.8% relative to EigenScore consistently, verified from the supplied timing sheet; the all-three-comparator mean is 51.6%.
2. **Section 4.4 / Figure 2:** define the actual unprojected input-state norm proxy;
   replace both model plots with the corrected 2,000-example cohorts; explicitly
   state positive Gemma norm correlations (+0.126/+0.215 on SQuAD/NQ). Llama now
   has +0.003/+0.367, so the old negative-correlation narrative is removed. QCBW
   is an observational topicality control; no causal-circuit claim remains here.
3. **Section 5:** distinguish the original A100 SXM4 work from the new A100 PCIe
   runs. Specify fixed seed-42 sampling, 2,000 examples per dataset/model,
   corrected first-answer-line stopping, same rows/labels for all scores,
   continuous PCC targets, no exclusions, and paired bootstrap intervals.
4. **Section 6.2 / new Table 4:** add HIDE/attention/norm AUC_s and PCC_s with
   positive-label counts. Panel A reproduces the unchanged Appendix B HIDE values; Panel B keeps the same-answer comparison. Report attention's 7/8 advantage, including factuality,
   and HIDE's Gemma/RACE win. Discuss the count-control limitation.
5. **Section 6.4 / new Table 6 / replacement Figure 4:** use one timing CSV for
   every new value; show four models including Gemma-2-27B. Define base/total/
   incremental overhead and ratio-of-means percentages. Report 200 queries,
   three repeats, warmups, paired token-ID checks, and query-bootstrap intervals.
   State earlier stopping and shared-host limitations. Retain O(d) only for fixed-n
   score arithmetic; remove constant end-to-end overhead and optimized-serving
   lower-bound claims.
6. **Figure 3:** regenerate every bar from the supplied aggregate timing CSV; state the equal-weight dataset/model averaging and the EigenScore comparator. Keep the SXM4 baseline comparison distinct from the PCIe paired-overhead suite.
7. **Figures 5, 6, 8:** retain plots; captions now name models/datasets, axes/metric,
   threshold, and varied parameter. Remove causal interpretations of robustness.
8. **Appendix F:** add ROUGE-L comparison, paired AUC_s differences and intervals,
   selected-token-count control, and timing-query length summaries.
9. **Conclusions / Limitations:** explicitly discuss the stronger attention baseline,
   count-dominated score, stopping sensitivity, benchmark/new protocol separation,
   imperfect first-reference correctness proxies, and limited timing generalization.
10. **Additional consistency edits:** correct the benchmark factuality win count and
    relative gains, mixed NQ probing results, below-threshold decision-rule direction,
    named baseline count, listed typos, and appendix cross-references. Preserve all
    benchmark per-dataset detector table values. See the numerical audit for details.

## Magenta markup and evidence-based positioning

All substantive manuscript edits since the submitted source are marked with `\revisionr3{}`. The definition is `\long\def\revisionr3#1{{\color{magenta}#1}}`, which supports the digit in the requested spelling. The response identifies these markings. New references discuss Lookback Lens, attention weights versus value vectors, and FlashAttention; none is used as proof of HIDE superiority. The Appendix B floating-table label is corrected from D.1 to B.1.

## Plot replacement files

- `paper/files/figures/computation_time_plot_updated.pdf`
- `paper/files/figures/llama3-8b_Mechanistic_Flow_updated.pdf`
- `paper/files/figures/gemma-2-9b_Mechanistic_Flow_updated.pdf`
- `paper/files/figures/Scalability_Analysis_updated.pdf`

All use vector graphics, white backgrounds, sans-serif labels, and the original
muted teal/red/gold color family. Every selected example is plotted; no pruning.

## Table integration

```latex
% Section 6.2:
\input{files/tables/paired_s}
% Section 6.4:
\input{files/tables/timing_revision}
% Appendix F:
\input{files/tables/paired_r}
\input{files/tables/paired_differences}
\input{files/tables/count_control}
\input{files/tables/timing_lengths}
```

These inputs are already inserted in the updated manuscript; do not insert them
a second time. Full table definitions are supplied as separate `.tex` files.

## Validation and remaining submission actions

- All 80 selected parts verified; 16,000 paired records; no metric exclusions.
- 144 independent sklearn/scipy AUROC/PCC checks passed to 1e-12 tolerance.
- Historical table aggregation checked; revised new numeric tables derive from
  saved CSVs rather than manually chosen values.
- Compiled the manuscript and response, checked references, and visually reviewed
  all PDF pages plus enlarged updated plots/tables. No clipping, missing punctuation
  glyphs, overfull boxes, oversized floats, or undefined-reference warnings remain.
- The authors should review the qualifications and submit the revised paper,
  response letter, and **current original decision letter** together. The existing
  benchmark measurements are not newly validated by these controls. Acceptance
  cannot be guaranteed by the completed edits.
