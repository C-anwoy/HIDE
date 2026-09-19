# Numerical and protocol audit for the 19 September revision

## New measurements

- Verified all hashes in the uploaded fixed-subset bundle.
- Checked all 80 required parts: 16,000 records, exact selected IDs, no duplicates,
  complete manifests/runtime/source snapshots, unchanged per-cohort protocol.
- Verified identical IDs, prompts, and first references across models for each dataset.
- Computed scores from full-precision JSON values; no outcome-based exclusions.
- Used the same bootstrap draws across detectors, with continuous and binary PCC
  targets kept distinct. Tables scale AUC/PCC by 100; mechanistic PCC is unscaled.
- New tables and Figure 2 share the corrected cohort; Figure 4 and Section 6.4
  share the earlier paired timing CSV. Timing was not rerun.
- Overhead arithmetic uses unrounded data. A last-decimal difference between
  subtracting rounded displayed means and the displayed overhead is expected
  and explained in the timing caption.

## Historical table arithmetic

Per-model summary cells match the mean of the four displayed dataset cells to
within 0.011 percentage points, consistent with two-decimal display rounding.
The historical underlying per-example outputs are not available for validating
every old table, so this is an arithmetic check, not a reproduction claim.

Using AUC_s, the best uncertainty comparator in each cell is the larger of
Perplexity and Energy; the best multi-pass comparator is the largest of LN-Entropy,
Lexical Similarity, and Eigenscore. Mean relative gains average `100*(HIDE/best-1)`
over the relevant model/dataset cells, with all cells retained.

| Quantity | From displayed cells | Revision |
|---|---:|---|
| Overall mean relative gain over uncertainty | 28.5348% | approximately 29%, historical comparator scope |
| Faithfulness mean relative gain over uncertainty | 33.7351% | 33.7% retained |
| SQuAD relative gain over uncertainty | 47.3769% | 47.4% retained |
| RACE relative gain over uncertainty | 20.0932% | 20.1% retained |
| Faithfulness mean relative gain over best multi-pass | 4.2262% | replace 6.5% with 4.2% |
| Factuality mean relative gain over uncertainty | 23.3346% | replace 22.9% with 23.3% |
| Factuality wins over best uncertainty | 9/12 | replace 10/12 with 9/12 |
| Overall gain over best multi-pass | 1.8802% | remove the imprecise 3% headline |

The old NQ probe narrative also contradicted its table: Gemma-2-9B HIDE 79.08 is
below probing 85.97; Gemma-2-9B-Instruct HIDE 85.19 is only slightly above probing
84.75. The updated text reports the mixed result rather than uniform superiority.

## Additional consistency corrections

- The original main comparison has five named baseline methods, not six.
- The contribution list has three populated items and no empty bullet.
- Corrected the threshold-direction prose: for a below-threshold hallucination
  rule, increasing the threshold flags more cases; decreasing it flags fewer.
- Do not treat the unprojected input-state norm as the actual value/output-projected
  residual update, or QCBW observations as evidence of causal circuits.
- Retained historical detection/ablation cells and Figure 3 are explicitly separated
  from new corrected cohorts. The stopping limitation is stated in Limitations.
- The supplied timing sheet has now been audited: the six-model mean reduction is 50.804151% versus EigenScore, 50.839130% versus lexical similarity, and 53.230222% versus LN-Entropy. The average across these comparators is 51.624501%. The paper consistently headlines 50.8% versus EigenScore and defines the aggregation. Figure 3 is regenerated from the sheet; Figure 4 remains the separate paired-overhead suite.

## Final review

The corrected evidence does not support claiming HIDE beats attention on factuality
tasks or that nonlinear HSIC is necessary. The response and manuscript disclose this.
These files address the comments with available data; they do not guarantee journal
acceptance or retroactively validate the historical outputs.

The historical threshold table SQuAD column mean is corrected from 0.11 to 0.12 (the displayed six thresholds average 0.1167). Per-model thresholds are unchanged; their applicability to corrected stopping is not established.

## Timing sheet and presentation follow-up

All 46 populated mean cells in the supplied timing CSV agree with recalculation within 1e-8 seconds. Llama-3-8B/SQuAD has base 1.616885471 s and total 1.844477796 s: overhead 0.227592325 s (14.07597069%). The total 1.710671223 s is for RACE, with base 1.5021086 s. No cross-dataset or cross-platform subtraction is used.

The new detector tables reproduce Appendix B values in Panel A and retain the matched comparison in Panel B, with explicit noncomparability across panels. All Appendix B numeric rows remain unchanged. Revision text, new tables, changed captions and equations are magenta.
