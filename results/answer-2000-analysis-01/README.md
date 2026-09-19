# Fixed 2,000-example comparison: verified revision analysis

Source bundle: `results/answer-2000-complete-01`, uploaded in commit `8c80bc8`.
All exported-file hashes verified. All 80 specified parts are complete; all 16,000
records enter every paired metric with zero exclusions. The full parent plan may
remain incomplete; only the recorded fixed subset is the target of this analysis.

The first 2,000 positions in each seed-42 shuffled dataset were selected for the
deadline, after earlier full-cohort results had been inspected, without choosing
examples by their scores or correctness. Both models have identical per-dataset
IDs, prompts, and references. The subset is not a preregistered study. Extra
completed examples in the export are preserved but not pooled into these tables.

## Protocol and numerical precision

Greedy first-answer-line stopping, existing EOS conditions, 256-token cap, original
prompts/layers/kernel/20-token budget. All detectors use the same generations and
correctness labels. Empty outputs, capped outputs, and zero fallbacks are retained.
There are 252 cap hits, nine empty evaluated answers, and seven no-output-state
fallbacks (all seven Gemma/NQ); these categories can overlap.

Raw JSON is decoded with `json.loads` before constructing DataFrames. Scores are
not rounded before ranking. This matters because Llama HIDE scores are extremely
close to a count-only control and tiny score differences can affect ranks.
PCC-continuous, PCC-binary, and PCC-with-HIDE are separately named in `metrics.csv`.
Correctness is the AUROC-positive class. Both norm signs are saved explicitly;
the paper reports the positive raw direction without selecting by test AUC.

Intervals use 2,000 paired example bootstrap draws, seed 42. They are pointwise
and not multiplicity-adjusted. Supplementary ROUGE-L and exact-match results are
retained along with all method-level intervals and diagnostics.

## Findings

Attention has higher AUC_s in seven of eight settings; HIDE wins on Gemma/RACE.
All eight pointwise HIDE-minus-attention intervals exclude zero. The same 7/8
point-estimate pattern holds for ROUGE-L and exact-match targets. The norm proxy
also outperforms HIDE on both TriviaQA cohorts.

| Model | Dataset | Correct/2,000 | HIDE AUC_s | Attention AUC_s | Norm AUC_s |
|---|---|---:|---:|---:|---:|
| Llama-3-8B | SQuAD | 741 | 76.33 | 85.34 | 53.79 |
| Llama-3-8B | RACE | 778 | 61.66 | 64.67 | 49.97 |
| Llama-3-8B | NQ | 245 | 78.47 | 84.24 | 72.05 |
| Llama-3-8B | TriviaQA | 1153 | 57.01 | 73.68 | 64.68 |
| Gemma-2-9B | SQuAD | 679 | 75.90 | 86.72 | 54.32 |
| Gemma-2-9B | RACE | 1150 | 57.93 | 51.23 | 50.56 |
| Gemma-2-9B | NQ | 448 | 62.74 | 67.97 | 58.63 |
| Gemma-2-9B | TriviaQA | 1257 | 49.83 | 66.94 | 61.53 |

The all-ones-kernel control `(n_eff-1)/n_eff**2` (zero for no state) correlates
with HIDE at least 0.998718. Absolute AUC_s differences are at most 0.42465
percentage points. This materially limits nonlinear-dependence interpretations.
No kernel tuning, direction flipping, or outcome-based subset selection was used.

The new Figure 2 correlations are from the same corrected cohorts. In SQuAD/NQ,
Gemma's HIDE/norm PCC is +0.126/+0.215; Llama's is +0.003/+0.367. These replace
the old figure's correlations, rather than mixing different protocols.

## Timing and historical tables

`timing_summary.csv` is copied unchanged from the verified earlier paired timing
analysis. No new timing was run. It uses the earlier stopping protocol, 200 queries,
three measured pairs per query, ten warm-up pairs, and an A100 80GB PCIe.
Differences and ratios use unrounded means. The new Figure 4 uses its overhead
means and query-bootstrap intervals. Host contention and missing continuous
power/clock telemetry limit interpretation. It does not establish optimized-serving
performance or reproduce the historical 51% multi-pass comparison.

Original detection table cells are retained and explicitly labeled historical;
their scientific validity is not certified by these new runs. The original
continuation issue and changed evaluation protocol are disclosed. A separate
arithmetic audit corrects misleading summary statements without altering those
historical per-cell measurements. See `docs/SUBMISSION_NUMERICAL_AUDIT.md`.

## Reproduce (CPU only)

```bash
python -m hide.export_results --restore results/answer-2000-complete-01 --output outputs/answer-2000-restored
python scripts/analyze_submission.py
```

This writes the metrics, matched table source, and updated mechanistic/scaling
PDFs. `analysis_manifest.json` lists the exact input file hashes. The script
snapshot is retained in `analysis_source.py`. Final manuscript-specific text and
the response letter are in `paper/`; plots are in `output/pdf/` and copied into
`paper/files/figures/` for compilation.
