# Review experiment analysis — 18 September 2026

## Status

Raw snapshot: `results/optimus-review-complete-01`, Git commit `5baab47`. All reconstructed file hashes passed verification. The strict merger accepted all 242 planned parts, including the explicitly pinned symbol-fallback compatibility policy for earlier successful detection parts. All 16 merged runs passed completeness checks. No missing, duplicate or error records were reported in the active runs.

- Detection: 45,992 generations, two models × four full dataset cohorts.
- Timing: 4,800 paired measurements, four models × two datasets × 200 queries × three repeats. Each paired measurement contains a base and a HIDE generation.
- Statistical analysis: 2,000 paired example bootstrap draws for detection; timing repeats averaged within query before 2,000 query bootstrap draws. Intervals are pointwise, not adjusted for multiple comparisons.
- This validates the transferred data and recorded protocol, not every scientific claim or the historical main tables.

## Main finding: the attention baseline is competitive or stronger

AUC below uses correctness as positive and the saved sentence-similarity > 0.9 target. Values and differences are on the 0–1 scale. All paired rows are retained. PCC in the detailed files is explicitly separated into continuous and binary targets.

| Model | Dataset | N | Correct | HIDE AUC | Attention AUC | Norm AUC | HIDE − attention, 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| gemma-2-9b | SQuAD | 5928 | 2007 | 0.7541 | 0.8568 | 0.5475 | -0.1028 [-0.1171, -0.0878] |
| gemma-2-9b | nq_open | 3610 | 27 | 0.7454 | 0.9008 | 0.5719 | -0.1554 [-0.3311, +0.0241] |
| gemma-2-9b | race | 3498 | 2047 | 0.5833 | 0.5047 | 0.5116 | +0.0786 [+0.0560, +0.0995] |
| gemma-2-9b | triviaqa | 9960 | 463 | 0.7876 | 0.9570 | 0.4837 | -0.1694 [-0.2057, -0.1336] |
| llama3-8b | SQuAD | 5928 | 2178 | 0.7677 | 0.8571 | 0.5375 | -0.0894 [-0.1035, -0.0759] |
| llama3-8b | nq_open | 3610 | 432 | 0.7913 | 0.8448 | 0.7470 | -0.0535 [-0.0821, -0.0261] |
| llama3-8b | race | 3498 | 1181 | 0.6831 | 0.7044 | 0.5385 | -0.0213 [-0.0368, -0.0063] |
| llama3-8b | triviaqa | 9960 | 5037 | 0.6526 | 0.8387 | 0.8075 | -0.1861 [-0.1997, -0.1727] |

Attention mass has higher point-estimate AUC in seven of eight settings. Six differences have pointwise intervals excluding zero. HIDE outperforms attention mass on Gemma/RACE. Gemma/NQ has only 27 positive labels and its difference interval includes zero. The seven-versus-one point-estimate pattern also holds for the saved ROUGE and exact-match targets. Do not claim that nonlinear HSIC adds universal detection value beyond attention mass.

## The token-count control substantially challenges the mechanism

The saved control is c(n) = (n−1)/n² for n ≥ 1, otherwise zero, where n is the realized number of selected token occurrences. It is exactly the adapted HIDE expression evaluated on all-ones kernels. It is not raw output length.

| Model/dataset | PCC(HIDE, count control) | HIDE − control AUC | 95% CI |
|---|---:|---:|---|
| gemma-2-9b/SQuAD | 0.998816552927 | -0.000112 | [-0.001159, +0.001009] |
| gemma-2-9b/nq_open | 0.999750742331 | +0.001980 | [-0.001178, +0.006158] |
| gemma-2-9b/race | 0.998781787153 | +0.001808 | [+0.000313, +0.003410] |
| gemma-2-9b/triviaqa | 0.999860111339 | +0.000500 | [-0.000088, +0.001094] |
| llama3-8b/SQuAD | 0.999999999990 | -0.000793 | [-0.001853, +0.000212] |
| llama3-8b/nq_open | 0.999999999991 | -0.000347 | [-0.002267, +0.001634] |
| llama3-8b/race | 0.999999999983 | -0.002345 | [-0.003596, -0.001030] |
| llama3-8b/triviaqa | 0.999999999992 | +0.000515 | [-0.000792, +0.001876] |

These observations are consistent with the score being dominated by selected-token count under the configured estimator/kernel. The largest absolute AUC difference from this control is below 0.0024 across these eight settings. Correlation alone does not prove that every representation-dependent contribution is absent, but the present evidence does not establish the claimed benefit of nonlinear dependence. Do not tune the kernel on these test labels to recover a preferred conclusion. Count-bin results are saved in `qa/count_*`.

## Revised mechanistic correlations

The new protocol aligns proxy layer/token positions and uses the full cohort without score filtering or class rebalancing. Its numbers must replace, not be mixed with, the earlier Figure 2 values. Delta is an unprojected attention-weighted input-state norm, not the actual projected residual update.

| Model | Dataset | PCC(HIDE, attention) | PCC(HIDE, norm) |
|---|---|---:|---:|
| gemma-2-9b | SQuAD | 0.5553 | 0.1397 |
| gemma-2-9b | nq_open | 0.4583 | 0.1293 |
| gemma-2-9b | race | 0.2206 | -0.0197 |
| gemma-2-9b | triviaqa | 0.6625 | 0.0003 |
| llama3-8b | SQuAD | 0.5618 | 0.0127 |
| llama3-8b | nq_open | 0.6595 | 0.4193 |
| llama3-8b | race | 0.5335 | 0.1540 |
| llama3-8b | triviaqa | 0.4794 | 0.5353 |

The previous negative Llama/SQuAD update-norm correlation is not reproduced by this corrected full-cohort protocol. A universal inverse magnitude-versus-precision interpretation is unsupported. The figures are observational and do not identify causal circuits.

## Answer-boundary and label limitations

The inherited NQ/TriviaQA stopping configuration uses particular token IDs and does not reliably prevent another question from being emitted. The saved full-output correctness proxy then penalizes answers followed by unrelated continuation. The preserved code in `archive/original/dataeval/` uses the same stopping construction. This is a protocol concern, not missing data.

| Model/dataset | Outputs containing newline + Q: | Full-output exact matches | First-nonempty-line exact matches (diagnostic only) | Generation-cap hits |
|---|---:|---:|---:|---:|
| gemma-2-9b_SQuAD | 0/5928 | 1592 | 1594 | 422 |
| gemma-2-9b_nq_open | 2921/3610 | 16 | 640 | 23 |
| gemma-2-9b_race | 0/3498 | 1996 | 1996 | 0 |
| gemma-2-9b_triviaqa | 8271/9960 | 421 | 6056 | 27 |
| llama3-8b_SQuAD | 0/5928 | 1801 | 1805 | 219 |
| llama3-8b_nq_open | 679/3610 | 346 | 348 | 48 |
| llama3-8b_race | 72/3498 | 880 | 1271 | 286 |
| llama3-8b_triviaqa | 2543/9960 | 4713 | 5558 | 2089 |

The first-line diagnostic applies the existing `normalize_text` to the first nonempty generated line and first reference; it does not recompute HIDE or the proxies, and is not a corrected paired answer-only experiment. There are 13 no-output-state cases (all Gemma/NQ), retained with the declared fallback, and 3,114 detection generations hit the token cap. These effects must be disclosed.

A corrected answer-boundary experiment would require a stated stop/extraction protocol and matching detector scores. Hidden-state tensors were not saved, so truncating text and relabeling alone cannot reconstruct answer-only HIDE/proxy scores. Do not silently replace full-output labels or claim these are model factual-accuracy estimates.

## Timing and scaling

All eight jobs record the same host (`optimus.lcs2`) and GPU UUID (`4957c764-5f46-b8cb-d688-a4a667c9e288`), an A100 80GB PCIe. The queue platform record states driver 555.42.02 and power limit 300 W. The user reported exclusive GPU use but other detection workers shared the host. The logs do not continuously measure power limits, clocks or CPU contention, and earlier platform-guard interruptions were not fully diagnosed. Do not describe this as an isolated-host benchmark or certify constant hardware conditions throughout.

| Model | Dataset | Base (s) | HIDE total (s) | Overhead (s), 95% CI | Overhead (%) |
|---|---|---:|---:|---|---:|
| gemma-2-27b | SQuAD | 4.1989 | 4.3958 | 0.1969 [0.1644, 0.2365] | 4.69 |
| gemma-2-27b | nq_open | 1.1389 | 1.2195 | 0.0806 [0.0751, 0.0864] | 7.08 |
| gemma-2-9b | SQuAD | 3.3214 | 3.5927 | 0.2713 [0.2197, 0.3273] | 8.17 |
| gemma-2-9b | nq_open | 1.9465 | 2.3275 | 0.3809 [0.3222, 0.4494] | 19.57 |
| llama3-3b | SQuAD | 1.2642 | 1.4642 | 0.2000 [0.1741, 0.2279] | 15.82 |
| llama3-3b | nq_open | 0.6215 | 0.7090 | 0.0874 [0.0776, 0.1002] | 14.07 |
| llama3-8b | SQuAD | 0.9277 | 1.1587 | 0.2310 [0.1943, 0.2752] | 24.90 |
| llama3-8b | nq_open | 0.7906 | 0.8832 | 0.0925 [0.0764, 0.1106] | 11.70 |

Gemma-2-27B has 27.23B actual parameters and width 4,608. It extends parameter scale substantially, but hidden-width coverage remains 3,072–4,608. Observed timings do not justify constant absolute end-to-end overhead independent of model or input. Different models generate different output lengths; report their length distributions with comparisons. Retain the fixed-token-budget O(d) arithmetic statement with its scope.

Gemma-2-9B/NQ scoring time has median 0.081 s and 95th percentile 2.469 s; the median within-query coefficient of variation of total time is 11.1%, and its 95th percentile is 70.6%. All measurements, including negative paired overheads, are retained. Query bootstrap intervals describe uncertainty in these recorded runs and do not remove shared-host/systematic confounding. The new suite does not measure multi-pass baselines or optimized serving, so it does not re-establish the historical 51% claim.

## Submission and GPU decision

The planned data collection is complete. There is no reason to rerun everything or the existing ablations solely because the transfer has finished. There is also no basis yet for an unconditional “no further GPU work” or “ready to submit” assurance: the answer-boundary issue is material and the new results weaken the strongest mechanism/necessity claims.

Next actions: decide and document the answer-boundary protocol; retain and report all current results as recorded; narrow the mechanistic and scalability claims; include both proxy baselines, the count-control finding and uncertainty; replace efficiency values consistently; finish the editor/reviewer response and compile/inspect the revised manuscript. If a corrected answer-only protocol is adopted, run a small validated pilot before any targeted rerun, with a new plan and preserved old results. A clean-host timing rerun is necessary only if making claims that these shared-host measurements cannot support.

## Reproduction

From the repository root, using the checked-in analysis sources at the raw snapshot commit and a Python environment with the project analysis dependencies:

```bash
python -m hide.export_results --restore results/optimus-review-complete-01 --output outputs/review-restored
python -m hide.parts merge --plan outputs/review-restored/plan.json --queues outputs/review-restored --output outputs/review-merged
python -m hide.analyze --source outputs/review-merged --bootstrap 2000
```

This folder contains derived analysis only; it is not an export/restore bundle. The original raw snapshot remains unchanged. `analysis_manifest.json` records merged-input hashes and code snapshots; absolute paths describe the local analysis workspace. `DERIVED_SHA256.json` verifies the derived files. Auxiliary diagnostic definitions are stated above; their values are in `validation/diagnostics.json`.
