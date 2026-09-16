# What to rerun, what to add, and how

All commands run from the repository root after installation and sourcing `configs/env.local.sh`. There is one supported package and one storage workflow. No provisional parallel implementation is required.

## Decide which results you are claiming

| Work | Why | Required action |
|---|---|---|
| New prompt-attention/norm baseline table, Figure 2 correlations | Explicit editor/reviewer requests; original layer alignment and plotted cohort were inconsistent | Full QA study on Llama-3-8B/Gemma-2-9B across all four datasets |
| Section 6.4 / Figure 4 latency and 27B scaling | Mutually inconsistent numbers and insufficient scale evidence | New synchronized timing suite; use one summary throughout |
| LN-Entropy rows in original main tables | Archived implementation uses argmax probability rather than sampled-token probability | Recompute the comparison on common fresh generations if retaining the printed definition |
| EigenScore rows in original main tables | Printed and coded covariance normalization differ; terminal-state indexing is fragile | Recompute and explicitly name the convention; do not silently relabel old values |
| Other main-table HIDE/baseline cells | No raw outputs; new BF16/error-handling/probability protocol can change generations/labels | Use the full six-model comparison to rebuild a coherent table; never mix fresh HIDE with old baseline cells as if paired |
| Figures 5–10 / cosine comparison | Estimator/SVD conventions and methodological claims need checking | Four full ablation jobs reuse one generation per example for all variants |
| Decoding Table 8 | Old sampler defaults and sampling resume were implicit | Run the six explicit stochastic configurations if replacing/retaining the exact robustness claim under the new protocol |
| Captions, typos, threshold direction and causal wording | Textual inconsistency/overstatement | Edit and compile; no GPU run needed |
| Probe and HaluEval dialogue results | Probe checkpoints/training split provenance and matched dialogue reproduction are not supplied | Recover artifacts and validate separately before claiming these experiments were reproduced |
| Historical 51% / Figure 3 multipass latency | Source aggregation missing; no validated batched latency implementation supplied | Recover the original evidence or remove/reframe this numerical claim; sequential comparison times cannot replace it |

**The reviewer suite alone does not validate every historical table.** Conversely, a full main-table comparison already contains the greedy HIDE/Omega/Delta results needed for the new baseline study. If running all six-model comparisons, skip the separate QA runs and run only the timing suite afterward. Export validation accepts those complete comparisons for the corresponding QA requirements.

There are five training-free baselines in the supplied main comparison (plus HIDE), although some prose says six baselines. Correct this count or explicitly identify the separate supervised probe.

## Stage 1: prepare and pilot

```bash
bash scripts/setup.sh
source .venv/bin/activate
cp configs/env.example.sh configs/env.local.sh
# Edit paths, then:
source configs/env.local.sh
bash scripts/check.sh
python -m hide.prepare_data --data-root "$HIDE_DATA_ROOT"
bash scripts/run_pilots.sh
python -m hide.summarize_runs --source "$HIDE_RESULTS_ROOT"
```

Read `analysis/run_overview.csv`, pilot JSONL and logs. Verify count, generated text/reference format, cap/no-state frequency, errors and peak memory. Throughput excludes model loading and must be measured locally. One A100 80GB does not guarantee a full replication fits a two-day deadline.

## Stage 2A: reviewer additions — eight full QA jobs

Each command stores an independently resumable run. All detectors are computed on each same answer.

```bash
bash scripts/run.sh qa llama3-8b nq_open
bash scripts/run.sh qa llama3-8b triviaqa
bash scripts/run.sh qa llama3-8b SQuAD
bash scripts/run.sh qa llama3-8b race
bash scripts/run.sh qa gemma-2-9b nq_open
bash scripts/run.sh qa gemma-2-9b triviaqa
bash scripts/run.sh qa gemma-2-9b SQuAD
bash scripts/run.sh qa gemma-2-9b race
```

This is 45,992 target generations. Nothing is limited to 1,000 examples. `bash scripts/run_review.sh` runs these eight jobs followed by the eight timing jobs below. Do not run the combined script alongside the individual commands.

## Stage 2B: main-table consistency rebuild — six models × four datasets

Use `comparison` in place of `qa` when rebuilding the full original comparison. It includes a greedy target and five stochastic samples per example, all five implemented training-free comparators, HIDE and the new attention/norm baselines.

```bash
bash scripts/run_consistency.sh --dry-run
bash scripts/run_consistency.sh
```

Separate model/dataset commands:

```bash
bash scripts/run.sh comparison llama3-3b SQuAD
bash scripts/run.sh comparison llama3-3b race
bash scripts/run.sh comparison llama3-3b nq_open
bash scripts/run.sh comparison llama3-3b triviaqa
bash scripts/run.sh comparison llama3-3b-instruct SQuAD
bash scripts/run.sh comparison llama3-3b-instruct race
bash scripts/run.sh comparison llama3-3b-instruct nq_open
bash scripts/run.sh comparison llama3-3b-instruct triviaqa
bash scripts/run.sh comparison llama3-8b SQuAD
bash scripts/run.sh comparison llama3-8b race
bash scripts/run.sh comparison llama3-8b nq_open
bash scripts/run.sh comparison llama3-8b triviaqa
bash scripts/run.sh comparison llama3-8b-instruct SQuAD
bash scripts/run.sh comparison llama3-8b-instruct race
bash scripts/run.sh comparison llama3-8b-instruct nq_open
bash scripts/run.sh comparison llama3-8b-instruct triviaqa
bash scripts/run.sh comparison gemma-2-9b SQuAD
bash scripts/run.sh comparison gemma-2-9b race
bash scripts/run.sh comparison gemma-2-9b nq_open
bash scripts/run.sh comparison gemma-2-9b triviaqa
bash scripts/run.sh comparison gemma-2-9b-instruct SQuAD
bash scripts/run.sh comparison gemma-2-9b-instruct race
bash scripts/run.sh comparison gemma-2-9b-instruct nq_open
bash scripts/run.sh comparison gemma-2-9b-instruct triviaqa
```

This is 137,976 target examples and 689,880 additional sampled sequences. It is much larger than the minor-review study. Do not promise a two-day finish without a pilot. If the deadline cannot accommodate necessary corrections, an extension or narrower supported claims is preferable to silently preserving a known definition mismatch.

The saved comparisons provide consistent per-example labels and paired evaluation. Original probe/dialogue tables and optimized multipass latency remain outside this rebuild; see the audit.

## Stage 3: ablation consistency — four full jobs

```bash
bash scripts/run.sh ablations llama3-8b SQuAD
bash scripts/run.sh ablations llama3-8b nq_open
bash scripts/run.sh ablations gemma-2-9b SQuAD
bash scripts/run.sh ablations gemma-2-9b nq_open
```

Or: `bash scripts/run_ablations.sh`.

These 19,076 generations supply layer, token-budget, kernel, gamma, centered-estimator, cosine and SVD comparisons. The main greedy rows are retained too. No separate generation is needed for each layer/kernel. Exact sweep values are declared in the saved source and METHOD.md. Unexpected variant failures are visible and prevent complete export; mathematically undefined small-count cases are separately counted.

## Stage 4: decoding consistency — 24 full jobs

```bash
bash scripts/run_decoding.sh --dry-run
bash scripts/run_decoding.sh
```

Individual commands follow this complete pattern for each of `llama3-8b` and `gemma-2-9b`, on `SQuAD` and `nq_open`:

```bash
bash scripts/run.sh temperature-0.3 llama3-8b SQuAD
bash scripts/run.sh temperature-0.6 llama3-8b SQuAD
bash scripts/run.sh temperature-0.9 llama3-8b SQuAD
bash scripts/run.sh nucleus-0.7 llama3-8b SQuAD
bash scripts/run.sh nucleus-0.8 llama3-8b SQuAD
bash scripts/run.sh nucleus-0.9 llama3-8b SQuAD
```

The dry-run prints all 24 fully resolved commands. Greedy references come from the common QA/comparison/ablation outputs, not another unrecorded run. This suite adds 114,456 generations. Do not conflate historical sampler settings with these explicit temperature-only/nucleus-only protocols.

## Stage 5: corrected timing and scaling — eight jobs

```bash
bash scripts/run.sh timing llama3-3b SQuAD
bash scripts/run.sh timing llama3-3b nq_open
bash scripts/run.sh timing llama3-8b SQuAD
bash scripts/run.sh timing llama3-8b nq_open
bash scripts/run.sh timing gemma-2-9b SQuAD
bash scripts/run.sh timing gemma-2-9b nq_open
bash scripts/run.sh timing gemma-2-27b SQuAD
bash scripts/run.sh timing gemma-2-27b nq_open
```

Or: `bash scripts/run_timing.sh`.

Use a quiet GPU. Every job has 200 seeded prompts, three repeats and ten warmups, with the same precision/backend/keyword placement. Timing alternates base/HIDE order, checks identical generated tokens, and records base/capture/scoring/total/overhead separately. The judge, attention baselines and ablations are excluded from this latency path. There is no quantization or CPU offload. Actual parameter count, width and peak memory are recorded.

## Stage 6: analysis and verified storage

```bash
bash scripts/analyze.sh
bash scripts/export.sh final --suite review
```

If you also completed the broader reruns:

```bash
bash scripts/export.sh full --suite review --suite consistency --suite ablations --suite decoding
```

Choose a new snapshot name if one exists. To share an explicitly incomplete checkpoint after stopping writers:

```bash
bash scripts/analyze.sh --allow-partial
bash scripts/export.sh checkpoint_01 --suite review --allow-incomplete
```

Analyze profiles separately. The software does not pool stochastic/greedy cohorts, select a favorable norm sign, rebalance classes, or filter low HIDE scores. Paired intervals, exclusions and count diagnostics are saved. All figures and summary CSVs are included in the result snapshot.

## Stage 7: manuscript and response

1. Use actual main-table comparison results if rebuilding those tables; use all detector scores and common labels from the same runs. Label the fresh protocol and precision.
2. Insert the new standalone-attention comparison and discuss closed-book NQ/TriviaQA, including HIDE losses or indistinguishable results.
3. Replace Figure 2 correlations using the corrected layer/token alignment and full valid cohort. Explicitly discuss Gemma's observed sign, even if it changes after regeneration.
4. Replace Section 6.4/Figure 4/response numbers from one timing CSV. Add the 27B point; distinguish parameter count, hidden width and pipeline latency.
5. Recover the historical Figure 3/51% provenance or remove that exact numerical claim. No sequential comparison timing may be presented as optimized batched inference.
6. Apply estimator/token-pairing/proxy/PCC/threshold corrections and qualify causal and universality claims. Interpret the count control honestly.
7. Make Figures 3, 5, 6 and 8 captions self-contained; verify typo corrections and the contribution list in the compiled PDF.
8. Fill every response-template placeholder using completed work, with final figure/table references. Compile with `bash scripts/build_paper.sh` and inspect changed pages.
9. Commit the code, verified result snapshot, final manuscript changes and response. Provide the branch/commit and snapshot path for subsequent analysis in this chat.

## Time allocation

Prioritize the two-model reviewer study, corrected latency, estimator/count checks and manuscript work. The six-model/five-sample rebuild plus every robustness experiment is a substantially larger replication, not a minor extra run. If you retain a result affected by an identified implementation mismatch, either verify/recompute it or qualify/remove the associated unsupported claim. Reserve several hours for writing, compilation and submission.
