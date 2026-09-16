# Results, provenance, recovery and Git

## Working layout

`HIDE_RESULTS_ROOT` defaults to `outputs/final`. Each experiment writes:

```text
outputs/final/
  qa/llama3-8b_nq_open.jsonl
  qa/llama3-8b_nq_open.manifest.json
  qa/llama3-8b_nq_open.runtime.json
  qa/llama3-8b_nq_open.sources/...
  qa/llama3-8b_nq_open.lock
  comparison/...
  ablations/...
  temperature-0.3/...
  timing/...
  pilot/...
  logs/PROFILE_MODEL_DATASET.log
  logs/PROFILE_MODEL_DATASET.packages.txt
  analysis/run_overview.csv
  analysis/analysis_manifest.json
  analysis/sources/...
  analysis/qa/metrics.csv
  analysis/qa/mechanistic/...
  analysis/ablations/ablations/...
  analysis/timing/timing_summary.csv
```

One JSONL record corresponds to one target example (or example/repeat for timing). All completed records are flushed and fsynced immediately, outside measured latency. A process lock prevents two runners from writing the same output concurrently. The launcher logs the resolved command and installed package list.

## Schema version 2

| Record group | Saved fields |
|---|---|
| Identity | schema version, dataset/model, example ID, repeat, per-example seed, decoding, selected layer/width |
| Content | full prompt/question/first reference, available aliases, input IDs, generated IDs and decoded answer |
| HIDE | score, keyword lists, selected token strings, realized matched-token count |
| Mechanism | Omega and unprojected Delta on matching layer/query positions |
| Labels | sentence similarity, ROUGE-L, first-reference EM, binary similarity correctness, question-generation similarity |
| Single-output baselines | actual-token log probabilities and log-normalizers, MNLL, energy and confidence orientations |
| Comparison | all five sampled texts/IDs/log probabilities, sample stop/cap flags, LN-Entropy, lexical similarity, centered N×N Gram matrix, width/alpha and both EigenScore conventions |
| Ablation | named variant, family/parameter, score, count, status and any undefined/error reason |
| Timing | base/capture/scoring/total/overhead seconds, repeat, generated answer/IDs and token counts |
| Diagnostics | success/error status, error message, generation-cap/no-state flags, elapsed time, peak allocated/reserved GPU memory |

Manifests save the full arguments, selected IDs, prompt/reference cohort hash, dataset fingerprint/count, checkpoint configuration, checkpoint file identity, software versions, source hashes and expected ablation names. Exact package/config source snapshots accompany each run. Runtime metadata saves actual generation config, layer count, parameter count, width and device memory.

Checkpoint config/tokenizer files are content-hashed. Weight files are inventoried by name, size and modification time by default; **this is not a cryptographic weight-content guarantee**. For strict full-weight hashing, use `--hash-weights` with the resolved `python -m hide.runner ...` command before starting a new run. Weight hashing reads all shards and can take time. Model weights themselves are never exported.

Full hidden-state and attention tensors are not persisted. Saved scores, token-level probability summaries and small Gram matrices support all supplied post-run analyses, including recomputing the two EigenScore conventions. Arbitrary new layers/kernels require new inference unless already included in the ablation run.

## Resume and failure recovery

Rerun the identical command after interruption. Successful `(id, repeat)` pairs are skipped, and each stochastic example has its own stable seed. Finished matching runs return before model loading. Changed manifests, duplicate records, unexpected IDs and errors stop automatic resume.

A truncated final line after process/storage failure can be repaired with:

```bash
python -m hide.recover outputs/final/qa/llama3-8b_nq_open.jsonl
```

A trailing error can be retried only explicitly, after investigating its cause:

```bash
python -m hide.recover outputs/final/qa/llama3-8b_nq_open.jsonl --retry-error
```

Recovery preserves the original bytes and metadata under `.history/` and records a checksum and reason. It removes only a damaged/trailing error record. It refuses earlier corruption. Backups are included in exported results. Changing the scientific configuration or source requires a new output root; recovery does not authorize mixing protocols or discarding difficult examples.

## Analysis

```bash
bash scripts/analyze.sh
```

The command checks completeness of the runs present, then analyzes each profile separately. It does **not** assert that every planned suite exists; final export with `--suite` performs that check. Pilots appear only in the operational overview. Analyses preserve finite zero scores and use a common finite cohort for each detector comparison. Exclusions, undefined statistics and ablation failures are explicit. Analysis source snapshots and input hashes are saved.

For an early look after stopping writers:

```bash
bash scripts/analyze.sh --allow-partial
```

Read counts and exclusions before interpreting any statistic. An ablation's primary comparator is recomputed on the same eligible cohort. Main-table intervals are paired, pointwise example bootstrap intervals, not multiplicity-adjusted claims. Scientific figures use the saved scores; no plotted resampling by score/class is performed.

## Verified Git snapshots

Working outputs are ignored by Git. Create a new named snapshot after writers have finished:

```bash
bash scripts/export.sh final --suite review
```

Require every suite that you claim to have completed:

```bash
bash scripts/export.sh complete --suite review --suite consistency --suite ablations --suite decoding
```

Or create an explicitly incomplete checkpoint:

```bash
bash scripts/export.sh checkpoint_01 --suite review --allow-incomplete
```

The exporter obtains run locks, checks counts/duplicates/errors/settings/source snapshots and matching code/cohorts within a requested suite, copies all files losslessly, and verifies reconstructed SHA-256 hashes before publishing the snapshot directory. JSONL/logs and large files are compressed in chunks of at most 16 MiB uncompressed. Chunks may split a JSON line; restore before parsing. `INDEX.json` maps original files to chunks, stores hashes and records requested suites and completeness. Summaries remain readable under `results/NAME/files/analysis/`.

The complete flag describes the exported runs and requested suites, not scientific acceptance or correctness of every historical paper result. Known mathematical undefined ablation cases are counted rather than fabricated. Unexpected ablation errors prevent complete export. Ordinary alias/one-token exclusions must still be discussed where material.

All names are new snapshots; existing ones are never overwritten. This bounds individual Git file sizes, **not total repository growth**. Full main-table sampling can produce substantial data. Keep only useful checkpoint/final snapshots in Git rather than committing every intermediate append. The raw local directory is an additional copy.

```bash
git status --short
git add -A
git diff --cached --stat
git commit -m "Add audited HIDE code and verified experiment snapshot"
git push origin HEAD
```

Nothing in the runner/exporter pushes automatically. Give the accessible repository branch/commit and snapshot path in this chat for analysis afterward.

## Restore after pulling

```bash
python -m hide.export_results --verify results/final
python -m hide.export_results --restore results/final --output outputs/restored
export HIDE_RESULTS_ROOT="$PWD/outputs/restored"
bash scripts/analyze.sh
```

Restore is byte-for-byte and refuses an existing output directory. No weights/GPU are required for saved-score analysis. Inference snapshots cannot be resumed across changed source/paths by bypassing manifests; analysis and resumption are separate operations.

## Earlier provisional runs

This is schema/protocol v2. If you already ran the earlier commands, keep those raw outputs and their original snapshots intact. They can be exported/restored and analyzed separately, but do not merge them into v2 files or claim that they include newly added raw-logit, ablation or five-sample baseline fields. A fresh output root is required for the new protocol.

## Distributed parts

See [PARALLEL_RUNS.md](PARALLEL_RUNS.md). Raw queues live under `outputs/`, with a checksummed plan, independent task folders, worker session logs and per-execution runtime records. A row's `execution_id` references its saved runtime record. Cooperative pauses retain ordinary successful rows; explicit repair preserves a damaged trailing record before retry. Source/manifest checks remain enforced. Merge verifies every planned part exactly once and retains original bytes and metadata under `provenance/`; source queues and merged roots must be separate. Export detects unfinished work plans and excludes active queue writers using a shared/exclusive lock. Merged results use the same analysis and Git snapshot format as single-process runs.
