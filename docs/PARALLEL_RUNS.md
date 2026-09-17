# Distributing the experiments across GPUs

## What is bounded, and what is preserved

Use one worker per available GPU. A worker processes independent **200-example parts**, stops starting new work at 18 hours, requests a cooperative pause, and terminates its model subprocess at 20 hours if necessary. Completed examples are flushed and fsynced individually. Rerun the worker to finish outstanding work. A soft stop is exit code 75 from the runner, not an error or hallucination label.

**This is a runtime limit, not a guarantee that a fixed amount of work completes in 20 hours.** Output length, GPU/CPU speed, loading, memory pressure, and filesystem/network delays vary. The hard limit is a process supervisor, not a scheduler reservation; OS/driver/filesystem hangs can delay termination. Budget a 24-hour allocation around the default 20-hour supervisor limit. A hard termination can leave an incomplete final JSON line; recovery preserves its original bytes before removing only that final record. Ordinary cooperative pauses need no repair.

Parts do not reduce the evaluation set. They partition the same seeded dataset order, and preserve each example's random seed, prompt, precision, layer, decoder, HIDE score definition, labels and paired baselines. A comparison part of 200 examples includes the same 200 greedy answers and 1,000 stochastic answers. Ablation variants share the same original generation. Parts are merged in original cohort order before AUC/PCC, bootstrapping, QCBW selection or plotting; never average per-part AUCs.

| Plan | Detection parts | Timing jobs | Scope |
|---|---:|---:|---|
| `review` | 234 | 8 | New HIDE/attention/norm study plus latency/scaling |
| `full` | 1,388 | 8 | 702 comparison + 98 ablation + 588 decoding parts, plus latency/scaling |

The full plan omits duplicate QA runs: comparisons already supply reviewer measurements. It does not add missing probe/dialogue reproduction or recover the original 51% batched latency evidence; see AUDIT.md. More GPUs reduce elapsed time, not total work. Reloading checkpoints and preparing each part adds overhead. We deliberately keep one example per generation call to retain the audited method, while distributing independent work across GPUs.

## Install and configure once per machine

Use the same Git commit and package versions on every machine. Recommended environment: Python 3.11, repository-pinned torch/transformers and dtype. For an appropriate driver, install torch 2.5.1 from its CUDA 12.4 wheel index, then `python -m pip install -r requirements.txt`. A Conda environment does not need `scripts/setup.sh` (which creates a venv).

Configure `configs/env.local.sh` using `configs/env.example.sh`. Set `HIDE_MODEL_ROOT=/models` on optimus. Named checkpoints are validated locally or downloaded at startup into the user-owned `HIDE_CHECKPOINT_CACHE`; see [CHECKPOINTS.md](CHECKPOINTS.md). Full weight SHA-256 values are cached and saved, so identical files can match across servers with different timestamps. Keep paths stable when resuming an unfinished part, and keep checkpoint contents and package versions consistent across workers. Downloaded revisions are pinned; local folders are not assumed to equal those revisions without matching content. Preparation is included in the supervised worker budget but excluded from timing measurements. GPU IDs are selected independently per worker.

For each GPU, create a tmux session and initialize its shell explicitly. Example for the user's **physical GPU 1 on optimus**:

```bash
tmux new-session -s hide-gpu1
```

Inside tmux:

```bash
cd ~/HIDE
source /home/anwoy/miniconda3/etc/profile.d/conda.sh
conda activate hide-paper
source configs/env.local.sh
export CUDA_VISIBLE_DEVICES=GPU-4957c764-5f46-b8cb-d688-a4a667c9e288
export HIDE_DEVICE=cuda:0
CUDA_VISIBLE_DEVICES="" bash scripts/check.sh
python -m hide.prepare_data --data-root "$HIDE_DATA_ROOT"
```

Select another GPU using its own UUID from `nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv`. Restrict visibility before starting Python. The selected physical GPU is logical `cuda:0`. These commands do not reserve the GPU or stop other users' jobs. Ensure enough free VRAM for the selected checkpoint, captured hidden states/attention/logits and correctness model. Do not silently change dtype, quantize or truncate inputs to make a job fit. Prefer A100s for consistent numerical behavior; accuracy results on different GPU types retain per-part hardware provenance, but are not guaranteed bit-identical across hardware.

Detach: Ctrl+B then D. Reattach on the same host: `tmux attach-session -t hide-gpu1`. New panes need their environment initialized again.

## Create one plan

For the entire supported rerun:

```bash
bash scripts/parts.sh plan --suite full --part-size 200 --output outputs/parts/plan.json
```

For only reviewer additions, use `--suite review` instead. Choose one; do not run both and duplicate the greedy QA work. Plans for `consistency`, `ablations`, `decoding` and `timing` are also available. `plan.tsv` lists every task, model, dataset and range. The JSON includes a portable source fingerprint and checksum. Do not edit the plan or change code after starting it.

## One GPU overnight: reviewer priority

Use this when starting a **review** plan on one exclusive GPU. It does not consume a full plan. If you already started a different plan, retain it and use its regular workers instead of creating duplicate scientific work.

The priority launcher automatically runs:

1. NQ, both Llama-3-8B and Gemma-2-9B: paired HIDE/attention mass/update norm and closed-book evidence.
2. SQuAD, both models: open-book comparison and model dependence of the norm relationship.
3. All eight latency/scaling jobs: 3B, 8B, 9B and base 27B on SQuAD/NQ, on the same physical GPU.
4. TriviaQA, both models: extend the factuality result.
5. RACE, both models: complete the four-dataset baseline table.

Within each accuracy dataset, models alternate after each 200-example part. These are full cohorts split into resumable units, not 200-example final evaluations. NQ has 3,610 examples per model; SQuAD 5,928; TriviaQA 9,960; RACE 3,498. Timing uses the declared 200-query protocol. The launcher requests a pause at 18 hours and passes the remaining 20-hour deadline to every successive worker; budgets do not restart per part. Completion of this entire schedule within one night is not guaranteed.

Initialize tmux, Conda and GPU selection as above. Prepare dependencies/data and validate/download the checkpoints **before leaving**, so authentication and missing-file errors appear while you are available:

```bash
bash scripts/prepare_models.sh --models llama3-3b llama3-8b gemma-2-9b gemma-2-27b keyword judge
python -m hide.prepare_data --data-root "$HIDE_DATA_ROOT"
```

Create the plan once (an existing path is intentionally refused):

```bash
bash scripts/parts.sh plan --suite review --part-size 200 --output outputs/review/plan.json
bash scripts/run_priority.sh --dry-run
```

First validate two real parts while you are present. They remain part of the final NQ results:

```bash
bash scripts/run_priority.sh --max-parts 2
bash scripts/parts.sh status --plan outputs/review/plan.json --queues outputs/review
```

On a new queue this should show 2 complete parts, 0 failed. Then start the overnight command inside tmux:

```bash
set -o pipefail
bash scripts/run_priority.sh --hours 18 --hard-hours 20 2>&1 | tee -a outputs/review/overnight.log
```

Detach with Ctrl+B then D. Closing SSH or the laptop does not stop the server's tmux processes. Server shutdown, scheduler eviction or a real experiment error can still stop work. No unattended scientific-error retry is performed: a failed part stops the launcher, with details in the task log and a nonzero exit. Inspect and repair it explicitly using the retry instructions below. Downloads, disk space and both model smoke parts should be checked before leaving. Run only this launcher on its GPU and queue while using this one-GPU schedule; use the regular distributed workflow when assigning extra servers.

The launcher automatically advances through the stages as they finish, and skips complete parts on restart. To inspect in the morning:

When an otherwise usable accuracy GPU is shared, add `--kind detection` to the priority launcher to exclude timing and continue the baseline datasets. This does not ensure sufficient free VRAM. Once exclusive access is available, run the timing worker separately on one fixed GPU. `--kind timing` is also supported by the priority launcher.

```bash
tail -n 60 outputs/review/overnight.log
bash scripts/parts.sh status --plan outputs/review/plan.json --queues outputs/review
```

Resume using the same overnight command, on the same GPU once timing has begun. Timing jobs can also be explicitly prioritized after stopping the launcher:

```bash
bash scripts/parts.sh work --plan outputs/review/plan.json --queue outputs/review --kind timing --hours 18 --hard-hours 20
```

All records, metadata, individual task logs and worker sessions are saved under `outputs/review`. After the launcher has stopped, create a uniquely named Git-ready snapshot even if the schedule is unfinished:

```bash
python -m hide.export_results --source outputs/review --output results/optimus-review-night1 --allow-incomplete
```

Use the analysis/export instructions below to push snapshots and merge complete cohorts. Never treat an unfinished queue or the two smoke parts as the final baseline table.

## Shared filesystem: dynamic assignment (preferred)

The pinned symbol-answer correction is the sole audited source-compatibility exception below. For other changes, keep the original strict same-code requirement.

All workers use the **same queue directory and the same plan**, on a filesystem supporting POSIX advisory locks. They claim different tasks automatically. Adding a new GPU requires no repartitioning, and returning a GPU leaves its completed parts available. Do not synchronize separate live directories with rsync and treat them as a shared queue.

First, test one real part on GPU 1:

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --max-parts 1
```

Then run a bounded worker:

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --hours 18 --hard-hours 20
```

On each additional GPU, in its own initialized tmux session with its own UUID, run **the same worker command**. For a shared mount on another machine, supply that mount's absolute plan/queue paths. One queue can be reached through different mount paths for complete parts; resume a partial part on its original checkout/environment/path.

Workers default to accuracy jobs; they never consume timing jobs. Each worker prints the current task and log location. Follow the printed log with `tail -f PATH`. Completed tasks are skipped. A real error stops the worker and marks that task failed; other workers skip failed tasks instead of retrying endlessly. Finished workers may exit when the only remaining work is already claimed by others; use status, not the worker exit code, to determine full-plan completion.

Optional restrictions for a GPU that should run only certain models or profiles:

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --models llama3-3b llama3-3b-instruct
```

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --profiles comparison --models llama3-8b gemma-2-9b
```

Run one exact part (the ID is in `plan.tsv`):

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --task comparison__llama3-8b__nq_open__00000-00200
```

## Separate machines without shared storage: fixed assignments

Copy the **same plan file** to every checkout. Pick a fixed assignment count in advance; four is shown below. The indexes select disjoint subsets of detection parts. One GPU can process multiple assignments sequentially if fewer GPUs are available. Do not change `--workers` mid-plan: that redistributes tasks and can create duplicate copies across queues.

Machine/assignment 0:

```bash
bash scripts/parts.sh work --plan configs/full-plan.json --queue outputs/worker0 --workers 4 --worker-index 0
```

Assignment 1:

```bash
bash scripts/parts.sh work --plan configs/full-plan.json --queue outputs/worker1 --workers 4 --worker-index 1
```

Assignment 2:

```bash
bash scripts/parts.sh work --plan configs/full-plan.json --queue outputs/worker2 --workers 4 --worker-index 2
```

Assignment 3:

```bash
bash scripts/parts.sh work --plan configs/full-plan.json --queue outputs/worker3 --workers 4 --worker-index 3
```

The copy can be created on the planning machine with `cp outputs/parts/plan.json configs/full-plan.json` and transferred unchanged. Queue output paths are controlled by `--queue`; `HIDE_RESULTS_ROOT` is used only for the original unpartitioned scripts and later analysis/export.

## Timing: one exclusive physical GPU

Run all eight timing/scaling jobs on one exclusive A100 80GB, after its accuracy worker stops. Each job retains the original 200 queries × 3 repeats, paired base/HIDE calls, ten warmups and alternating order. Timing jobs are intentionally not divided into smaller query parts, because repeated checkpoint loading/warmups would change the timing protocol.

```bash
bash scripts/parts.sh work --plan outputs/parts/plan.json --queue outputs/parts --kind timing --hours 18 --hard-hours 20
```

For separate queues, use the common plan and `--queue outputs/timing`. The worker requires an explicit GPU UUID and checks that no compute process occupies it before each job. It pins host, GPU UUID, driver and power limit in `timing_device.json`, rejects concurrent timing workers on the same queue, and records runtime identity. This is a preflight check, not a reservation; arrange exclusive access for the entire run. Keep power limits and CPU environment consistent. Resume on the same GPU. Do not merge scaling measurements from different physical GPUs.

## Inspect, resume, and repair

```bash
bash scripts/parts.sh status --plan outputs/parts/plan.json --queues outputs/parts
```

Normal 18-hour pause: rerun the exact worker command. Completed examples are skipped with matching manifest checks. Every invocation has its own session record, and each completed example points to a saved execution record identifying its runtime/GPU.

After a real error or hard termination, inspect the printed log first and fix its cause. Stop workers on that queue, then explicitly permit repair/retry of the affected task:

If the log contains only `Input/Output keyword extraction failed`, the current runner has discarded the underlying exception chain when serializing the error. Before retrying, replay the single failed detection example with diagnostic logging:

```bash
python scripts/diagnose_failure.py --run outputs/review/runs/qa__gemma-2-9b__nq_open__03400-03600/qa/gemma-2-9b_nq_open.jsonl
```

Use the actual failed JSONL path. The command reads its final error and manifest, identifies the original cohort position, and uses the recorded model paths, seed, dtype and generation settings. It requires matching scientific source, writes only a new `outputs/diagnostics/<timestamp>/` folder, and does not clear the failed marker or edit the original records. `--dry-run` previews the selected example without loading models. A reproduced exception is expected to exit nonzero; share `keyword_traceback.txt` and `keyword_inputs.json` from the printed diagnostic directory. If failure occurs earlier, share `runner_traceback.txt`. The replay is excluded from the paper results. Resolve the actual cause before using the retry command below; do not substitute a zero score or skip the example.

```bash
bash scripts/parts.sh retry --queue outputs/parts --task comparison__llama3-8b__nq_open__00000-00200
```

This preserves failure history and can remove only a trailing error/damaged record; it rejects corruption earlier in the file. It does not bypass protocol/source/version checks. A resumed partial part must retain matching paths, code, packages, checkpoint identity and GPU model. Complete parts from different machines can be merged using portable identities.

## Collect and merge

Stop workers before copying/exporting/merging. Keep each separate queue in a separate directory. For example, collect `worker0`, `worker1`, `worker2`, `worker3` and `timing` under one machine's `outputs/`. Never concatenate JSONL files manually or flatten the directories.

Shared queue:

```bash
bash scripts/parts.sh merge --plan outputs/parts/plan.json --queues outputs/parts --output outputs/merged
```

Separate queues:

```bash
bash scripts/parts.sh merge --plan configs/full-plan.json --queues outputs/worker0 outputs/worker1 outputs/worker2 outputs/worker3 outputs/timing --output outputs/merged
```

A merge requires every selected task exactly once, complete successful records, exact parent cohort coverage/order, matching scientific arguments/packages/checkpoint inventories and source snapshots. It rejects duplicate copies even if they look identical, missing parts and mixed timing platforms. It preserves original part JSONL bytes as `.jsonl.bak`, plus manifests, runtime records, source snapshots, recovery histories, logs and worker sessions under `provenance/`. The canonical merged JSONL files appear once under their usual profile directories.

To analyze completed detection work before timing finishes, add `--kind detection` and choose a different new output directory. This still requires **all detection parts in that plan**; it is not permission to silently omit unfinished experiments. Merging is atomic and never overwrites an existing result root.

## Analyze and push results

For the full plan after the final merge:

```bash
export HIDE_RESULTS_ROOT="$PWD/outputs/merged"
bash scripts/analyze.sh
bash scripts/export.sh full --suite review --suite consistency --suite ablations --suite decoding
```

For the reviewer-only plan, export with `bash scripts/export.sh review --suite review` instead. Primary analyses and confidence intervals use the full merged cohorts, not per-part summaries.

For a transfer/checkpoint before the whole plan finishes, stop that queue's workers and export it directly:

```bash
python -m hide.export_results --source outputs/worker0 --output results/worker0-checkpoint-01 --allow-incomplete
```

This includes every raw file and explicitly marks missing plan parts. Choose a new snapshot name each time. After a Git pull, restore it to a fresh queue directory with `python -m hide.export_results --restore results/worker0-checkpoint-01 --output outputs/worker0-restored`. Complete restored parts can be merged; resume partial parts in their original environment/path.

```bash
git add results/
git diff --cached --stat
git commit -m "Save verified HIDE experiment results"
git push origin main
```

Do not update source code from another commit while any planned experiment is running. Keep all parts on the plan's code version; analysis code updates should be audited separately after collection.

## Upgrade a queue after the symbol-answer fix

The confirmed Gemma NQ failure was a valid `*` answer with no word candidates. The core now routes the confirmed empty-vocabulary case into its existing first-token fallback. This changes source fingerprints, so **do not edit the old plan or simply clear its failed marker**.

With all workers on the old queue stopped, pull the fix and create a new queue:

```bash
git pull --ff-only origin main
python scripts/migrate_keyword_fallback.py --source outputs/review --output outputs/review-fixed
```

The upgrade accepts only the pinned pre-fix source inventory and audited corrected core. It validates and copies complete detection parts with unchanged JSONL, manifests and source snapshots. It preserves original partial/error records in `previous_incomplete_parts/` (using `.jsonl.bak` so they are not evaluated twice), and copies worker sessions plus the previous overnight log. The old queue is retained. Incomplete parts restart from their beginning under the new source; all timing parts must be rerun. The reported user queue should retain 35 complete parts and restart the one failed 200-example part. Its first 160 successful answers are archived; only the fresh 200-example rerun enters final analysis.

Test the repaired part first:

```bash
bash scripts/parts.sh work --plan outputs/review-fixed/plan.json --queue outputs/review-fixed \
  --task qa__gemma-2-9b__nq_open__03400-03600
bash scripts/parts.sh status --plan outputs/review-fixed/plan.json --queues outputs/review-fixed
```

For that reported queue, success gives 36 complete parts and no failed parts. Continue the reviewer baseline datasets in tmux:

```bash
set -o pipefail
bash scripts/run_priority.sh --plan outputs/review-fixed/plan.json --queue outputs/review-fixed \
  --kind detection --hours 18 --hard-hours 20 2>&1 | tee -a outputs/review-fixed/overnight.log
```

Remove `--kind detection` to include timing in the usual priority order only when the selected GPU is exclusive. Otherwise run timing later on an exclusive GPU:

```bash
bash scripts/parts.sh work --plan outputs/review-fixed/plan.json --queue outputs/review-fixed \
  --kind timing --hours 18 --hard-hours 20
```

Use `outputs/review-fixed` and its plan for all subsequent status/worker/merge/export commands. Merge **only the upgraded queue**, not both old and new (that would duplicate the 35 retained parts):

```bash
bash scripts/parts.sh merge --plan outputs/review-fixed/plan.json \
  --queues outputs/review-fixed --output outputs/review-merged
```

The merge verifies the pinned compatibility exception and preserves generation sources per part, along with the migration record and archived failed output. A merged manifest's top-level source inventory is explicitly the merge implementation, not a claim that old rows were generated with new code. New timing and incomplete-part reruns use the corrected version throughout.

To push a snapshot before the whole queue finishes, stop its workers first and choose a fresh export name:

```bash
python -m hide.export_results --source outputs/review-fixed \
  --output results/optimus-review-fixed-01 --allow-incomplete
```
