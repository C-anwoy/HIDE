# Corrected answer-boundary experiments

## Current selection: both detection workers on GPU1, retain existing timings

To run the corrected detection experiments only, sharing physical GPU1:

```bash
bash scripts/start_answer_tmux.sh --detection-only --llama-gpu 1 --gemma-gpu 1
```

This creates only `hide-answer-llama` and `hide-answer-gemma`, with both processes
restricted to GPU1. Each worker runs its own pilot and then its full cohorts.
There is no timing session. The separate `outputs/answer-detection` plan has
**234 detection parts** plus its eight pilot jobs in the pilot subqueue. It is
complete when main status shows 234 complete and 0 running; no timing parts are
missing from this plan. Sharing a GPU is supported for detection; each worker
still loads its own model/judge and needs sufficient combined free VRAM.

```bash
bash scripts/run_answers.sh status --root outputs/answer-detection
```

To resume an expired worker in an activated tmux session on GPU1:

```bash
export CUDA_VISIBLE_DEVICES=GPU-4957c764-5f46-b8cb-d688-a4a667c9e288
export HIDE_DEVICE=cuda:0
bash scripts/run_answers.sh detection --root outputs/answer-detection --models llama3-8b
# For the other worker, use --models gemma-2-9b in its own session.
```

After all workers finish:

```bash
python -m hide.export_results --source outputs/answer-detection --output results/answer-detection-complete-01
python -m hide.export_results --verify results/answer-detection-complete-01
git add results/answer-detection-complete-01
git commit -m "Save corrected detection-only experiments"
git push origin main
```

The original eight timing jobs stay in the earlier review snapshot. They measure
the earlier stopping protocol with the recorded shared-host limitations. Report
them separately and label the generation protocol; they are not measurements of
the corrected first-answer-line protocol. No additional timing run is scheduled.

The rest of this guide also documents the optional full detection-plus-timing
workflow. Use the detection-only command above for the current selection.

## Why these runs

The completed review snapshot is preserved at `results/optimus-review-complete-01`.
Its [analysis](../results/optimus-review-analysis-01/README.md) found that many
generations continue into a new question, materially changing full-output
correctness labels. In particular, Gemma/TriviaQA has 421 full-output exact
matches but 6,056 first-nonempty-line exact matches. The latter is a diagnostic,
not a replacement result: the saved scores describe the original full generation.

The new protocol stops at the first line break after non-whitespace answer text,
including double-newline tokens. It retains the existing prompts, period/EOS
stops, bad-word lists, first-reference labels, layers, BF16 precision, keyword
budget, gamma and HIDE formula. Leading blank lines are skipped when defining
the first nonempty answer line. The final generated token remains excluded from
HIDE/proxy state extraction, as in the original cached-generation convention.
Every row verifies that the answer boundary was not crossed before that token.

Raw generated text and token IDs remain saved. `evaluated_text` records the
trimmed first nonempty answer line used for all correctness labels. The
generation itself terminates at that boundary; this is not relabeling old
full-generation scores. A trailing token may include characters after a newline;
the pilot reports and blocks such cases for inspection. Empty answers and token
cap hits remain saved and counted. Pilot checks never require a particular
accuracy, AUC, or HIDE advantage.

The boundary criterion decodes the generated prefix at each step. It is included
identically in base and HIDE timing calls; timing therefore describes this
explicit application protocol, not generic unconstrained generation latency.

## Exact scope and priority

1. **Pilots:** 25 examples per dataset for each of Llama-3-8B and Gemma-2-9B:
   100 per model, 200 total. They test protocol wiring, not paper performance.
2. **Detection:** the same two models, all four full cohorts: NQ 3,610; TriviaQA
   9,960; SQuAD 5,928; RACE 3,498 per model. Total **45,992** answers, **234**
   resumable parts of at most 200. Both models use the same boundary rule on
   every dataset, so the replacement table has a consistent protocol.
3. **Timing/scaling:** 3B, 8B, 9B and 27B on SQuAD and NQ; **8 intact jobs**, each
   200 queries × 3 paired repeats. Run after detection workers on the same host
   have finished, on one exclusive GPU. CPU thread budgets are fixed at four
   for Torch/MKL, one for OpenBLAS, and recorded. GPU telemetry and host load
   are sampled every 30 seconds in `telemetry/*.ndjson`; this is not continuous
   proof of isolation. Other users' CPU workloads can still affect timing.

The main plan still contains **242 parts**, and the pilot plan contains eight.
Do not rerun the main baseline table, ablations, decoding sweeps, or 70B for this
correction. Attention mass, norm, both norm directions, token-count control,
output-length control and uncertainty estimates remain in the analysis. A
stopping correction does not guarantee that HIDE will beat these baselines.

## Setup on optimus

Finish the old queue before updating. In the existing environment:

```bash
cd ~/HIDE
source /home/anwoy/miniconda3/etc/profile.d/conda.sh
conda activate hide-paper
git pull --ff-only origin main
source configs/env.local.sh
bash scripts/run_answers.sh init
```

No dependency changes are required. New outputs go to `outputs/answer-boundary`;
the completed `outputs/review-fixed` queue is never reused or migrated. The
scientific source fingerprint has changed. Old snapshots remain readable, and
their saved analysis is available; use commit `c1bdd6b` or `5baab47` to reproduce
the old strict merge. The earlier keyword-only migration does not authorize
mixing these answer-boundary results with the old protocol.

## Recommended: launch everything in detached tmux sessions

```bash
bash scripts/start_answer_tmux.sh \
  --llama-gpu 0 \
  --gemma-gpu 1 \
  --timing-gpu 1
```

The launcher resolves physical indices to full GPU UUIDs and sets the process
device to `cuda:0`. It uses the absolute Python executable from the active
environment, and sources `configs/env.local.sh` in each session.

- `hide-answer-llama`: pilot for Llama, automatic pilot checks, then its four
  full detection cohorts in NQ/TriviaQA/SQuAD/RACE order.
- `hide-answer-gemma`: the same for Gemma, concurrently on the other GPU.
- `hide-answer-timing`: waits until every detection part is complete and
  unlocked, then starts the eight timing jobs on physical GPU1. GPU1 must be
  exclusive when timing begins. Existing platform/exclusivity guards stay on.

Each session has one **18-hour soft / 20-hour hard total budget**, including
pilots or waiting. It saves progress and can be resumed. This bounds runtime;
it does not guarantee that the entire assigned workload finishes in 20 hours.
If detection uses most of the budget, timing may need another invocation.

```bash
tmux ls
tmux attach-session -t hide-answer-gemma
```

Detach with **Ctrl+B, then D**. Ctrl+Z suspends the parent and can leave its child
running; it is not the detach command. Logs persist under
`outputs/answer-boundary/logs/`. Existing session names are rejected to prevent
duplicate launchers. Do not also launch the manual commands below while the
automatic sessions are active.

## Individual commands (alternative to the automatic launcher)

Run these inside activated tmux sessions after sourcing `configs/env.local.sh`.
Use one command per worker; `detection` automatically runs/resumes and validates
that model's four pilot datasets before full inference.

Llama, physical GPU0:

```bash
export CUDA_VISIBLE_DEVICES=GPU-005f009a-9e24-f684-0b6a-662d593d93ce
export HIDE_DEVICE=cuda:0
bash scripts/run_answers.sh detection --models llama3-8b --hours 18 --hard-hours 20
```

Gemma, physical GPU1:

```bash
export CUDA_VISIBLE_DEVICES=GPU-4957c764-5f46-b8cb-d688-a4a667c9e288
export HIDE_DEVICE=cuda:0
bash scripts/run_answers.sh detection --models gemma-2-9b --hours 18 --hard-hours 20
```

Timing, in a third session; it waits for the detection queue:

```bash
export CUDA_VISIBLE_DEVICES=GPU-4957c764-5f46-b8cb-d688-a4a667c9e288
export HIDE_DEVICE=cuda:0
bash scripts/run_answers.sh timing --wait-for-detection --hours 18 --hard-hours 20
```

To run only pilots first:

```bash
bash scripts/run_answers.sh pilot --models gemma-2-9b
bash scripts/run_answers.sh pilot --models llama3-8b
```

Pilot summaries, including all pilot questions/answers, are saved under
`outputs/answer-boundary/pilot/checks/`. The automatic checks establish structural
validity, not a scientific guarantee. They preserve all records if they fail.

### One dataset at a time

These are alternatives to the full-model detection commands above. They use
the same queue and resume completed parts. Set the intended GPU first.

```bash
bash scripts/run_answers.sh detection --models gemma-2-9b --datasets nq_open
bash scripts/run_answers.sh detection --models gemma-2-9b --datasets triviaqa
bash scripts/run_answers.sh detection --models gemma-2-9b --datasets SQuAD
bash scripts/run_answers.sh detection --models gemma-2-9b --datasets race

bash scripts/run_answers.sh detection --models llama3-8b --datasets nq_open
bash scripts/run_answers.sh detection --models llama3-8b --datasets triviaqa
bash scripts/run_answers.sh detection --models llama3-8b --datasets SQuAD
bash scripts/run_answers.sh detection --models llama3-8b --datasets race
```

All use the default 18/20-hour budget. Dataset filters never change cohort size.
Separate servers can own distinct model/dataset combinations using the same
commit, environment and checkpoint contents. Avoid overlapping assignments
across separate filesystems. Export each server as a distinct partial snapshot
with `--allow-incomplete`, then merge its restored queue with the others. A
timing waiter only observes its local queue; run timing explicitly after
coordinating separate servers. Run the two models' small pilots on the timing
server as well if their pilot results have not been transferred there.

## Status, resume and results

```bash
bash scripts/run_answers.sh status
tail -n 15 outputs/answer-boundary/logs/detection-gemma-2-9b.log
tail -n 15 outputs/answer-boundary/logs/timing-all.log
```

For a worker that exhausted its budget, rerun that worker's identical command
inside tmux. Completed parts and successful rows are skipped. Do not update
scientific code while the new plan is running. The timer includes waiting, so
restarting only the timing session is sufficient if detection finished first.

When status shows **242 complete, 0 running**, all workers have exited, and the
timing waiter is also finished, export the new queue:

```bash
python -m hide.export_results \
  --source outputs/answer-boundary \
  --output results/answer-boundary-complete-01
python -m hide.export_results --verify results/answer-boundary-complete-01
git add results/answer-boundary-complete-01
git commit -m "Save corrected answer-boundary experiments"
git push origin main
```

The export includes pilot reports, raw measurements, sources, worker logs and
telemetry. Choose a new snapshot name if the destination already exists. For
CPU-only analysis after all workers exit:

```bash
bash scripts/parts.sh merge \
  --plan outputs/answer-boundary/plan.json \
  --queues outputs/answer-boundary \
  --output outputs/answer-boundary-merged
bash scripts/analyze.sh --source outputs/answer-boundary-merged
```

No CUDA reruns are required merely to compute tables/figures from that snapshot.
