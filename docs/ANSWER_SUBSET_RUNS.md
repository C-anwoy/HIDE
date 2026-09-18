# Deadline-limited paired detector comparison

This comparison uses exactly the first **2,000 positions in the existing seed-42
shuffled cohort** for each dataset and each model. It is not a ranking by detector
score, correctness, or ease. The choice was made for the deadline after examining
earlier full-cohort results; it is not a preregistered study.

Models: Llama-3-8B and Gemma-2-9B. Datasets: NQ and TriviaQA (closed-book
factuality), SQuAD and RACE (faithfulness). All three detector scores—HIDE,
attention mass, and the input-state norm proxy—come from each same corrected
generation and share its correctness labels. Retain the saved count control too.
Do not substitute historical aggregate HIDE numbers or adjust the subset to
produce a preferred ranking.

There are **80 existing 200-example parts**, or 16,000 answers total. Each model
has 40 parts. Completed corrected parts are skipped and partial parts resume via
the unchanged worker. Existing pilots are reused and their gates remain active.
Earlier-protocol generations cannot substitute for these corrected generations.
NQ/TriviaQA parts run first, alternating datasets at part boundaries, followed by
SQuAD/RACE. No timing or scaling experiment is scheduled.

## Switch from full-cohort workers

First stop each old detection worker cooperatively: attach to its tmux session,
press **Ctrl+C once**, and wait for its child to save and exit. Do not use Ctrl+Z.
If the session is detached, send Ctrl+C with:

```bash
tmux send-keys -t hide-answer-llama C-c
tmux send-keys -t hide-answer-gemma C-c
```

A missing session may already have exited. Check the processes and wait for the
old parents and their runner children to exit before starting the new workers:

```bash
ps -u "$USER" -o pid,ppid,etime,stat,args \
  | grep -E '[h]ide[.]answer_runs|scripts/[a]nswer_worker[.]py|[h]ide[.]runner'
```

Then:

```bash
cd ~/HIDE
git pull --ff-only
source /home/anwoy/miniconda3/etc/profile.d/conda.sh
conda activate hide-paper
python scripts/answer_subset.py status
bash scripts/start_subset_tmux.sh
```

This starts `hide-2000-llama` and `hide-2000-gemma`, both on physical GPU1, with
8-hour soft and 9-hour hard budgets per worker. This bounds the worker sessions;
it does not guarantee all selected examples finish. Existing outputs and plans
remain in `outputs/answer-detection`. Session shells stay open on exit/error.

```bash
python scripts/answer_subset.py status
tail -f outputs/answer-detection/logs/subset-gemma-2-9b.log
# Or attach; detach using Ctrl+B, then D:
tmux attach-session -t hide-2000-gemma
```

The subset is complete at **80 complete, zero missing/partial/failed/running**.
The original full-plan status still has 234 tasks; it is deliberately not edited.
The persisted subset specification lists every included task and the original
plan hash in `subsets/answer-2000.json`. Its script snapshot is retained alongside
it. Only launcher/tests/docs files changed, preserving scientific fingerprints.

To resume just one model from an activated tmux shell, without rerunning completed
parts, use (adjust the remaining time budget as appropriate):

```bash
source configs/env.local.sh
python -u scripts/answer_subset.py run --model gemma-2-9b --gpu 1 --hours 8 --hard-hours 9
```

## Export after both workers exit

```bash
python scripts/answer_subset.py export --output results/answer-2000-complete-01
python -m hide.export_results --verify results/answer-2000-complete-01
git add results/answer-2000-complete-01
git commit -m "Save fixed 2000-example corrected detector comparison"
git push origin main
```

The export command requires all 80 selected parts and successful pilot checks.
It retains the entire existing queue, including any additional examples already
run, and labels the parent full-cohort plan as incomplete when applicable. That
label does not mean the declared subset is incomplete. Analysis must use exactly
the task list in the subset specification, not pool all exported JSONL files or
report the original full suite as complete. Per-example IDs, labels, scores,
source snapshots and runtime metadata remain available for CPU analysis.

Report AUROC/PCC and paired confidence intervals for all eight model/dataset
cells, including losses. A 2,000-example cohort may still contain few positive
labels; report the class counts and resulting uncertainty. No superiority or
publication outcome is guaranteed by this sample size.
