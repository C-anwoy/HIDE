# HIDE and Seek

Reproducible code for **HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations**.

HIDE scores a generated answer using selected prompt/output hidden states collected during one autoregressive generation. This repository contains the detector, audited QA data loaders, paired baseline experiments, score ablations, synchronized timing, analysis, and verifiable result snapshots.

**Status:** the supported workflow has offline unit and tiny-model integration tests. Full checkpoint/CUDA experiments still need to be run on the A100. This is an experiment repository; it is not an optimized serving implementation. The supplied manuscript is included in `paper/`, with its submitted numbers unchanged pending fresh results.

## Repository layout

```text
hide/                  Installable Python package and command-line tools
  config/              Model mappings and declared experiment suites
  datasets/            Four QA loaders and original prompt/stop conventions
scripts/               Setup, individual/suite launch, analysis, export, paper build
configs/               Example local environment configuration
tests/                Offline numerical, persistence, protocol and integration tests
docs/                 Method specification, audit, experiment plan and manuscript checklist
paper/                LaTeX source and original figures; provenance in SOURCE.json
outputs/              Local raw runs and analyses (created at runtime; Git-ignored)
results/              Verified, compressed snapshots intended for Git
archive/original/     Original code for provenance and parity tests; unsupported entry points
```

## Update an existing GPU checkout

Commit/push the reorganized source from this working tree, then pull that commit on the GPU machine before starting runs. A Git update records the moves into `archive/original/`; copying new files over an old checkout does not remove obsolete entry points. The `hide_repository.zip` transfer bundle contains a fresh `HIDE/` source directory without `.git`, weights, caches or results.

Use a new `HIDE_RESULTS_ROOT` for this schema/protocol version. Keep any earlier outputs intact and analyze them separately; do not append new records to old files.

## Install

Use Linux with Python 3.10+ and a CUDA-compatible PyTorch environment for inference. The intended GPU is one A100 80GB. The local tests also run on CPU.

```bash
bash scripts/setup.sh
source .venv/bin/activate
cp configs/env.example.sh configs/env.local.sh
# Edit model/data/output paths in configs/env.local.sh, then:
source configs/env.local.sh
bash scripts/check.sh
python -m hide.prepare_data --data-root "$HIDE_DATA_ROOT"
```

The inference dependency versions are declared in [pyproject.toml](pyproject.toml). For saved-result analysis without installing inference dependencies, use `python -m pip install -e .`. Export verification/restoration needs only Python's standard library when run from this checkout.

### Checkpoints

Place these directories under `HIDE_MODEL_ROOT`, or edit [hide/config/models.json](hide/config/models.json):

| Run name | Checkpoint directory |
|---|---|
| `llama3-3b` | `Llama-3.2-3B` |
| `llama3-3b-instruct` | `Llama-3.2-3B-Instruct` |
| `llama3-8b` | `Meta-Llama-3-8B` |
| `llama3-8b-instruct` | `Meta-Llama-3-8B-Instruct` |
| `gemma-2-9b` | `gemma-2-9b` |
| `gemma-2-9b-instruct` | `gemma-2-9b-it` |
| `gemma-2-27b` | `gemma-2-27b` |

Also provide `all-MiniLM-L6-v2` for KeyBERT and `nli-roberta-large` for correctness similarity. The launcher checks local directories; it does not download model weights. Data caches are created under `HIDE_DATA_ROOT/datasets`. Dataset counts must match the paper before sampling: SQuAD 5,928; RACE 3,498; NQ 3,610; TriviaQA 9,960.

## Run

Start with the stored 25-example pilots:

```bash
bash scripts/run_pilots.sh
python -m hide.summarize_runs --source "$HIDE_RESULTS_ROOT"
```

Run one experiment with:

```bash
bash scripts/run.sh PROFILE MODEL DATASET
```

For example:

```bash
bash scripts/run.sh qa llama3-8b nq_open
bash scripts/run.sh comparison llama3-8b nq_open
bash scripts/run.sh ablations llama3-8b SQuAD
bash scripts/run.sh timing gemma-2-27b nq_open
```

`qa` computes HIDE, attention/norm proxies, MNLL and energy on one answer. `comparison` additionally generates five stochastic answers for LN-Entropy, lexical similarity and EigenScore. `ablations` computes all declared score variants from one generation. `timing` measures base versus HIDE latency, without correctness judging or extra detectors inside the timed pipeline.

### Named suites

| Command | Purpose | Jobs |
|---|---|---:|
| `bash scripts/run_review.sh` | Reviewer-requested baseline/mechanism results and larger-model timing | 8 full QA + 8 timing |
| `bash scripts/run_consistency.sh` | Rebuild the six-model/four-dataset main comparison | 24 full comparison |
| `bash scripts/run_ablations.sh` | Regenerate layer/budget/kernel/estimator/selection analyses | 4 full ablation |
| `bash scripts/run_decoding.sh` | Recompute temperature/nucleus results with explicit samplers | 24 full detection |
| `bash scripts/run_timing.sh` | Run only the four-model/two-dataset timing study | 8 timing |

Use `--dry-run` on any launcher to inspect commands. Run suites sequentially on one GPU. **Do not launch every suite blindly under a two-day deadline:** comparison sampling alone multiplies the generation cost. The [experiment plan](docs/EXPERIMENTS.md) distinguishes mandatory corrections, reviewer additions, and a complete table rebuild.

All primary accuracy runs use full splits (`samples=0`). Timing uses 200 seeded prompts, three repeats and ten warmups. Runs resume automatically only with matching metadata. Stochastic seeds are stable per example. For recovery and storage details, see [RESULTS.md](docs/RESULTS.md).

## Analyze, export and push

```bash
bash scripts/analyze.sh
bash scripts/export.sh final --suite review
```

If also rebuilding the original comparison/ablation/decoding tables, require the suites actually being claimed:

```bash
bash scripts/export.sh full --suite review --suite consistency --suite ablations --suite decoding
```

The exporter verifies completeness, the declared settings, source snapshots and checksums. Every working output file is exported losslessly into `results/NAME/`. Large files are split into bounded compressed chunks; CSV/Markdown summaries remain readable. **No successful run automatically commits or pushes.**

```bash
git status --short
git add -A
git diff --cached --stat
git commit -m "Organize HIDE experiments and add verified results"
git push origin HEAD
```

Inspect staging before committing, particularly the source-file moves. Model weights, data caches, environments and working outputs are ignored. See [results/README.md](results/README.md) for restoring a snapshot after a Git pull.

## Method and audit

- [METHOD.md](docs/METHOD.md): exact score, token alignment, labels, baseline conventions and complexity.
- [AUDIT.md](docs/AUDIT.md): code-to-paper findings and correction status.
- [EXPERIMENTS.md](docs/EXPERIMENTS.md): separate commands, rerun priorities and manuscript dependencies.
- [RESULTS.md](docs/RESULTS.md): schemas, resume/recovery, provenance and lossless export.
- [MANUSCRIPT_CHANGES.md](docs/MANUSCRIPT_CHANGES.md): text/caption changes and interpretation constraints.
- [RESPONSE_LETTER_TEMPLATE.md](docs/RESPONSE_LETTER_TEMPLATE.md): point-by-point response scaffold.
- [VALIDATION.md](docs/VALIDATION.md): what was tested and what still requires real checkpoints.

The adapted HIDE formula, FP32 kernel arithmetic, RBF gamma `1e-7`, keyword ordering and duplicate token occurrences are retained and tested against the original implementation. The new mechanistic, baseline and timing protocols correct documented discrepancies; their results must be identified as freshly measured. Do not splice new HIDE numbers into an old comparison with different generations/labels.

## Manuscript

Build the included source with a TeX Live/MacTeX installation:

```bash
bash scripts/build_paper.sh
```

The build command does not update results or claims. Insert verified measurements and finish the response checklist first, then inspect the compiled PDF. No acceptance, published DOI, or unmeasured result is implied by this repository.
