# Local models and startup downloads

## Current server: optimus

The supplied `/models` directory listing contains the six original-study checkpoints and both auxiliary encoders:

| Experiment name | Local directory |
|---|---|
| `llama3-3b` | `/models/Llama-3.2-3B` |
| `llama3-3b-instruct` | `/models/Llama-3.2-3B-Instruct` |
| `llama3-8b` | `/models/Meta-Llama-3-8B` |
| `llama3-8b-instruct` | `/models/Meta-Llama-3-8B-Instruct` |
| `gemma-2-9b` | `/models/gemma-2-9b` |
| `gemma-2-9b-instruct` | `/models/gemma-2-9b-it` |
| KeyBERT encoder | `/models/all-MiniLM-L6-v2` |
| Correctness encoder | `/models/nli-roberta-large` |

The listing shows `/models/gemma-2-27b-it`, but the scaling plan uses **base `google/gemma-2-27b`**. The downloader fetches that exact model; it does not substitute the instruction-tuned checkpoint. Directory names alone do not prove file completeness or model identity; validation happens on the server before inference.

## Configure

Set these in `configs/env.local.sh`, then source it inside the worker's tmux pane:

```bash
export HIDE_MODEL_ROOT=/models
export HIDE_CHECKPOINT_CACHE="$PWD/checkpoints"
export HIDE_DOWNLOAD_MISSING=1
```

Keep the data/output/device settings from the normal environment configuration. `/models` is read only to this workflow. Missing weights are downloaded into the user-owned cache, not written back into the shared model directory. `checkpoints/` is Git-ignored. Another server may use a different local root/cache, provided it runs the same code, package versions and checkpoint contents.

## Automatic behavior

Both the ordinary launchers and partitioned workers prepare their named checkpoints at startup:

1. Prefer the requested local directory if present.
2. Validate config, tokenizer vocabulary, weight files/indexed shards, sentence-transformer module/pooling configuration, and absence of a quantization configuration. An incomplete/invalid existing directory raises an error; it is never silently replaced.
3. If absent, download the exact repository **and full commit SHA** in `hide/config/checkpoints.json` to `HIDE_CHECKPOINT_CACHE/downloads/REPOSITORY/REVISION`.
4. Filter downloads to the declared Transformers/sentence-transformer files; original-format, ONNX/OpenVINO and bundled framework exports are not downloaded.
5. Hash checkpoint weights once and cache their SHA-256 values, with size/mtime/ctime invalidation. Numeric sentence-transformer submodule configs/weights are included. This reads large weight files but does not load the model on the GPU. Subsequent parts reuse verified hash records while file stats remain unchanged.
6. Save resolved paths, local-versus-download origin, downloaded revision (if known) and full checkpoint identity next to the run as `.checkpoints.json`.

Local directories are not claimed to match the Hub revision solely because their names match. Their actual file identities are saved. Download revisions were resolved from the official Hub metadata on 2026-09-17 (local date); all servers running this code use the same pinned revisions. Preparation locks serialize downloads/hashing so concurrent workers using the same cache do not write over each other.

Downloads, validation and hashing happen **before GPU model loading and before warmups or latency measurement**. For partitioned workers they are inside the supervised process, so they count toward the 18/20-hour allocation. A download interrupted by the supervisor can reuse Hugging Face's saved progress when the worker restarts. The ready marker is written only after download, validation and hashing complete. A complete cached download can be reused with downloads disabled.

To disallow new downloads:

```bash
export HIDE_DOWNLOAD_MISSING=0
```

Custom direct-runner model names outside the catalog retain explicit local-path behavior.

## Optional prefetch before allocating a GPU

Prepare all nine required checkpoints (existing ones are reused):

```bash
bash scripts/prepare_models.sh
```

Prepare only the missing scaling checkpoint:

```bash
bash scripts/prepare_models.sh --models gemma-2-27b
```

Validate/hash the existing first-run models without downloading anything:

```bash
bash scripts/prepare_models.sh --models llama3-8b keyword judge --local-only
```

The command writes `checkpoints/checkpoint_inventory.json` and per-checkpoint hash caches. It is a CPU/network/disk task, so it can run in tmux while the GPU is unavailable. Unlike a partition worker, this standalone prefetch command has no 20-hour supervisor. For a bounded invocation, allow the normal partition worker to perform preparation at startup instead.

## Hugging Face authentication

Llama and Gemma repositories are gated. Existing local weights do not need Hub access. To download a gated checkpoint, authenticate in the same environment and accept its access conditions on the model page if necessary:

```bash
huggingface-cli login
```

For the missing model, use [google/gemma-2-27b](https://huggingface.co/google/gemma-2-27b). Enter tokens only into the CLI's interactive prompt; do not commit credentials. The helper uses Hugging Face's normal authentication and does not read unrelated server credential files or bypass access controls. Authentication/network/disk failures are surfaced, and partial download files are retained for retry.

## Matching checkpoints across servers

Named runs prepare full weight hashes automatically. The merge compares content hashes, so **different file timestamps no longer prevent matching identical downloaded/copied checkpoint files**. Keep the same checkpoint files and software versions: a newer Hub snapshot, a different file format, different tokenizer/pooling config or a locally fine-tuned checkpoint can still fail identity checks. Do not equate local folders with official repository revisions without evidence. If another server has a different version, copy the same trusted checkpoint directory or agree on a common pinned download before running parts.

The original model and detector precision, layer, token selection and generation settings are unchanged. Checkpoint files/cache never enter result exports; `.checkpoints.json`, run manifests and source snapshots do, so the results remain traceable after Git transfer.

## Update before creating the work plan

This feature changes the code fingerprint. Pull the update and install the pinned dependencies **before** generating `plan.json`. Do not mix old and new code within a plan.
