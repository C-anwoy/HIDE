# Verified experiment snapshots

This directory is tracked by Git. It contains no fabricated experiment results.

After running and analyzing experiments:

```bash
bash scripts/export.sh final --suite review
```

Each named snapshot has an `INDEX.json`, exact hashes/completeness information, readable summaries under `files/analysis/`, and losslessly compressed chunks of per-example results. Everything under the working output directory is included, including logs, source/config snapshots and recovery history.

After pulling a snapshot:

```bash
python -m hide.export_results --verify results/final
python -m hide.export_results --restore results/final --output outputs/restored
export HIDE_RESULTS_ROOT="$PWD/outputs/restored"
bash scripts/analyze.sh
```

See [the results guide](../docs/RESULTS.md) for schema, suite requirements, recovery and checkpoint export. Do not claim a partial snapshot is a completed experiment. Full activation tensors/model weights are not included.
