# Original implementation archive

`original/` preserves the pre-reorganization experiment scripts, helpers, notebook and figures for provenance. These files contain known implementation/protocol issues documented in [AUDIT.md](../docs/AUDIT.md). They are not installed and are not supported execution entry points.

The active implementation is `hide/`; use the root README and `scripts/`. Tests extract original scoring functions from `original/func/metric.py` to verify preservation of HIDE's formula and keyword behavior without importing legacy modules or their dependencies.

Moving files here does not certify historical experiments or reconstruct missing outputs. The supplied raw historical results and probe-training artifacts were not present. The archive is retained to make corrections reviewable, not to conceal them.
