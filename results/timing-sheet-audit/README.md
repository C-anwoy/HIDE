# Timing-sheet verification and revision presentation

Source: user-supplied `HIDE_CL_Experiments - Time(1).csv`, preserved as `source.csv`.

For each model/method, average the four dataset means equally. For each comparator,
calculate `100*(1-HIDE/comparator)` within each model, then average the six percentages.

- EigenScore: 50.8041514218% reduction.
- Lexical similarity: 50.8391304353% reduction.
- LN-Entropy: 53.2302216984% reduction.
- Mean across these three comparators: 51.6245011851%.

The paper headlines **50.8% relative to EigenScore** consistently. It does not pool
this benchmark on A100 SXM4 with the paired-overhead measurements on A100 PCIe.
Figure 3 uses exactly the sheet's dataset means. No GPU measurement was rerun.

Llama-3-8B/SQuAD: base 1.616885471 s; total 1.844477796 s; incremental overhead
0.227592325 s, or 14.0759706907%. The 1.710671223-second HIDE total belongs to RACE,
whose base is 1.5021086 seconds. Mixing these datasets caused the prose discrepancy.
All 46 supplied mean cells pass independent recalculation within 1e-8 seconds.

The file contains aggregate cells, not timing repetitions, so it cannot supply
confidence intervals or verify every runtime configuration. Its HIDE identifier
is retained literally; no actual token budget is inferred from that identifier.

## Reproduce without GPU

```bash
python scripts/audit_timing_sheet.py
bash scripts/build_paper.sh
```

`audit.json`, `overheads.csv`, and `validation.json` preserve the calculations.
The paper's original Appendix B data rows are unchanged. The new detector tables
include Appendix B reference panels and distinct same-generation panels; numbers
from different cohorts are never substituted into a paired comparison.

The relevant literature supports a hidden-state access distinction and motivates
both types of detector. It does not establish universal necessity of HSIC or an
unmeasured HIDE speed advantage over attention. The selected-token-count limitation
remains disclosed. All substantive manuscript revisions are magenta.
