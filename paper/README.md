# Manuscript source

Copied from the supplied `3205_HIDEandSeek_Revised_Round_3` directory. `SOURCE.json` records original source hashes. The stale bundled `COLI_template.pdf` was not copied; the supplied `3205_HIDEandSeek_Revised_Round_3-1.pdf` is the submission record.

Main file: `COLI_template.tex`. Updated on 19 September 2026 using the verified fixed
2,000-example paired detector comparison and the retained earlier paired timing
suite. Figures 2 and 4 use the paired analyses. Figure 3 is regenerated from the supplied timing sheet, verifying 50.8% reduction versus EigenScore. Main detector/ablation values remain unchanged; reference panels reproduce Appendix B values alongside a separately labeled matched comparison. They are not claimed to
have been regenerated or retroactively validated.

The point-by-point letter is `response_letter.tex`. Ready PDFs and updated plot
PDFs are in `../output/pdf/`. The manuscript and letter were compiled with
Tectonic 0.17.0 and visually reviewed. See
`../docs/SUBMISSION_NUMERICAL_AUDIT.md` and `../docs/FINAL_MANUSCRIPT_CHANGES.md`.

Build from the repository root with:

```bash
bash scripts/build_paper.sh
```

The earlier planning documents are historical checklists. The completed changes
are documented in [FINAL_MANUSCRIPT_CHANGES.md](../docs/FINAL_MANUSCRIPT_CHANGES.md).
Before submission, review the qualifications of historical results and include
the current original decision letter alongside the revised manuscript and response.

Changes since the submitted source are marked with `\revisionr3{}` in magenta. The supported contribution and limits relative to attention are discussed with three primary references. Benchmark comparison and paired-overhead timing have explicit platforms and averaging rules; the manuscript does not label them as historical.
