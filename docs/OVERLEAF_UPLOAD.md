# Final Overleaf upload

## Recommended: use the complete source ZIP

Upload `output/HIDE_final_revision_source.zip` with Overleaf's **New Project → Upload Project**, or replace the corresponding files in the existing project. The ZIP opens at the project root; it does not add a `paper/` directory.

- Paper main document: `COLI_template.tex`.
- Response main document: `response_letter.tex` (compile separately).
- The supplied PDFs were built with Tectonic's XeTeX engine. Select **XeLaTeX** in Overleaf for the closest match.
- Keep the folder structure and upload the `.bib` and figure PDFs as well as the `.tex` files.
- The revision command is already defined. Do not redefine it or add the table inputs a second time.

## Exact files to replace/add relative to the Overleaf project root

### Root files

- `COLI_template.tex`
- `COLI_template.bib`
- `response_letter.tex`

### Manuscript sections

- `files/abstract.tex`
- `files/intro.tex`
- `files/preliminary.tex`
- `files/method.tex`
- `files/exp_setup.tex`
- `files/results_summ_table.tex`
- `files/results.tex`
- `files/ablations.tex`
- `files/error_analysis.tex`
- `files/conclusion.tex`
- `files/proofs.tex`
- `files/results_table.tex`
- `files/Prompting_techniques.tex`
- `files/paired_diagnostics.tex`

### New table inputs

- `files/tables/paired_s.tex`
- `files/tables/paired_r.tex`
- `files/tables/paired_differences.tex`
- `files/tables/count_control.tex`
- `files/tables/timing_revision.tex`
- `files/tables/timing_lengths.tex`

### Replacement figures

- `files/figures/llama3-8b_Mechanistic_Flow_updated.pdf` — Figure 2, Llama panel.
- `files/figures/gemma-2-9b_Mechanistic_Flow_updated.pdf` — Figure 2, Gemma panel.
- `files/figures/computation_time_plot_updated.pdf` — Figure 3.
- `files/figures/Scalability_Analysis_updated.pdf` — Figure 4.

All other existing source files, the class, the bibliography style, and unchanged figures remain needed. They are included in the complete ZIP.

## Final package

Submit the revised manuscript, response letter, and the current original decision letter requested by the journal. The decision letter is a separate journal document; it is not replaced by our response.

The final paper has 47 pages and the response has 6 pages in the supplied build. Minor pagination changes between TeX installations are possible; response locations use stable section, figure, and table numbers rather than PDF page numbers.
