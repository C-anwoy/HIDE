# Proposed manuscript changes

These are copy-edit instructions and insertion-ready drafts, not claims that new experiments have finished. The original LaTeX folder has not been overwritten. Replace all `RESULT_*` fields with actual regenerated measurements before submission. Recompile the complete manuscript and inspect page layout; the submitted PDF inspection was read-only. Paths below are relative to paper/.

## 1. Section 4.4 — describe measured proxies accurately

File: `files/method.tex`, subsection `Structural Validation of Decoupling`.

Replace the definition/approximation paragraph with:

```latex
To investigate the association between the HIDE score and attention-mediated
information flow, we measure two structural proxies at the decoder block
corresponding to the selected hidden-state layer. Attention mass to the prompt is
\[
\Omega = \frac{1}{|\mathcal J|H}
\sum_{j\in\mathcal J}\sum_{h=1}^{H}\sum_{i=1}^{I}
\alpha_{j,i}^{(h)},
\]
where $\mathcal J$ contains the generated-token positions whose hidden states
are available during cached autoregressive generation. We also measure the
unprojected attention-weighted input-state norm
\[
\widetilde\Delta_{\mathrm{in}} = \frac{1}{|\mathcal J|}
\sum_{j\in\mathcal J}\left\|
\sum_{i=1}^{I}\left(\frac{1}{H}\sum_{h=1}^{H}\alpha_{j,i}^{(h)}\right)
h_{i,\mathrm{in}}^{(\ell-1)}\right\|_2.
\]
This second quantity is a proxy for the magnitude of input-related information
aggregation. It is not the projected attention update written to the residual
stream: it omits value and output projections and architecture-specific
normalization. Both proxies are averaged over the same output positions.
The final generated token is excluded because its hidden state is not available
without an additional language-model forward pass. For generations with no
available output-token state, we retain an explicit zero-score fallback and
report their frequency.
```

If retaining the theoretical projected update equation above it, call it `Delta_in_projected`, and say it motivates rather than equals the measured proxy. Propagate the tilde/name consistently into Figure 2 and the baseline table. Keep layer notation l/ell consistent.

### Interpretation, including Gemma explicitly

For a **text-only correction using the submitted Figure 2**, this paragraph accurately states its reported values:

> The relationship between HIDE and the update-norm proxy depends on the model and dataset. In the submitted analysis, Llama-3-8B exhibits a negative correlation on SQuAD (r = −0.207) and an approximately zero correlation on NQ (r = −0.004), whereas Gemma-2-9B exhibits positive correlations on SQuAD (r = 0.302) and NQ (r = 0.091). These observations do not support a universal inverse relationship between update magnitude and input–output coupling. We therefore treat the magnitude-versus-precision interpretation as a model- and dataset-dependent possibility rather than an established general mechanism. Correlation with attention mass motivates the standalone proxy comparisons reported below; it does not by itself establish that HIDE provides additional detection value.

For the **recommended regenerated analysis**, remove “in the submitted analysis” and substitute every correlation from the new full-cohort `correlations.csv`. Do not keep the old numbers under a new figure or new filtering/layer definition.

Replace broad causal claims throughout this subsection:

| Current formulation | Replacement |
|---|---|
| “empirical evaluation validates this explanatory framework” | “the observed correlations are consistent with an association between the HIDE score and attention to the prompt” |
| “causing the HIDE score to collapse” | “and lower HIDE scores are observed in these examples” |
| “leading the joint probability distribution to factorize” | “which motivates testing whether lower empirical dependence scores accompany errors” |
| “the specific structural circuit weakens, reducing mutual information” | “the relationship between attention, stored knowledge, and answer correctness may differ from the context-grounded setting” |
| “HIDE successfully isolates genuine statistical alignment from mere vector magnitude” | “the standalone comparisons assess whether HIDE offers discrimination beyond the two structural proxies” |
| “isolates causal operations” | “is sensitive to errors that remain semantically related to the question” |
| “confirming ... structural collapse across both failure modes” | “providing an observational control against an explanation based only on off-topic outputs” |

Use this QCBW paragraph:

```latex
We additionally examine question-conditioned but wrong (QCBW) answers in
Natural Questions. We define this diagnostic subset as incorrect generations
in the highest quartile of question--generation sentence-embedding similarity.
The QCBW examples remain topically related to the question by this criterion,
allowing us to inspect whether low HIDE scores also occur for on-topic errors.
This analysis is observational: it does not identify a causal circuit or establish
that reduced prompt attention is necessary or sufficient for hallucination.
```

Only add a statement about where QCBW clusters if it remains true in the regenerated figure. If adding TriviaQA QCBW, explicitly extend the definition and report per-dataset counts. Do not evaluate QCBW-only AUC.

## 2. Add standalone baseline table in the main text

Place after the main mechanistic discussion or in the main results, with a cross-reference from Section 4.4. Do not put all new evidence exclusively in the appendix.

Suggested setup prose:

```latex
We evaluate attention mass to the prompt and the unprojected input-state norm
as standalone detection baselines, alongside HIDE, using identical generations
and correctness targets. The comparison covers Llama-3-8B and Gemma-2-9B on
SQuAD, RACE, Natural Questions, and TriviaQA. We use the full evaluation split of each dataset and process examples in a
fixed seeded order, reporting the actual evaluated counts for every setting.
We retain all valid paired examples without HIDE-score filtering or class
rebalancing and report the number of excluded failures, if any.
We use a fixed confidence orientation for each reported score and include the
negative update norm as a separately specified sensitivity analysis.
Confidence intervals for AUC differences are obtained by resampling paired
examples with replacement 2,000 times.
```

Main table layout:

| Model | Dataset | N correct / N total | HIDE AUC_s / PCC_s | Omega AUC_s / PCC_s | Delta AUC_s / PCC_s | HIDE − Omega AUC_s, 95% CI |
|---|---|---|---|---|---|---|
| Llama-3-8B | SQuAD | actual counts | actual result | actual result | actual result | actual CI |
| Llama-3-8B | RACE | actual counts | actual result | actual result | actual result | actual CI |
| Llama-3-8B | NQ | actual counts | actual result | actual result | actual result | actual CI |
| Llama-3-8B | TriviaQA | actual counts | actual result | actual result | actual result | actual CI |
| Gemma-2-9B | SQuAD | actual counts | actual result | actual result | actual result | actual CI |
| Gemma-2-9B | RACE | actual counts | actual result | actual result | actual result | actual CI |
| Gemma-2-9B | NQ | actual counts | actual result | actual result | actual result | actual CI |
| Gemma-2-9B | TriviaQA | actual counts | actual result | actual result | actual result | actual CI |

If this is too wide, use one row per detector and model/dataset group, with AUC/PCC columns; move paired differences and secondary metrics to an adjacent small table/appendix. Preserve every cell, including baselines outperforming HIDE. Clearly identify whether numbers are percentages.

**Caption:**

> Standalone detection performance of HIDE, attention mass to the prompt (Omega), and the unprojected attention-weighted input-state norm (Delta) on identical generations from Llama-3-8B and Gemma-2-9B. AUC_s uses sentence similarity >0.9 to define correctness; PCC_s correlates each detector score with continuous sentence similarity to the first reference answer. Higher reported confidence scores indicate predicted correctness. Bracketed intervals are paired 95% bootstrap confidence intervals; N denotes evaluated examples. Both factuality datasets are closed-book. [Add exact location of negative-norm, ROUGE and EM sensitivities.]

Choose the interpretation from the actual data:

- **HIDE advantage with positive paired CI:** “HIDE improves AUC over attention mass by X percentage points (95% CI [...]) in [setting], suggesting that attention mass alone does not reproduce its discrimination in this setting.” This supports practical added value, not a causal explanation.
- **Tie/uncertain difference:** “The two scores perform similarly in [setting]; the interval includes zero, and we do not establish an advantage for HIDE there.”
- **Baseline advantage:** “Attention mass/update norm outperforms HIDE on [setting]. Thus, HIDE's advantage is not universal; simple structural proxies are competitive in some model–dataset combinations.”
- **Closed-book interpretation:** “Attention to the question measures use of the query, not whether the model's parametric answer is correct. The results on NQ and TriviaQA indicate [actual finding].” Avoid predicting a weaker baseline before observing it.

If token-count control is competitive, add:

> The adapted estimator has a finite-sample count-dependent component: for constant unit Gram matrices it equals (n−1)/n². We therefore include a count-based control and analyses within selected-count ranges. [Actual result.] These findings qualify the interpretation of HIDE as a pure measure of semantic dependence while leaving its empirical detection performance directly testable.

## 3. Section 5 — clarify metrics and implementation

Replace the PCC definition to distinguish targets:

> We report Pearson correlation with the continuous reference-based correctness measure (sentence similarity or ROUGE-L), denoted PCC_s and PCC_r, respectively. AUC_s and AUC_r use binary labels obtained with thresholds of 0.9 and 0.5. For the new structural-baseline comparison we additionally report correlation with the corresponding binary correctness labels. Exact-match evaluation is binary.

State that the new experiment uses first-reference scoring to match the historical code, if following the supplied runner. Add BF16/eager attention/new sample-size/cap/CPU placement details for the supplementary experiment; do not rewrite the old experiments as though they used these settings.

Clarify token selection: k=20 is the **configured budget**, while n_eff is the realized count after keyword-to-subtoken matching and truncation. Repeated matched positions are retained by the submitted implementation. Avoid calling this a set of unique token types unless changing and validating the implementation.

## 4. Section 6.4 — replace conflicting efficiency account

Use the newly measured figures; the following is intentionally a result template:

```latex
\paragraph{Measurement protocol.}
We measure request latency using Hugging Face Transformers RESULT_VERSION on
one NVIDIA A100 RESULT_VRAM GPU, with RESULT_DTYPE weights, batch size one,
greedy decoding, a maximum of 256 generated tokens, and RESULT_BACKEND
attention. The timing substudy uses RESULT_N fixed prompts per dataset and
three repetitions per prompt, following ten warmup requests. The keyword model
is initialized once and remains resident on RESULT_KEYWORD_DEVICE. Model
loading and warmup are excluded. CUDA is synchronized at timing boundaries.
Base and HIDE runs alternate order and are checked to generate identical tokens.

\paragraph{Latency accounting.}
We distinguish base generation time $T_{\mathrm{base}}$, generation with
hidden-state capture $T_{\mathrm{capture}}$, auxiliary keyword-selection and
HSIC time $T_{\mathrm{score}}$, and directly measured end-to-end HIDE time
$T_{\mathrm{HIDE}}$. Incremental overhead is
$T_{\mathrm{HIDE}}-T_{\mathrm{base}}$; the reported percentage is
$100(\overline T_{\mathrm{HIDE}}-\overline T_{\mathrm{base}})/
\overline T_{\mathrm{base}}$. On RESULT_MODEL/RESULT_DATASET, the respective
mean base and total times are RESULT_BASE and RESULT_TOTAL seconds,
corresponding to RESULT_OVERHEAD seconds (RESULT_PERCENT percent) of overhead.
All values in this subsection and the associated figure use the same saved
measurements and aggregation. We retain all valid measurements, including
large relative overheads on short requests.

\paragraph{Scope of the scaling claim.}
With $n$ selected tokens and hidden width $d$, RBF Gram construction costs
$\mathcal O(n^2d)$; the dense matrix operations in our score implementation add
$\mathcal O(n^3)$. Thus, for fixed $n$, this score computation is linear in
$d$. This arithmetic statement does not imply constant end-to-end latency:
keyword selection, state capture and generation have additional dependencies
on sequence length and the implementation. Our empirical measurements cover
RESULT_MODELS; they do not establish scalability to arbitrary model sizes.

\paragraph{Serving-system scope.}
These measurements characterize a standard Transformers implementation.
We have not measured HIDE in an optimized serving engine, and therefore do
not extrapolate a numerical speedup or a lower bound on speedup to such
systems. Integration of intermediate-state extraction can introduce
implementation-dependent storage and transfer costs. This is an engineering
consideration, not empirical evidence of a particular serving-time advantage.
```

If retaining a verifiable historical 51% number, add **only**:

> In the original standard-Hugging-Face benchmark, HIDE required approximately 51% less end-to-end time than the specified five-generation baselines, averaged using [the verified original aggregation]. This is a comparison under that inference stack, not a measured gain in an optimized serving engine.

If provenance cannot be recovered, remove the exact number from abstract, contribution bullet, Section 6.4 and conclusion. Replace with the new measured overhead result where appropriate. No “lower bound” language is needed. Neither a complexity argument nor an example of vLLM hidden-state extraction supplies unmeasured latency results.

### Figure 4 replacement caption

> Descriptive overhead analysis for [model, datasets, hardware, dtype, backend, N queries and repetitions]. (a) Auxiliary keyword-selection and HSIC time versus the realized number of selected token positions under a configured budget of 20. (b) Incremental end-to-end HIDE overhead, 100(T_HIDE−T_base)/T_base, versus base generation latency for the same requests. All valid measurements are included. The plot is not a controlled sweep of sequence length, token budget or model size.

The new timing plot has exactly these meanings. If you instead design a controlled sweep, change the caption and experiment, not merely the x-axis title.

## 5. Requested self-contained captions

### Figure 3 — historical method timing

> Mean end-to-end request latency for the six indicated language models on RACE, SQuAD, Natural Questions (NQ), and TriviaQA under [exact HF version/backend/dtype] on one NVIDIA A100 80GB. Single-generation methods use one greedy answer; multipass methods use five sampled answers generated together with `num_return_sequences=5` [verify original configuration]. HIDE latency includes generation, keyword extraction and score computation. Bars aggregate [verified averaging rule]; error bars show [verified statistic]. These measurements concern this inference stack and do not quantify gains in an optimized serving engine.

**Do not fill the bracketed error-bar definition by guessing.** If replacing Figure 3 with the fresh narrower experiment, rename its caption/legend and revise references rather than implying that all six historical methods were rerun.

### Figure 5 — token budget

> Sensitivity of HIDE detection AUC_s (%) to the configured token budget k on SQuAD and Natural Questions (NQ), using Llama-3-8B and Gemma-2-9B. AUC_s defines correct answers by reference-answer sentence similarity >0.9. The realized number of selected token positions may be smaller than k for short outputs. The plotted performance changes little beyond approximately 15–20 tokens in these evaluated settings.

Change the x-axis notation from n_eff to k when possible, because the sweep is a requested budget. Name both models/datasets in the caption rather than relying on the legend.

### Figure 6 — layer

> HIDE detection AUC_s (%) across selected hidden-state layers of Llama-3-8B and Gemma-2-9B on SQuAD and Natural Questions (NQ), with other settings fixed. Correctness is defined by reference-answer sentence similarity >0.9. Layer indices follow the implementation's hidden-state convention [specify whether 0 is the embedding output]. Performance varies modestly across the plotted layers; these observations do not establish a layer-independent causal mechanism.

Verify whether index 0 in the actual ablation data is the embedding output: the legacy function loops over the entire hidden-state tuple. Do not call every plotted index a decoder layer without checking.

### Figure 8 — adapted versus biased estimator

> Detection AUC_s (%) obtained using the adapted HIDE score and the biased HSIC estimator on identical evaluation settings for Llama-3-8B and Gemma-2-9B on SQuAD and Natural Questions (NQ). Correctness uses reference-answer sentence similarity >0.9. Colors distinguish estimators and patterns distinguish models. The adapted score has higher AUC_s in the displayed comparisons.

Remove “scores close to zero make threshold determination unreliable” unless numerical-resolution evidence supports it. A strictly monotonic rescaling changes magnitude without changing AUC.

### Figure 2 — refreshed mechanistic figure

> Association of HIDE scores with attention mass to the prompt (top row) and the unprojected attention-weighted input-state norm (bottom row), for [models/datasets]. Each point is one generation; r is Pearson correlation calculated on all valid paired examples, with no score-based filtering or class rebalancing. Correctness is defined by sentence similarity to the first reference answer >0.9. Stars identify incorrect closed-book answers in the highest question–generation similarity quartile (QCBW). Correlation with update magnitude varies by model and dataset. [State selected layers and per-panel sample counts, or refer to counts printed on axes.]

## 6. Typos, list, threshold direction and cross-references

- `files/results.tex`: “factuality” and “performance” are already spelled correctly; fix “the performance ... are further detailed” to “the performance ... is further detailed.”
- `files/error_analysis.tex`: “verbatim repetition” is already correct.
- `files/intro.tex`: exactly three active contribution `\item`s; no blank bullet in the provided PDF p.3. Verify again after final recompilation.
- Section 6.5: with the rule `score < tau`, **raise tau to increase hallucination recall** and lower tau to reduce false positives. The current deployment paragraph reverses this. Replace both corresponding examples.
- `files/abstract.tex`, `files/intro.tex`, `files/results.tex`, `files/conclusion.tex`: search for every `51`, `constant`, `invariant`, `negligible`, `lower bound`, and claim of production advantage; harmonize scope with the final timing evidence.
- Use LaTeX labels/references instead of assuming figure/table numbers remain unchanged after the new table.

## 7. Final source and PDF verification

Work from a copy of the supplied LaTeX folder. Use its existing compiler workflow; ordinarily:

```bash
pdflatex -interaction=nonstopmode -halt-on-error COLI_template.tex
bibtex COLI_template
pdflatex -interaction=nonstopmode -halt-on-error COLI_template.tex
pdflatex -interaction=nonstopmode -halt-on-error COLI_template.tex
```

Check for missing citations/references, overfull boxes, clipped legends, tiny captions, unmatched braces and float placement. Inspect the complete revised pages containing Sections 4.4 and 6.4, the new table, Figures 2–6 and 8, and the contribution list. Use the actually regenerated PDF as the submission file. This package does not include a compiled revised manuscript, because the experiment-dependent numbers remain pending and no LaTeX compiler is installed in this workspace.

## 8. Additional consistency edits identified in the audit

Section 3.2 should read:

> We use an HSIC-inspired score adapted from the unbiased estimator to remain computable for short token sequences. This modification introduces finite-sample bias; its definition and limitations are given in Section 4 and Appendix A. Population HSIC's independence guarantees should be distinguished from the empirical behavior of this adapted score on selected token representations.

For the asymptotic lemma in Appendix A.3, explicitly assume bounded kernels and i.i.d. **paired** observations from a fixed joint distribution. A compact valid proof route is to write A=Tr(K̃L̃), B=sum(K̃)sum(L̃), and C=1ᵀK̃L̃1. Their absolute values are O(n²), O(n⁴), and O(n³), respectively, for kernels bounded by one. The coefficient differences between the adapted and unbiased formulas are O(n⁻³), O(n⁻⁵), and O(n⁻⁴). Consequently the samplewise estimator difference is O(1/n); expectation and almost-sure convergence then follow from the unbiased estimator under the stated sampling assumptions. This replaces the unsupported approximate multiplicative relation. State that the token-selection heuristic does not supply i.i.d. paired sampling guarantees.

Finally, match the biased HSIC formula in Appendix A.2 to the comparator actually used for Figure 8: the code's `compute_hsic` uses 1/n², while the manuscript displays 1/(n−1)². Resolve this by documenting the measured convention or recomputing the comparator; do not imply they give identical rankings when realized n differs across examples.

## Dataset and baseline details found during repository consolidation

- RACE: the pinned grouped source contains 1,045 article records and 3,498 questions. Reconcile “1,050 passages” with the actual grouped-source convention used in the runs.
- The original main comparison has five training-free baselines plus HIDE. Correct prose saying six baselines unless it explicitly includes the separate probe.
- If replacing the main table, disclose raw-logit actual-token probabilities for MNLL/LN-Entropy, mean per-token energy at T=1, and the exact EigenScore covariance/log convention. New comparisons must share fresh greedy answers and correctness labels across methods.
- Report model dtype: the supported default is BF16, while the archived loader defaults to FP16. Do not label fresh generations as exact historical reproduction.
- For Table 8, state that unused top-k/top-p/temperature samplers are neutralized in the new experiment.
- For SVD, state exact thin SVD and compare adapted/centered estimators separately; the old path mixed estimator and selection changes.


## Confirmed symbolic-answer edge case

Describe the existing token fallback explicitly: “When keyword extraction yields no lexical candidates, including symbol-only answers, we select the first available tokens up to the token budget and equalize input/output counts.” The September 17 correction makes a custom-vectorizer empty-vocabulary exception reach that fallback; it does not discard such answers. For one selected token the adapted estimator is exactly zero, so a correct short/symbol answer can still be a detector false positive. Report count-stratified results and discuss this limitation if material. Retained complete pre-fix detection parts and rerun failed parts are distinguished in the saved source provenance; timing is measured entirely with the corrected implementation.
