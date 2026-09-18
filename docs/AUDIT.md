# Code-to-paper audit

## Current correction status (2026-09-16)

### Answer-boundary correction (2026-09-18)

The completed 242-part snapshot has been verified and analyzed in
`results/optimus-review-analysis-01`. It exposed frequent extra-question
continuations, especially for Gemma on NQ/TriviaQA. Full-output versus first-line
exact-match counts demonstrate a material evaluation difference; merely changing
the saved labels would leave detector states describing a different text span.

The opt-in `first-line` protocol now terminates generation at the first line
break after non-whitespace answer text, including multi-newline tokens. It
preserves the original score/selection code and remaining generation settings.
It records raw and evaluated text, checks that no scored state crosses the answer
boundary, and uses the same boundary criterion for base and HIDE latency.
New profiles, plans and result roots separate this protocol from the completed
legacy runs. The default historical protocol remains `legacy`; old scientific
fingerprints are not relabeled or approved for mixing with these results.
See [the run guide](ANSWER_BOUNDARY_RUNS.md). Tiny-model tests are complete;
the real-checkpoint first-line pilots must still run on the user's GPUs.

This correction does not resolve the near-equivalence to the token-count control
or establish an advantage over attention mass. Those comparisons remain fixed
in the analysis and must be reported even if the corrected generations retain
the same unfavorable pattern. Kernel parameters and score formulas are not
tuned using the evaluation labels.

### Symbol-only answer correction (2026-09-17)

A server diagnostic confirmed that NQ example `2720` generated ` *` (Gemma token `649`) for “what is the multiplication sign on the computer.” KeyBERT's custom CountVectorizer path raised `ValueError: empty vocabulary; perhaps the documents only contain stop words`. This prevented the existing no-keyword token fallback from running. `hide/core.py` now handles only that error when the analyzer also confirms no word candidates, then uses the existing token fallback and unchanged score formula. Other exceptions still fail. The generated symbol answer is retained and judged by the existing correctness protocol. For this one-token representation, `n_eff=1` gives an estimator value of zero; it does not establish that the answer is incorrect. Keep the count-stratified analyses and discuss short/symbolic-answer limitations if they materially affect detection.

Previously successful detection records are unaffected because the newly handled branch previously raised rather than returning a successful record. A pinned compatibility policy permits only the exact pre-fix source inventory, the audited new core hash, and changes to the parts/provenance handling needed for this upgrade. Complete detection parts retain their original raw bytes, manifests, source snapshots and execution records. The upgrade starts every incomplete part again, preserves its old bytes outside the active run tree, and requires timing to be remeasured. It does not permit arbitrary mixed code versions or rewrite old generation provenance as new code. Merged manifests identify their source inventory as the merge implementation and list generation sources separately for every original part.

The supported code is now `hide/`; original file names in the detailed historical findings below refer to `archive/original/`. Original numerical claims have not been replicated. Fresh protocols and raw results must accompany any replaced table/figure.

| Paper description / issue | Supported implementation | Status and consequence |
|---|---|---|
| Adapted score, RBF gamma and keyword ordering | `hide/core.py`, `hide/kernels.py` | Preserved; exact original-function parity tests, including duplicate occurrences |
| Single generation and prompt/output states | `hide/runner.py` | Final unforwarded token excluded; one-token fallback recorded; tiny Llama/Gemma alignment tests |
| Attention/norm mechanism | `hide/runner.py:proxies` | Aligned to HIDE layer and output positions; requires fresh Figure 2 and qualified proxy wording |
| First reference and zero-shot prompt | `hide/datasets/`, `hide/runner.py:labels` | Prompt/stop conventions retained; first answer explicitly returned; unused NQ train download removed; corrupt caches no longer silently replaced |
| Full evaluation populations | `hide/runner.py` | Counts checked before sampling; no score/class filtering; cohort fingerprints retained |
| Perplexity / LN-Entropy equation | `hide/baselines.py:token_statistics` | Actual generated-token probabilities from raw logits; fixes argmax-on-sampled-output bug; old rows must not be silently retained as this implementation |
| EigenScore equation | `hide/baselines.py:eigen_statistics` | Printed centered-Gram formula and old normalized-covariance/log10 convention both saved separately; state convention explicit |
| Lexical similarity | `hide/baselines.py:sampled_baselines` | Mean pairwise ROUGE-L over five stored samples; no batching-dependent truncation |
| Sampler robustness | `hide/runner.py`, `hide/config/experiments.json` | T-only/p-only sampling neutralizes unused settings; per-example seeds make stochastic resume reproducible |
| Estimator and selection ablations | `hide/ablations.py` | Both centered denominators plus adapted score; exact thin SVD with separate estimator factors; full saved variant records |
| Latency | `hide/runner.py`, `hide/summarize_timing.py` | Synchronized/warmed paired base-vs-HIDE measurements; not a reproduction of historical multipass/51% numbers |
| Results retention | `hide/provenance.py`, `hide/export_results.py`, `hide/recover.py` | Immediate durable records, locks, exact source snapshots, explicit recovery, suite validation and verified Git chunks |
| Probes, dialogue, optimized serving | Archive only | No matching probe training/checkpoints/splits or completed serving benchmark; not certified by the package tests |

The new default BF16 inference differs from the archived model loader's FP16 default. Precision is configurable and recorded; a newly generated table must use one stated protocol. Raw-logit probabilities also differ from the archived use of HF processed generation scores. These are disclosed protocol corrections, not guaranteed identical historical results.

**Additional scope correction:** the main baseline formulation/table contains five training-free baselines plus HIDE; prose saying six baselines must identify another method or be corrected. The new comparison runner implements those five, not a missing supervised probe.


### Fresh dataset validation

Read the [pinned original RACE test JSONL](https://huggingface.co/datasets/EleutherAI/race/tree/e30efe648089df42c548e93faa9c5f1816e2c44f): 1,045 grouped article objects and 3,498 questions; `problems` is a JSON list. The archived loader unconditionally applies `ast.literal_eval`, which fails for this list schema. The supported loader accepts lists or legacy serialized lists and pins/checks the raw source SHA-256. Reconcile the manuscript's 1,050-passage wording with this source's actual grouped-record convention.

## Scope and evidence

Reviewed the active generation paths, all scoring families, kernels, model/dataset loaders, evaluation/plot scripts, probe utilities, shell launchers, environment files, and notebook code. Inspected supporting legacy evaluation helpers and their imports/call sites. Read the full submitted PDF text and the manuscript sources; visually inspected submitted pages 3, 13, 17, 20, 23–25 (contributions; Figures 2–6 and 8). This is a static audit plus local synthetic/tiny-model verification, not replication of the paper's numerical results. Model checkpoints, GPU access and historical outputs are absent in this workspace.

The bundled `COLI_template.pdf` is a short template document, not the submitted 44-page paper. Use the supplied Downloads PDF as the submission record and `COLI_template.tex` plus `files/` as the editable manuscript. Do not infer correctness from the stale bundled PDF.

## Findings requiring attention for this revision

### 1. Mechanistic scores are filtered and class-rebalanced before PCC

`plot_mechanistic.py`, approximately lines 88–163:

- Removes every row with HIDE ≤ .001.
- Resamples correct/incorrect examples to approximately 40%/60%.
- Subsamples QCBW separately.
- Computes PCC on the resampled cohort.

This is outcome/score-dependent selection, not the full evaluation distribution. Its effect on correlations is unknown without the raw outputs. The supported plot/evaluator uses all valid paired rows and reports exclusions; finite zero scores are retained. If plotting a subsample for readability, compute and label statistics on the full cohort, but the supplied new plot does not subsample.

### 2. Layer and generated-token conventions differ

`func/metric.py:get_unbiased_hsic_score_keybert` uses `hidden_states[layer]` and output token IDs `[:-1]`. During ordinary cached generation, prefill returns prompt states, later steps return representations for the previously generated tokens, and the final generated token has not yet been forwarded. This exclusion is internally aligned; it should be documented, not casually “fixed” by adding a second pass.

`generate_mechanistic.py` instead computes attention from `forward_out.attentions[l_mid]`, weights `hidden_states[l_mid-1]`, and includes all output positions in a second teacher-forced pass. HF attention index `b` belongs to decoder block `b`, whose input is `hidden_states[b]` and output is `hidden_states[b+1]` for an internal block. Thus the old attention/previous-state pair is mismatched, and `--layer` affects HIDE while the proxies separately hard-code the midpoint. The new runner aligns HIDE index l with attention l−1 and input states l−1, and uses the same available output positions. This is a **corrected mechanistic protocol**, so regenerate Figure 2 rather than claiming exact legacy reproduction.

HF's generation output structure is documented in [Transformers 4.51.3 generation utilities](https://huggingface.co/docs/transformers/v4.51.3/en/internal/generation_utils). The architecture-specific normalization and projections are visible in the [Gemma2 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma2/modeling_gemma2.py).

### 3. Delta is a proxy, not the projected residual update

`generate_mechanistic.py` computes `mean_heads(attention) @ H_input`, then a mean L2 norm. It omits per-head value/output projections, grouped-query attention head mapping, layer normalization and Gemma's post-attention normalization. The paper acknowledges an approximation but repeatedly interprets it as the actual residual write. Label the measured quantity as an **unprojected attention-weighted input-state norm** and reserve Delta_projected for an actual projection-aware measure.

The new runner deliberately evaluates the published *type* of proxy, with corrected indices. It does not implement the projected residual update or claim that it does. A true projected-update baseline is optional and requires architecture-specific hooks, tests and explicit treatment of Gemma normalization. A negative/positive correlation of this proxy cannot establish that noisy residual writes cause hallucination.

### 4. Error-to-zero conversion can contaminate low-score detections

The old mechanistic runner replaces HIDE exceptions and NaNs in Omega/Delta with zero. Since zero HIDE is the strong hallucination end of the score, an implementation failure can become a confident scientific prediction. The old keyword extraction also catches every exception and falls back silently.

The core retains the legitimate documented one-generated-token fallback and normal no-keyword first-token fallback. Unexpected extraction errors now raise, and the runner writes an error record and stops. The evaluator uses one common valid cohort for comparisons and lists exclusions. Check failure counts before reporting any result.

### 5. Timing protocols and plotted quantities do not support the strongest claims

- `generate.py` and `generate_single_exp.py` use `time.time()` around asynchronous CUDA operations, with no explicit synchronization. Subsequent `.cpu()`/`.item()` may force synchronization outside the intended section.
- `extract_keyword_representation` creates KeyBERT on every example. This cost is inside `hsic_time`; auxiliary time is not HSIC arithmetic alone.
- The “base” greedy generation in the legacy runner already requests all hidden states and scores, so it is an instrumented generation baseline.
- `plot_scaling_time.py` removes overhead >200%, silently ignores missing fields, and plots realized selected-token count for one fixed-budget run. It is neither a controlled sequence-length sweep nor a model-scaling experiment.
- Figure 4's y-axis is auxiliary scoring time, but its caption/text call it complete pipeline latency.
- The Figure 3 plotting program and raw means/error-bar definitions are not present.
- `generate_vllm.py` is byte-identical to `generate.py`; it uses Transformers, not vLLM.
- The notebook contains an exploratory vLLM hidden-state example, not a completed latency study.

The new timing script warms resident models, alternates base/HIDE order, synchronizes CUDA, directly measures total time, saves per-query repeats and preserves all valid measurements. These are newly specified measurements; do not use them to invent the provenance of historical numbers.

### 6. Count dependence of the adapted score deserves a control

`unbiased_HSIC` matches the adapted estimator in the manuscript, despite its misleading historical function name. For K=L=J (all entries 1), the zero-diagonal matrices are J−I, and:

```
trace((J−I)^2) = n(n−1)
sum(J−I) = n(n−1)
sum((J−I)^2) = n(n−1)^2
HIDE = [n(n−1) + (n−1)^2 − 2(n−1)^2] / n²
     = (n−1)/n²
```

Examples: n=2 → .25; n=4 → .1875; n=20 → .0475. Thus, unlike population HSIC, the adapted finite-sample score need not approach zero for uninformative constant kernels. The RBF implementation defaults to gamma=1e-7, so inspect this empirically. The vertical score bands in the submitted figure motivate, but do not prove, the issue. The supplied control and count-stratified analyses let the new data determine its practical importance.

Kernel construction costs O(n²d), but `tK @ tL` is a dense O(n³) operation (computed twice). Fixed n permits an O(d) arithmetic statement. It does not make keyword processing or state capture independent of full sequence length.

### 7. Keyword matching is a multiset of ranked token occurrences

`extract_keyword_representation` tries both `keyword` and `" "+keyword`, keeps all matches/subtokens, and does not deduplicate matching indices. Tokens are ordered by keyword rank and matching order; paired input/output rows are not aligned by a semantic correspondence rule. It caps this list at k. Therefore realized count is determined by matched positions, not merely `min(k,input_length,output_length)` or number of unique token types. The core preserves this behavior and tests parity. Describe the actual procedure; deduplication would change results.

HSIC's formal independence interpretation assumes appropriately paired observations. Independently ranked/truncated token sequences supply a heuristic pairing. Retain “HSIC-inspired heuristic dependence score”; do not claim the experiment demonstrates joint-distribution factorization or a causal information-flow mechanism.

### 8. Correctness evaluation and reference answers

- The existing evaluation scripts correlate detector scores with **continuous** sentence similarity/ROUGE values, but the text calls them binary labels. AUC uses thresholds .9/.5. The new evaluator reports both continuous and binary PCC explicitly.
- Main generation saves `additional_answers=[]` for the four QA datasets. NQ's extra references and SQuAD alternatives are therefore discarded; TriviaQA canonicalization also drops aliases. Preserve the historical first-reference protocol for a paired supplemental comparison and disclose it. If evaluating any-reference exact match, reconstruct aliases from source data and apply it to **all** detectors; do not mix label protocols.
- Similarity/ROUGE labels are automatic correctness proxies, not independent factuality annotations. This is especially relevant to the mechanistic interpretation.

### 9. Baselines and legacy paths needing care before a full rerun

- `getAvgBertScore` adds zero inside its loop and always returns 1. This is a stub, not a functioning SelfCheck/BERTScore baseline. It is not one of the six methods in Figure 3, but do not reuse it as evidence.
- `get_lenghthNormalized_entropy` takes maximum token probability at each step rather than the sampled token's probability; this is not automatically the stated sample sequence log-likelihood. Audit it against the exact claimed baseline before rerunning.
- `get_num_tokens` treats every ID >2 as a non-special token, which is not valid across Llama/Gemma tokenizers.
- `getEigenIndicator_v0` indexes `hidden_states[num_tokens[ind]-2]` and can reach prefill/negative indices on short outputs; it allocates on generic `cuda` rather than the selected device. Check reference implementation and EOS conventions before using it in a revised comparative result.
- When multiple generation chunks are used, only the last chunk's `multiple_generation_time`, entropy and eigen quantities are retained. Default batch=5 often avoids that path, but memory-driven chunking can expose it.
- `generate.py` accepts decoding strings other than greedy but does not populate required variables for most of them. The single-experiment file supports more regimes, but temperature/top-p defaults can inherit HF top-k settings. Explicitly neutralize unused samplers if rerunning that ablation.
- `func/eval_ablations.py` imports nonexistent `.umwp_eval`; older `dataeval/load*`/OpenAI helper paths require undeclared packages/older OpenAI APIs and include unrelated benchmark assumptions. These paths are not required for the new study.
- Probe configuration maps Llama-3-8B to a `llama3_1_8b_linear` probe and applies the same base probe to instruct variants. Verify model compatibility, label orientation, dataset provenance and “supervised comparison” wording; no probe training code/splits are supplied here.
- Hard-coded CUDA visibility/devices/server paths, broad imports, and implicit model loads make ad hoc execution fragile. The supported workflow avoids these legacy imports and accepts explicit paths/devices. LN-Entropy and EigenScore now have explicit corrected implementations and tests; the archive remains unchanged.

These findings are not proof that all published baseline numbers are wrong. They show why a broad last-minute rerun through the current scripts requires additional validation. They are recorded separately from the minimal reviewer-facing revision.

### 10. Additional manuscript corrections

- In Section 6.5, the proposed threshold direction is reversed: if hallucination is flagged when score < tau, **raising** tau increases hallucination recall and false positives; lowering tau reduces flags. Correct both deployment examples.
- Figure 6's stability across layers does not establish a globally distributed causal mechanism.
- Figure 8's score magnitude alone cannot explain AUC; AUC is unchanged by strictly monotonic rescaling. Replace the claim that small scores make thresholding inherently unreliable with the observed discrimination comparison. Numerical precision is a separate claim requiring evidence.
- “Structural collapse,” “isolates causal operations,” “factorizes the joint distribution,” and “knowledge circuit weakens” exceed an observational correlation study. Rephrase consistently, not just the single magnitude/precision sentence.
- The reported typographical issues are already absent in the supplied version; acknowledge verification rather than falsely claiming to have made edits.
- Modern vLLM documents a [hidden-state extraction feature](https://docs.vllm.ai/en/v0.22.0/features/speculative_decoding/extract_hidden_states/). Avoid timeless claims that hidden-state extraction is impossible. No optimized-serving HIDE latency is established by that feature or the notebook.

## Local verification and known limits

The test suite checks the original formula and token selection against extracted original function definitions, constant-kernel behavior, AUC agreement with sklearn including ties, paired bootstrap deltas, zero-score retention, missing-data cohorts, normalization, and timing arithmetic. Random tiny Llama/Gemma models test cached/replayed hidden-state and attention alignment before the cache boundary.

A targeted tiny-model test found that Transformers 4.51.3 Gemma HybridCache can differ from full replay at the allocated sliding-cache boundary. The implementation's shift condition is `cache_position >= max_cache_len - 1`. The main runner records generation-cap hits, and rejects local-attention windows whose absolute prompt indexing it cannot safely handle. Do not claim cached/replayed equivalence for every length/configuration based on the short tests. This does not itself show an error in any specific submitted example, for which outputs are unavailable.

Large checkpoint loading, tokenization with real model files, dataset downloads, BF16/CUDA numerical behavior, CPU keyword/judge throughput and A100 memory must still be checked by the supplied pilots. No local test is a replacement for them.

### 11. Theoretical wording and biased-comparator convention

`files/preliminary.tex` Section 3.2 still says the method uses an unbiased estimator reliable for small samples, although Section 4 correctly calls HIDE an adapted biased heuristic. Harmonize these descriptions. In `files/proofs.tex`, asymptotic claims need an explicit i.i.d. paired-sample assumption. The current approximate multiplicative relation does not prove them; bound the difference between estimators term-by-term instead. These population-statistical guarantees do not automatically apply to adaptively selected, independently ordered token occurrences.

The appendix's biased estimator divides by `(n−1)^2`, whereas `func/metric.py:compute_hsic` divides by `n²`. Both conventions occur for biased HSIC-type statistics, but they are different when n varies and can change cross-example rankings. Before attributing Figure 8 differences entirely to estimator design, identify which convention produced its plotted numbers and state that exact convention. Changing the comparator now requires a paired recomputation; do not silently rename the existing results. The proposed Figure 8 caption deliberately avoids a causal explanation for the AUC difference.
