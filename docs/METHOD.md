# Executable method specification

This document describes the supported code. The supplied manuscript still contains the submitted text and measurements; [AUDIT.md](AUDIT.md) identifies where it needs correction.

## Inputs and generation

- SQuAD: answerable development-v2.0 examples; first reference; `context + ' Q: ' + question + ' A:'`.
- RACE: high test questions, expanded from the original EleutherAI grouped-article source; options are included as text. Count is checked at 3,498.
- NQ and TriviaQA: zero-shot `Answer these questions:\nQ: ...\nA:`; no evidence context or hidden few-shot examples.
- Base and instruction-tuned checkpoints use these same original plain prompts. Applying a chat template is a separate experiment and is not done silently.
- Original punctuation/EOS/banned-token conventions remain in `hide/datasets/`. No tokenizer-independent `token_id > 2` rule is used.
- Default: greedy, batch size one, 256 new-token cap, eager attention, BF16 model weights; HIDE kernels use FP32. Original loaders defaulted to FP16, so BF16 reruns are a newly specified protocol, not guaranteed identical historical generations. Set `HIDE_DTYPE=float16` before starting a new output root if matching that precision is essential; keep it fixed across a suite.
- Temperature studies use T=0.3/0.6/0.9 with top-p=1 and top-k=0. Nucleus studies use p=0.7/0.8/0.9 with T=1 and top-k=0. These settings remove hidden sampler defaults and must be disclosed when replacing Table 8.
- Each example/repeat has a stable seed derived from seed 42, dataset, example ID and repeat. Sampling therefore resumes without depending on how many earlier examples were skipped. Bitwise reproducibility across different GPU/software stacks is not promised.

## Cached state alignment

For HF hidden-state index l, HIDE takes prompt states from prefill and available generated-token states from subsequent decoding steps. The final emitted token has not yet been forwarded; it is excluded from HIDE's output-state/token pairing. A generation with only one emitted token has no observed output state: the historical HIDE fallback is zero and is flagged in the record.

For mechanistic proxies, use attention block l-1 and its input hidden state l-1 over the same output query positions. Omega is mean attention mass to the whole prompt, averaged over heads and observed output tokens. Delta is the mean L2 norm of `mean_heads(attention_to_prompt) @ prompt_hidden_states[l-1]`.

Delta is an **unprojected attention-weighted input-state proxy**. It is not the actual projected residual write. Value/output projections, per-head mappings and Gemma normalization are not reconstructed. Report associations, not causal conclusions.

The all-layer ablation uses HF indices 1..L. Index L includes final model normalization; it is not interchangeable with an unnormalized final decoder-block output. Gemma local-window boundary cases that cannot be indexed safely are rejected, recorded, and require investigation. A known Transformers 4.51.3 cache-boundary limitation is detailed in AUDIT.md.

## Keyword selection and HIDE score

KeyBERT uses a resident all-MiniLM-L6-v2 encoder, unigram CountVectorizer with case preserved, MMR enabled and diversity=1. For each ranked keyword, the tokenizer matches both bare and leading-space spellings. All matched subtoken occurrences are appended in rank/match order, including duplicates. Missing matches use the first available tokens. Both sides are truncated to the same realized count up to k=20.

The resulting pairs are heuristic ordered token occurrences, not i.i.d. observations with a demonstrated joint distribution.

RBF Gram matrices are `exp(-1e-7 * cdist(X, X)^2)`. With zero-diagonal matrices A and B and count n, the retained adapted score is:

```text
[trace(A @ B) + sum(A)*sum(B)/n^2 - 2*sum(A @ B)/n] / n^2
```

It is an adapted, biased HSIC-inspired score, despite the archived function name `unbiased_HSIC`. For all-one kernels it equals `(n-1)/n^2`, not zero. Count and output-length controls are therefore included. A score below tau flags hallucination: increasing tau increases both recall and false positives.

The arithmetic is O(n^2 d + n^3) for the existing dense implementation. It is O(d) only when n is fixed. Keyword extraction and state capture also depend on text/model dimensions and implementation.

## Correctness targets and score orientation

- Sentence similarity: cosine similarity of generated answer and first reference using nli-roberta-large. Correct if strictly greater than 0.9.
- ROUGE-L: stemmed F-measure; correct if strictly greater than 0.5.
- Exact match: normalized lowercase/article/punctuation/whitespace removal against the first reference.
- Additional references/aliases are retained where available, but are not silently mixed into the historical first-reference labels.
- AUC treats correctness as positive. Confidence scores are oriented higher=more correct. Continuous PCC and binary PCC are distinct columns. No sign is selected to maximize test AUC.
- The evaluator includes all finite, successful common-cohort rows, retains finite zero HIDE scores, and lists exclusions. All-one labels/constant scores produce explicit undefined statistics where appropriate.
- Main intervals use 2,000 paired example-level percentile bootstrap draws. Passage-shared SQuAD/RACE questions are not resampled as clusters; this is a limitation, not an independence guarantee. Timing resamples entire queries, retaining their repeats.

## Baselines

All accuracy profiles save raw-logit MNLL/energy. The comparison profile also uses five stochastic samples with T=0.5, p=0.99 and k=10.

| Score | Definition / orientation |
|---|---|
| `negative_mnll` | Mean log probability of the **actual generated tokens** under unprocessed LM logits, including the terminal token |
| `negative_energy` | Mean logsumexp of raw logits; negative energy at T=1 |
| `negative_ln_entropy` | Negative mean of the five sampled sequences' length-normalized negative log likelihoods |
| `lexical_similarity` | Mean pairwise ROUGE-L over the five sampled texts |
| `negative_eigenscore_paper` | Negative mean natural log eigenvalue of `Z^T C_d Z + 1e-3 I`, matching the printed equation |
| `negative_eigenscore_cov_log10` | Separately labeled sensitivity using `(Z^T C_d Z)/(d-1)` and log10, corresponding to the old covariance convention |

Each sampled representation is the last forwarded generated-token state at the selected layer. If a sample has no such state, EigenScore is undefined; it is not imputed with zero. Centered Gram matrices, alpha and width are saved, allowing either eigen statistic to be recomputed without the full hidden vectors. The sampled answers' token IDs/log probabilities are also retained.

Sampling in the comparison runner is sequential to keep memory predictable. Its wall time **must not be presented as the paper's batched multipass latency benchmark**. The dedicated timing profile measures only base versus HIDE. No fresh 51% multipass reduction is produced here.

## Ablations

One ablation generation yields all 1..L layer scores; budgets 5/10/15/20/25/30; nine archived kernels; RBF gamma 1e-9/1e-7/1e-5/1e-3/1e-1/1; adapted and both centered estimator denominators; flattened cosine; and exact thin-SVD selection with adapted and n^2 centered scores.

The new exact-SVD calculation implements the manuscript's thin-SVD construction. The archived path used randomized `svd_lowrank` and changed the estimator simultaneously. Therefore the new factorial comparison is a corrected experiment, not a claim of historical parity. The `(n-1)^2` estimator is explicitly undefined at n=1. Unexpected variant failures prevent a complete export; known mathematical undefined cases are counted and reported.
