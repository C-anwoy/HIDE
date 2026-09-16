# Validation status — 2026-09-16

## Completed

- **22 automated tests passed.** Exact output: [validation_tests.log](validation_tests.log).
- HIDE formula, FP32 score and keyword/token ordering checked against extracted original functions, including duplicate occurrences and the one-token fallback.
- Constant-kernel formula, both centered estimator denominators, actual-token likelihoods and the printed EigenScore matrix formula checked numerically.
- Cached/replayed state and attention alignment checked with random tiny Llama and Gemma models before the documented cache boundary.
- Detection, timing, all score ablations, five-baseline comparison wiring, nucleus sampling and resume exercised using real tiny Transformers generation with fake data/keyword/judge services. Interrupted stochastic runs reproduce later token IDs/scores.
- Original prompt and stopping conventions compared; SQuAD first-reference mapping and both RACE question-field schemas checked with offline fixtures.
- AUC verified against sklearn with ties; paired bootstrap, common-cohort exclusions, finite-zero retention and timing arithmetic checked.
- Single-writer locks, explicit recovery with exact backups, corruption detection, lossless chunk export/restoration and declared-suite checks tested.
- All named suite commands resolve and parse; primary sample counts remain zero (full dataset). Python compilation, Bash syntax and diff whitespace checks passed.
- The package built as a wheel, installed into a temporary target and resolved a CLI command from outside the repository, including packaged model/experiment configuration.
- Analysis → export → restoration passed on synthetic QA/comparison/ablation/timing fixtures. Mechanistic, timing and scaling plot layout was inspected; final plotting also emits PDF files. Synthetic results are not paper evidence and are not stored in the repository.
- The pinned public RACE source was fetched and inspected: 1,045 grouped articles, 3,498 questions, list-valued `problems`, SHA-256 `e0ff122cd99f1802cec824b824d576436b4c19c313744c7e1fe3f60c6b17f9d2`.
- 65 archived tracked files match their original Git versions byte-for-byte. The archived `_settings.py` preserves the environment overrides already made earlier in this task. The original external LaTeX directory was not modified; copied manuscript files retain source hashes and all literal includes resolve.

Tests used Python 3.12 with isolated temporary dependencies including Torch 2.5.1 and Transformers 4.51.3. No working research environment was overwritten. A GitHub Actions workflow is included; a remote CI run has not been performed from this task.

## Still requires the A100 environment

Real model/tokenizer files, full dataset/cache behavior, BF16/CUDA numerical behavior, stopping/cap rates, memory consumption and throughput. Run pilots before long jobs. No real-model AUC/PCC, 27B performance or latency measurement is claimed by the package tests.

The main comparison implements the five described training-free baselines with explicit corrected conventions. This does not certify missing supervised-probe training artifacts, dialogue experiment provenance or a batched/optimized-serving multipass benchmark.

No revised manuscript PDF was compiled: TeX/latexmk is not available locally. Original numeric tables/figures remain unchanged. Proposed manuscript changes and the response template contain explicit result-dependent placeholders.
