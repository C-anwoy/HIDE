# Response to action editor and Reviewer 1 — working draft

**Not ready to submit.** The following is a response scaffold. Replace every bracketed field after completing the corresponding work. Do not claim an experiment or edit has been completed until it has. Use final section/table/figure numbers after compilation. The current plan includes a Gemma-2-27B timing experiment on the confirmed A100 80GB. Use the larger-model paragraph only after that experiment is actually complete; retain narrow scaling claims in either case.

## Opening

We thank the action editor and Reviewer 1 for their careful reading and constructive suggestions. We have revised the efficiency accounting, qualified the mechanistic interpretation, and [added the completed structural-baseline comparison]. Below we address each point and identify the corresponding changes in the revised manuscript.

## Action editor

### 1. Inconsistent latency descriptions in Section 6.4 and Figure 4

We agree that the previous presentation mixed auxiliary scoring time with complete pipeline latency and reported incompatible overhead values. [Describe the actual reconciliation: a fresh timing substudy or recovered original records.] We now define base generation, generation with hidden-state capture, auxiliary scoring, and total HIDE latency separately. For [model/dataset/cohort], base generation is [X] seconds and total HIDE latency is [Y] seconds, an overhead of [Y−X] seconds ([100(Y−X)/X]%). Section 6.4, [table], and the figure now use the same saved measurements and aggregation. Figure 4 is labeled as [the actual descriptive/controlled analysis] and includes [all valid measurements, with any failures accounted for].

### 2. Rephrase Section 4.4 and explicitly discuss Gemma-2-9B

We have qualified the interpretation as model- and dataset-dependent. The revised main text reports the update-norm correlations for both Llama-3-8B ([SQuAD r], [NQ r]) and Gemma-2-9B ([SQuAD r], [NQ r]), including the positive Gemma correlations [if confirmed in the regenerated analysis]. We no longer present an inverse magnitude–coupling relationship as a universal mechanism. We also clarify that the measured update norm is an unprojected attention-weighted input-state proxy, and that this analysis establishes associations rather than a causal account of hallucination. [Explain corrected layer/token conventions and regenerated Figure 2 if implemented.]

### 3. Add a larger model or soften the scalability claim

**Scope qualification (retain with either outcome):** We have narrowed the claim to match the empirical scope. We retain the fixed-token arithmetic dependence on hidden width and explicitly distinguish it from parameter count, sequence-length effects and end-to-end latency. We no longer claim constant absolute overhead regardless of model size or scalability to substantially larger architectures on the basis of two nearby widths. The revised text states the models actually measured and the limits of that evidence.

**Planned addition, only after the larger-model timing study completes:** We additionally measured [larger model, parameter count, actual hidden width] under the same [hardware/dtype/backend/prompt/repetition] protocol. Results appear in [table/figure]. We retain a qualified claim, because these measurements cover a limited set of architectures and do not establish constant overhead at arbitrary scales.

### 4. Add standalone attention-mass and update-norm baselines

We have added [table] comparing HIDE, prompt-attention mass and the unprojected input-state update-norm proxy on identical generated answers from [models] on [all datasets]. We report [AUC/PCC targets and confidence intervals] and discuss the closed-book results on both NQ and TriviaQA explicitly. [Summarize actual gains, ties and losses.] The text distinguishes correlation with HIDE from predictive discrimination of correctness. [Mention negative-norm orientation sensitivity and count-based controls if included.]

## Reviewer 1

We appreciate the positive assessment of the contribution and the detailed suggestions for strengthening the final presentation.

### Weakness 1 / encouraged substantive revision: necessity of HSIC versus cheap attention proxies

Please see [new table] and Section [X]. We evaluate the suggested quantities as standalone scores on a common cohort, rather than inferring detection performance from their correlation with HIDE. [Give per-dataset/model findings, with emphasis on NQ and TriviaQA; do not summarize only the favorable cells.] In [settings where a proxy ties or wins], we explicitly acknowledge that HIDE does not establish an advantage. We interpret the new evidence as [appropriately limited conclusion].

[If full-cohort protocol changed: We also removed score-based filtering and class rebalancing from the reported mechanistic correlations and aligned the attention block and output-token positions with the hidden states used by HIDE. We regenerated the associated figure under this stated protocol.]

### Weakness 2 / required minor revision 2: magnitude versus precision is not universal

We agree. Section 4.4 now includes both models' update-norm results, and the Figure 2 caption no longer characterizes all correlations as weak or negative. We describe the magnitude-versus-precision account as a possible interpretation in some model–dataset settings, not as an established general law. The revised text also distinguishes the measured proxy from the projected residual-stream update.

### Weakness 3 / required minor revision 1: incompatible latency numbers

We have reconciled these numbers using [actual source/protocol]. The revised accounting is [base X; total Y; overhead Y−X; percentage], and the figure and response use the same values. We distinguish total request latency from auxiliary HIDE scoring latency and specify [warmup, device synchronization, resident keyword model, repetitions and averaging]. We do not attribute the old discrepancy to an unverified cause.

### Encouraged minor revision 1: consistent efficiency framing

We now restrict efficiency statements to the measured standard Hugging Face setup. [If retaining the historical result: The approximately 51% value is explicitly identified as the original five-generation baseline comparison with its verified aggregation.] [If not retaining it: We have removed the unreconciled percentage claim and replaced it with the newly measured, explicitly scoped latency result.] We do not present a hypothetical optimized-serving advantage as an empirical result or a measured lower bound. The serving-engine discussion is now a limited engineering consideration.

### Encouraged minor revision 2: scaling evidence

[Use the selected response to AE point 3.] We also distinguish linear dependence of the fixed-token score calculation on hidden width from model parameter count and total inference time.

### Typographical point 1: “actuality”

We checked the supplied manuscript and final revised PDF: Section 6.2 reads “factuality hallucinations.” [Verify final build before sending.]

### Typographical point 2: “perfromance”

We checked that the results introduction reads “performance” and [if made] corrected the associated subject–verb agreement.

### Typographical point 3: “repeatation”

We checked that Section 8.2 reads “verbatim repetition.”

### Typographical point 4: empty contribution bullet

We verified the three-item contribution list in the compiled PDF and found no empty item. [Confirm after the final edit/compile.]

### Typographical point 5: self-contained captions

We revised/verified the captions of Figures 3, 5, 6 and 8 to state the models, datasets, evaluation quantities and relevant aggregation or fixed settings. Figure 3 defines [actual error bars and averaging]; Figures 5, 6 and 8 define AUC_s and the compared configurations. We removed mechanistic conclusions that the plotted comparisons alone do not establish.

## Pre-submission cross-check

- Replace every bracket, placeholder, future-tense commitment and alternative paragraph.
- Ensure every “we have” statement describes completed work.
- Verify all correlation/latency numbers against the final saved tables.
- Include negative results and exact sample counts.
- Verify final figure/table numbers and page references.
- Keep the response focused on evidence and corrections; acceptance is the editor's decision.
