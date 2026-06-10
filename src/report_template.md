# Mechanistic Anatomy of Memory-Retrieval Failure as Context Fills

## Abstract
Long-context models lose access to stored facts well before their context window
is exhausted ("context rot"), but the mechanism is unobserved in correlational
reports. Using Gemma-2-2B within its native 8k window, we build controlled
haystacks from LongMemEval-S material: the same question is evaluated at every
fill level k ∈ {{k_levels}} distractor sessions, with the gold session's position
randomized and logged. On {{n_questions}} questions (accuracy by k:
{{acc_by_k}}), retrieval-head attention mass on the gold span and
recall-associated SAE feature activations decline with k (mixed-effects
β_k = {{collapse_beta}}, p = {{collapse_beta_p}}; feature β_k = {{feature_beta}},
p = {{feature_beta_p}}), controlling for gold position. Within questions, the
fill level at which the internal signal collapses predicts the level at which
the answer flips (Spearman ρ = {{rho_flip_collapse}}, p = {{rho_p}},
n = {{n_paired}}; Cox HR per unit signal-AUC = {{cox_hr}}, p = {{cox_p}}).
Causally, {{ablation_mode}}-ablating the {{n_retrieval_heads}} identified
retrieval heads at k=0 drops accuracy by {{ablation_drop}} (random-head control:
{{ablation_drop_random}}), and {{intervention_mode}}-mode intervention at high k
recovers {{recovery_frac}} (95% CI [{{recovery_ci_lo}}, {{recovery_ci_hi}}]) of
{{n_flipped}} flipped answers.

## 1. Introduction
Context rot — degradation of retrieval as the context fills, well inside the
model's window — has been documented behaviorally. Retrieval heads and sparse
autoencoder (SAE) features give us instruments to watch the failure happen. We
ask three questions: does the internal retrieval machinery degrade as context
fills (H1)? Does its per-question collapse point predict the per-question
behavioral flip point (H2)? Is the machinery causally responsible (H3)?

## 2. Method: controlled-haystack paired design
The novelty over correlational context-rot reports is the **paired
within-question design**: each question is its own control. From LongMemEval-S
we keep questions with exactly one gold evidence session (≤ 2k tokens). For each
question and each k we pack the gold session plus k topic-matched distractor
sessions (truncated to a fixed per-distractor cap so context grows ~linearly
with k) into ≤ 7,500 tokens, placing the gold session at a uniformly random,
logged position. A question enters analysis only if it is answered correctly at
k=0, so failures at k>0 are attributable to context fill, not knowledge gaps.
Retrieval heads are identified by copy-score probes built from benchmark
vocabulary; SAE layers ({{sae_layers}}) are chosen by an empirical separability
scan over all 26 Gemma Scope layers. Decoding is greedy and the full pipeline is
seeded and re-runnable byte-identically.

![Collapse curves](figures/fig1_collapse_curves.pdf)

## 3. Results
**H1 (descriptive).** Retrieval-head attention mass on the gold span declines
with k (β_k = {{collapse_beta}}, p = {{collapse_beta_p}}, mixed-effects with
question as random intercept and gold position as covariate; gold-position
β = {{gold_pos_beta}}). Recall-feature activation shows the same decline
(β_k = {{feature_beta}}, p = {{feature_beta_p}}). Fig. 1, Fig. 4.

**H2 (paired, headline).** Among the {{n_k0_correct}} questions correct at k=0,
per-question signal-collapse level predicts answer-flip level:
ρ = {{rho_flip_collapse}} (p = {{rho_p}}, n = {{n_paired}} with both events
observed); the survival model over all questions (censoring included) gives a
hazard ratio of {{cox_hr}} (p = {{cox_p}}) per unit normalized signal-AUC.
Feature-signal variant: ρ = {{rho_flip_collapse_feat}}. Fig. 2.

**H3 (causal).** {{ablation_mode}}-ablation of retrieval heads at k=0 on
{{ablation_n}} previously-correct questions drops accuracy by {{ablation_drop}}
versus {{ablation_drop_random}} for matched random heads. Intervening at high k
({{intervention_mode}} mode) recovers {{recovery_frac}}
(95% CI [{{recovery_ci_lo}}, {{recovery_ci_hi}}]) of {{n_flipped}} flipped
answers. Fig. 3.

## 4. Limitations
2B base model, single benchmark source, eager-attention requirement (no
flash-attention numerics), heuristic (non-LLM-judge) grading, and distractor
construction choices (fixed per-distractor token cap; pool drawn from the
benchmark's own topic-matched filler sessions). Gemma-2 interleaves
sliding-window and global attention layers; gold spans beyond the 4,096-token
window are structurally invisible to sliding layers, which we log but do not
model separately.

## 5. Related work
Context rot (Chroma); lost-in-the-middle (Liu et al.); retrieval heads (Wu et
al.); Gemma Scope SAEs (Lieberum et al.); LongMemEval (Wu et al.).

## 6. Reproducibility
Seed {{seed}}, git {{git_sha}}, env hash {{env_hash}}, artifact schema v
{{schema_version}}. Three SLURM jobs reproduce everything; Run 1 re-execution is
byte-identical on answers.parquet.
