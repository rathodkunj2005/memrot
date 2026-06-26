# MEMROT next steps after reading "Open Problems in Mechanistic Interpretability"

Source: arXiv:2501.16496v1, "Open Problems in Mechanistic Interpretability"

Core takeaway: MEMROT should be framed less as "we found the mechanism" and more as a concrete model organism for one mechanistic-interpretability open problem: can internal explanations predict and control real retrieval failures under long-context pressure?

## The highest-leverage direction

Turn MEMROT into a validation-first mechanistic case study.

The Open Problems paper repeatedly warns that interpretability work often confuses hypotheses with conclusions. MEMROT is currently vulnerable to exactly that critique, especially around SAE feature selection and attention-mass interpretation.

So the next version should be built around validation:

1. Can the internal signal predict unseen failures?
2. Does the signal survive held-out feature/head selection?
3. Does intervention change behavior in the predicted direction?
4. Is the explanation better than black-box baselines?

## P0: fixes before posting/submitting widely

### 1. Reframe the claim

Current vibe: "mechanistic anatomy of context rot."

Better: "a model organism for validating mechanistic explanations of context rot."

Use the Open Problems framing:

- decomposition: retrieval heads + SAE latents
- description: these components route/recover gold evidence
- validation: held-out prediction + causal intervention

Suggested thesis:

"We use context rot as a model organism for testing whether mechanistic signals can predict and partially control behavioral retrieval failures."

That is stronger and safer than claiming a final mechanism.

### 2. Add held-out validation for SAE features

Current weakness: SAE recall features are selected by correlation with correctness, then used in H1/H2/H3. That is circular.

Fix:

- Split questions into discovery and evaluation sets, e.g. 70/70.
- Select SAE layers/features only on discovery questions.
- Freeze them.
- Report H1/H2/H3 only on held-out questions.
- Add controls:
  - random SAE features
  - anti-correlated features
  - activation-magnitude-matched random features
  - same-layer random features

Deliverable: `src/analyze_holdout.py` and a table comparing discovered features vs controls.

### 3. Normalize attention mass

Current weakness: absolute gold attention mass can fall just because context length grows.

Add metrics:

- gold_attention_enrichment = gold_attention_mass / gold_token_fraction
- gold_vs_random_span_attention
- gold_vs_position_matched_distractor_attention
- gold_vs_best_distractor_attention

Deliverable: new columns in `attn_mass.parquet` / analysis:

- `gold_token_frac`
- `attn_enrichment_uniform`
- `attn_random_span_mean`
- `attn_pos_matched_distractor_mean`
- `attn_gold_minus_control`

### 4. Model Gemma-2 sliding-window visibility directly

Current weakness: if gold falls outside the sliding window, some layers structurally cannot attend to it. That can mimic context rot.

Add:

- `gold_visible_to_sliding_layer` per layer/q/k
- separate analysis for global vs sliding layers
- H1/H2 using global layers only
- a controlled rerun where gold is always within sliding-window range

Deliverable: a section called "architectural visibility control."

### 5. Replace p-value flexing with robustness

The Open Problems paper emphasizes validation over appearances. Huge p-values do not buy trust here.

Add:

- bootstrap CIs over questions
- permutation tests shuffling signal curves across questions
- collapse threshold sensitivity: 0.3, 0.4, 0.5, 0.6, 0.7
- tie-aware test for discrete k grid
- compare against black-box predictors: k, context length, gold position, answer type

Deliverable: robustness table.

## P1: make MEMROT a stronger mechanistic-interpretability contribution

### 6. Turn H2 into prediction, not post-hoc correlation

Train/choose nothing on test questions. Then ask:

"Given internal signals up to k, can we predict whether the model will fail at the next fill level?"

Compare:

- black-box baseline: k, ctx_tokens, gold_pos_frac, question_type
- behavioral baseline: previous correctness only
- mechanistic model: black-box features + attention/SAE signals

If mechanistic signals improve held-out prediction, MEMROT directly answers the Open Problems application: using interpretability to predict behavior in novel situations.

### 7. Strengthen H3 as control, not just ablation

Current H3 says retrieval heads matter. Good, but not enough.

Add:

- multiple random-head control sets, not one
- layer-matched controls
- copy-score-matched but non-gold-attending controls if possible
- activation patching from k=0 into high-k at selected components
- negative controls: patch irrelevant question's activations
- rescue/harm tradeoff: does intervention hurt already-correct high-k answers?

Best causal test:

For the same question, patch retrieval-head outputs or residual/SAE features from k=0 into k=8/k=14. If answer recovers more than controls, the mechanism story becomes much more credible.

### 8. Validate the grader

Create a 100-example audit set:

- 25 correct k=0
- 25 flipped high-k
- 25 intervention recovered
- 25 disagreement/low-F1 edge cases

Have GPT-4/Claude judge plus manual review. Report agreement with substring/F1.

If the grader is noisy, rerun the headline stats on judged labels.

### 9. Include small auditable artifacts in the repo

Do not track huge H5/parquet tensors. Do track summary artifacts sufficient to reproduce paper numbers:

- `analysis/results_summary.json`
- `analysis/accuracy_by_k.csv`
- `analysis/h1_stats.csv`
- `analysis/h2_stats.json`
- `analysis/h3_stats.csv`
- `analysis/robustness_summary.json`

This solves the reproducibility critique without bloating the repo.

## P2: Substack angle

Use the Open Problems paper as the frame:

"Mechanistic interpretability has a validation problem. Context rot is a good toy-but-real failure mode to test whether internals actually predict behavior."

Suggested Substack structure:

1. Long-context models forget facts that are still in the prompt.
2. The mech-interp field says explanations need validation, not just pretty stories.
3. MEMROT treats context rot as a model organism.
4. I test three things: signal decay, signal-vs-flip prediction, causal poke.
5. The results are promising but not proof.
6. Next step: held-out validation and stronger controls.

This makes the post honest and research-forward.

## Concrete next 7-day plan

Day 1:
- Fix README/paper stale claims: model name, n=140, non-monotonic wording.
- Add tracked summary artifacts if available.

Day 2:
- Implement attention normalization controls.
- Rebuild figures/table with enrichment metrics.

Day 3:
- Implement discovery/eval split for SAE features.
- Add random/anti-correlated feature controls.

Day 4:
- Add H2 permutation + threshold sensitivity.
- Add bootstrap CIs.

Day 5:
- Add sliding-window visibility analysis.
- Separate global vs sliding layers.

Day 6:
- Run stronger H3 controls if GPU time is available.
- At minimum, multiple random-head controls.

Day 7:
- Rewrite paper and Substack around validation-first framing.
- Use title: "The answer was in the prompt. The model still forgot it."

## Bottom line

Do not chase another flashy result yet.

The next move is validation. If MEMROT can show that internal retrieval signals predict held-out context-rot failures better than black-box baselines, and that targeted interventions rescue failures better than controls, it becomes exactly the kind of application-oriented mechanistic interpretability work the Open Problems paper is asking for.
