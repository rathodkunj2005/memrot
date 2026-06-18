# MEMROT v2 — Mechanistic Anatomy of Memory-Retrieval Failure as Context Fills

Gemma-2-2B-it (8k window) + Gemma Scope SAEs + LongMemEval-S material, on CHPC/SLURM.
Implements the v2 agent execution plan: controlled haystacks at fill levels
k ∈ {0,1,2,4,8,14}, paired within-question design, gold-position randomization,
in-hook attention reduction, empirical SAE layer scan, mean-ablation + intervention.

> **Status (2026-06-17):** the 2B run is complete and the paper
> (`paper/paper.tex` → `paper/paper.pdf`) is written. The paper is framed as a
> **single-model, single-seed case study**; its Limitations section catalogs known
> open confounds (in-sample SAE feature selection, unnormalized attention,
> sliding-window visibility, H2 shared-trend, heuristic grader) that the next round
> addresses **before** the planned 9B/27B scaling. Do not treat the current numbers
> as a finished mechanistic result.

## One-time human prerequisite (before Run 1)
The instruct model `google/gemma-2-2b-it` (and `-9b-it`/`-27b-it` for the scaling
legs) is a **gated** HF model:

1. Accept the license at https://huggingface.co/google/gemma-2-2b-it (instant).
2. On a login node: `pip install -U "huggingface_hub[cli]" --user && HF_HOME=$SCRATCH/hf_cache huggingface-cli login`
   (or simply write the token to `$SCRATCH/hf_cache/token`).

Run 1 checks this first and fails with instructions if missing. Gemma Scope and
LongMemEval are ungated.

## The three runs
```bash
cd ~/memrot
sbatch sbatch/run1_experiment.sbatch   # heavy GPU (~40min on A100): smoke -> prep -> layer scan -> head ID -> 840-pass sweep -> verify
# read $SCRATCH/memrot/artifacts/run1_verify.json, then:
sbatch sbatch/run2_analyze.sbatch      # light GPU (~10min): H1/H2 stats, H3 ablation+intervention, figures
# read $SCRATCH/memrot/analysis/results_summary.json, then:
sbatch sbatch/run3_report.sbatch       # CPU (~minutes): markdown report; the canonical paper is hand-built paper/paper.tex -> paper.pdf via pdflatex (x2)
```
The sweep is **140 questions × 6 k-levels = 840 passes** (the `single-session-preference`
question type is excluded — its gold answers are prose rubrics the heuristic grader
cannot score — leaving 140 of the 170 exactly-one-gold LongMemEval-S questions).
Each sbatch script bootstraps its own conda env idempotently
(`env/setup_env.sh` → Miniforge at `$SCRATCH/memrot/miniforge3`; the old
`~/miniconda3` was scratch-purged and is unusable). Every stage is resumable;
re-submitting a failed job continues where it stopped. `STATUS: <stage> <PASS|FAIL>`
lines in `logs/runN_*.out` are machine-parseable.

## Model-scaling study (2B → 9B → 27B)
Artifacts are **namespaced per model** by `$MEMROT_MODEL_ID` (`src/paths.py`), so the
three models coexist under `$SCRATCH/memrot/artifacts/<id>/` and never clobber each
other. Per-model configs (`config/config_{2b,9b,27b}.yaml`) carry only the model,
its Gemma Scope SAE repo, and the namespace; everything else inherits the shared
base via `extends:`.

```bash
# one-time: accept the HF license for each instruct model you will run
#   https://huggingface.co/google/gemma-2-9b-it     https://huggingface.co/google/gemma-2-27b-it

bash sbatch/submit_model.sh 9b     # chains experiment -> analyze -> report (afterok), A100 40GB
bash sbatch/submit_model.sh 27b    # 80GB A100 on the preemptible guest partition (--requeue; resumable)
sbatch  sbatch/run_compare.sbatch  # CPU: cross-model scaling_summary.json + fig5..fig8
```
**27B caveat:** Gemma Scope ships 27B residual SAEs for only **3 layers (10, 22, 34)**,
so 27B's SAE-feature results (H1-feature, H3b) are restricted to those layers; its
attention-based results (H1-attn, H2, H3a) are fully comparable. The headline
scaling quantity is attention-mass `β_k`, which is available for every model.

> The current paper is **2B only** and framed as a case study; the 9B/27B legs and
> the scaling rewrite are pending (and gated on the methodology fixes in the paper's
> Limitations — scaling a confounded method is not worth the compute).

## Artifact contract (schema_version 2; per-model namespace)
```
$SCRATCH/memrot/artifacts/<id>/        # <id> = 2b | 9b | 27b
  longmemeval_records.parquet   longmemeval_sessions.parquet
  layer_scan.json               retrieval_heads.json
  attn_mass.parquet             # (qid,k,layer,head,attn_mass_gold,gold_pos_frac,ctx_tokens)
  sae_acts.h5                   # /q/<qid>/k<k>/layer<L> -> feat_ids[64], acts[64]
  answers.parquet               # (qid,k,pred,gold,f1,correct,error_type,gold_pos_frac,ctx_tokens)
  run1_manifest.json            run1_verify.json   figures/   # fig1..fig4 pdf
$SCRATCH/memrot/analysis/<id>/         # Run 2 outputs incl. results_summary.json
$SCRATCH/memrot/analysis/compare/      # scaling_summary.json (cross-model)
$SCRATCH/memrot/artifacts/compare/figures/   # fig5..fig8 pdf (scaling)
$SCRATCH/memrot/data/                  # shared raw LongMemEval download (not namespaced)
results/<id>/                          # small summary JSONs tracked in-repo (auditable)
paper/paper.tex -> paper/paper.pdf     # canonical paper (pdflatex x2); paper.md is the Run-3 draft
```
Runs 2/3 hard-fail on `schema_version` mismatch.

## Cluster facts discovered (2026-06-09), baked into sbatch
- GPU: account/partition `notchpeak-gpu`; bf16-capable GPUs there are
  3090 (notch293, notch328) and A100 (notch293) → `--constraint="3090|a100"`.
  V100/2080Ti/P40 lack bf16 (Gemma-2 in fp16 overflows; bf16 is required).
- CPU: account/partition `notchpeak-shared-short` for Run 3.
- `$SCRATCH=/scratch/general/vast/u1497420`; `HF_HOME=$SCRATCH/hf_cache`.
- pip torch==2.4.1 bundles its CUDA runtime → no `module load cuda` needed.

## Deviations from the plan document (engineering judgment, all logged)
1. **No sae_lens/transformer_lens.** Gemma Scope JumpReLU SAEs are loaded
   directly from `google/gemma-scope-2b-pt-res` `params.npz` (`src/sae_loader.py`,
   ~30 lines), avoiding that dependency stack. Per layer we use width_16k with
   average_l0 nearest `sae.target_l0=100` (canonical convention); chosen l0s are
   recorded in `layer_scan.json` and the manifest.
2. **Distractor pool.** Distractors come from kept questions' own non-gold
   LongMemEval-S haystack sessions — these are the benchmark's topic-matched
   filler drawn from other histories, i.e. the plan's "other questions' sessions,
   topic-matched where possible". `haystack.hard_distractors: true` restricts to
   the same question's haystack only (hardest mode, the de-risk fallback).
3. **H3b intervention** implements SAE feature re-injection (primary,
   `analysis.intervention_mode: feature`) and retrieval-head output amplification
   (`head` mode) rather than attention-logit reweighting, which would require
   patching eager attention internals. H3a additionally runs a matched
   random-head ablation control.
4. **Attention mass query rows**: reduced over the answer-emission positions
   (last prompt token through end of generated answer) in a single instrumented
   forward over prompt+answer — strictly more informative than last-prompt-token
   only, same memory contract.
5. **n_questions = 140 (not 300, not 170).** Measured on LongMemEval-S
   (2026-06-09): only 170 non-abstention questions have exactly one gold evidence
   session; the plan's 300 does not exist in the data. Of those 170 we then
   **exclude the 30 `single-session-preference` questions** (their gold answers are
   prose preference-rubrics the heuristic substring/F1 grader cannot score),
   leaving **140** (64 single-session-user + 56 single-session-assistant + 20
   temporal). Evidence sessions are also long (median ~2.5k Gemma tokens), so
   instead of the plan's hard ≤2k filter (which would keep ~50 questions) long
   gold sessions are **trimmed** to a turn window centered on the `has_answer`
   evidence turn(s) (`gold_trimmed` flag in records — so this is a controlled
   LongMemEval-S *variant*, see paper Limitations). Sweep is therefore
   **140×6 = 840 passes** (~40 min on A100 for 2B).

## De-risk fallbacks (from the plan; decision thresholds unchanged)
- **k=0 accuracy < 60%** (smoke or verify): inspect preds in
  `smoke_artifacts/answers.jsonl`. If the base model rambles, set
  `model.name: google/gemma-2-2b-it` in config (pt SAEs transfer; note in paper)
  and resubmit Run 1 with `--force` semantics (delete `$SCRATCH/memrot/artifacts`).
- **Pace > 20s/pass**: set `benchmark.n_questions: 200` and drop k=2 from
  `k_levels`; resubmit (completed (qid,k) pairs are reused).
- **H3 head-ablation null**: pivot headline to the SAE-feature explanation —
  Run 1 artifacts already contain everything; only Run 2 emphasis changes.
- **Distractors too easy** (accuracy barely drops by k=14): set
  `haystack.hard_distractors: true`, delete only the sweep artifacts
  (answers/attn_mass/sae_acts), resubmit Run 1 — stages 1–4 are cached.

## Determinism
Seed 0; greedy decoding; per-(qid,k) distractor RNG seeded by
sha256(qid) (process-stable replacement for the plan's salted `hash(qid)`);
questions in sorted-qid order; `CUBLAS_WORKSPACE_CONFIG=:4096:8` +
`torch.use_deterministic_algorithms(warn_only=True)`; pinned pyarrow so a
re-executed Run 1 reproduces `answers.parquet` byte-identically.
