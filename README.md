# MEMROT v2 — Mechanistic Anatomy of Memory-Retrieval Failure as Context Fills

Gemma-2-2B (8k window) + Gemma Scope SAEs + LongMemEval-S material, on CHPC/SLURM.
Implements the v2 agent execution plan: controlled haystacks at fill levels
k ∈ {0,1,2,4,8,14}, paired within-question design, gold-position randomization,
in-hook attention reduction, empirical SAE layer scan, mean-ablation + intervention.

## One-time human prerequisite (before Run 1)
`google/gemma-2-2b` is a **gated** HF model and no token is currently installed:

1. Accept the license at https://huggingface.co/google/gemma-2-2b (instant).
2. On a login node: `pip install -U "huggingface_hub[cli]" --user && HF_HOME=$SCRATCH/hf_cache huggingface-cli login`
   (or simply write the token to `$SCRATCH/hf_cache/token`).

Run 1 checks this first and fails with instructions if missing. Gemma Scope and
LongMemEval are ungated.

## The three runs
```bash
cd ~/memrot
sbatch sbatch/run1_experiment.sbatch   # heavy GPU (~6h): smoke -> prep -> layer scan -> head ID -> 1800-pass sweep -> verify
# read $SCRATCH/memrot/artifacts/run1_verify.json, then:
sbatch sbatch/run2_analyze.sbatch      # light GPU (~2h): H1/H2 stats, H3 ablation+intervention, figures
# read $SCRATCH/memrot/analysis/results_summary.json, then:
sbatch sbatch/run3_report.sbatch       # CPU (~minutes): paper/paper.md (+pdf if pandoc)
```
Each sbatch script bootstraps its own conda env idempotently
(`env/setup_env.sh` → Miniforge at `$SCRATCH/memrot/miniforge3`; the old
`~/miniconda3` was scratch-purged and is unusable). Every stage is resumable;
re-submitting a failed job continues where it stopped. `STATUS: <stage> <PASS|FAIL>`
lines in `logs/runN_*.out` are machine-parseable.

## Artifact contract (schema_version 2)
```
$SCRATCH/memrot/artifacts/
  longmemeval_records.parquet   longmemeval_sessions.parquet
  layer_scan.json               retrieval_heads.json
  attn_mass.parquet             # (qid,k,layer,head,attn_mass_gold,gold_pos_frac,ctx_tokens)
  sae_acts.h5                   # /q/<qid>/k<k>/layer<L> -> feat_ids[64], acts[64]
  answers.parquet               # (qid,k,pred,gold,f1,correct,error_type,gold_pos_frac,ctx_tokens)
  run1_manifest.json            run1_verify.json
$SCRATCH/memrot/analysis/       # Run 2 outputs incl. results_summary.json
$SCRATCH/memrot/artifacts/figures/   # fig1..fig4 pdf
paper/paper.md                  # Run 3 output (in repo)
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
5. **n_questions = 170, not 300.** Measured on LongMemEval-S (2026-06-09): only
   170 non-abstention questions have exactly one gold evidence session; the
   plan's 300 does not exist in the data. Evidence sessions are also long
   (median ~2.5k Gemma tokens), so instead of the plan's hard ≤2k filter (which
   would keep ~50 questions) long gold sessions are trimmed to a turn window
   centered on the `has_answer` evidence turn(s) (`gold_trimmed` flag in
   records). Sweep is therefore 170×6 = 1,020 passes (~3-4h), inside the
   original wall-time request.

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
