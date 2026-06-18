# Tracked summary artifacts

Small JSON summaries committed so every headline number in `paper/paper.tex` is
auditable from the repo alone (the large parquet/HDF5 artifacts stay in
`$SCRATCH/memrot/artifacts/<model_id>/` and `analysis/<model_id>/`).

## `2b/` — Gemma-2-2B-it run (2026-06-15)
- `run1_verify.json` — `acc_by_k`, k=0 gate, NaN/row-count checks, chosen SAE layers.
- `run1_manifest.json` — git sha, env hash, seed, schema version, config snapshot.
- `layer_scan.json` — empirically chosen SAE layers + per-layer l0.
- `results_summary.json` — H1 β_k (attn & feature), H2 ρ, H3 ablation drop +
  recovery fraction/CI. Source of the abstract/Results numbers.
- `h2_paired.json`, `recall_features.json` — H2 paired stats and the (in-sample)
  recall feature selection.

Caveat: per the paper's Limitations, the SAE-feature numbers here are produced by
in-sample feature selection and are descriptive, not held-out evidence.
