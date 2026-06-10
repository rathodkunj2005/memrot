"""RUN 2 MAIN: validate artifacts -> H1 -> H2 -> H3 (GPU) -> figures -> summary.

    python -u src/run_analysis.py --config config/config.yaml [--force] [--zero]
"""
import argparse

import ablation
import analyze_collapse
import analyze_paired
import intervention
import io_utils
import load_artifacts
import make_figures
import model_loader
import paths
from io_utils import status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.config_path()))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--zero", action="store_true", help="zero- instead of mean-ablation")
    args = ap.parse_args()
    cfg = io_utils.load_config(args.config)

    a = load_artifacts.load(cfg)
    collapse_df, recall_feats = analyze_collapse.run(cfg, a, force=args.force)
    h2 = analyze_paired.run(cfg, a, recall_feats, force=args.force)

    # GPU stages (small inference set)
    model, tok, _ = model_loader.load(cfg)
    abl_df = ablation.run(cfg, a, model, tok, zero=args.zero, force=args.force)
    int_df = intervention.run(cfg, a, model, tok, recall_feats, force=args.force)

    make_figures.run(cfg, a, recall_feats)

    h1 = {r["signal"]: r for r in collapse_df.to_dict("records")}
    h2r = {r["signal"]: r for r in h2["results"]}
    abl = abl_df.set_index("condition").to_dict("index")
    itv = int_df.iloc[0].to_dict()
    acc_by_k = a.answers.groupby("k")["correct"].mean().round(4).to_dict()

    summary = {
        "n_questions": int(a.answers["qid"].nunique()),
        "k_levels": sorted(int(k) for k in a.answers["k"].unique()),
        "acc_by_k": {str(k): float(v) for k, v in acc_by_k.items()},
        "sae_layers": a.sae_layers,
        "n_retrieval_heads": len(a.retrieval_heads),
        "collapse_beta": h1["retrieval_head_mass"]["beta_k"],
        "collapse_beta_p": h1["retrieval_head_mass"]["p_k"],
        "feature_beta": h1["recall_feature_act"]["beta_k"],
        "feature_beta_p": h1["recall_feature_act"]["p_k"],
        "gold_pos_beta": h1["retrieval_head_mass"]["beta_pos"],
        "rho_flip_collapse": h2r["retrieval_head_mass"]["spearman_rho"],
        "rho_p": h2r["retrieval_head_mass"]["spearman_p"],
        "n_paired": h2r["retrieval_head_mass"]["n_both_observed"],
        "n_k0_correct": h2r["retrieval_head_mass"]["n_k0_correct"],
        "cox_hr": h2r["retrieval_head_mass"]["cox_hr_signal_auc"],
        "cox_p": h2r["retrieval_head_mass"]["cox_p_signal_auc"],
        "rho_flip_collapse_feat": h2r["recall_feature_act"]["spearman_rho"],
        "ablation_mode": abl["retrieval"]["mode"],
        "ablation_drop": abl["retrieval"]["drop"],
        "ablation_drop_random": abl["random_control"]["drop"],
        "ablation_n": abl["retrieval"]["n_questions"],
        "intervention_mode": itv["mode"],
        "recovery_frac": float(itv["recovery_frac"]),
        "recovery_ci_lo": float(itv["recovery_ci_lo"]),
        "recovery_ci_hi": float(itv["recovery_ci_hi"]),
        "n_flipped": int(itv["n_flipped"]),
        "run1_manifest": {k: a.manifest[k] for k in
                          ("git_sha", "env_hash", "seed", "schema_version")},
    }
    io_utils.write_json(paths.analysis_dir() / "results_summary.json", summary)
    status("run2", True, f"results_summary.json written "
                         f"(rho={summary['rho_flip_collapse']:.3f}, "
                         f"recovery={summary['recovery_frac']:.2f})")


if __name__ == "__main__":
    main()
