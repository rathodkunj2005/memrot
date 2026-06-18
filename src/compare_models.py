"""Cross-model scaling comparison (2B -> 9B -> 27B).

Reads each model's analysis/<id>/results_summary.json and artifacts/<id>/answers.parquet
and writes:
  - analysis/compare/scaling_summary.json   (one record per model, machine-readable)
  - artifacts/compare/figures/fig5..fig8.pdf (scaling figures)

Robust to missing models: only models with a results_summary.json are included, so
this runs meaningfully even when just 2b is present. Single-model input still emits
a (degenerate one-point) summary and figures.

Caveats baked into the captions, per the paper's Limitations:
  - feature/SAE quantities for 27b come from only 3 available Gemma Scope layers;
  - accuracy CIs are Wilson intervals over questions (single seed).

Usage:
    python -u src/compare_models.py --models "2b 9b 27b"
"""
import argparse
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import io_utils
import paths

# Approx parameter counts (billions) for the scaling x-axis.
MODEL_PARAMS_B = {"2b": 2.6, "9b": 9.2, "27b": 27.2}
ORDER = ["2b", "9b", "27b"]


def _model_dirs(m):
    root = paths.scratch_root()
    return root / "artifacts" / m, root / "analysis" / m


def _wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def _acc_ci(ans):
    """k -> (acc, lo, hi) Wilson 95% over questions."""
    out = {}
    for k, g in ans.groupby("k"):
        p = float(g["correct"].mean())
        lo, hi = _wilson(p, len(g))
        out[int(k)] = (p, lo, hi)
    return out


def load_models(model_ids):
    rows = []
    for m in model_ids:
        art, ana = _model_dirs(m)
        summ_path = ana / "results_summary.json"
        if not summ_path.exists():
            io_utils.status("compare", True, f"skip {m}: no results_summary.json")
            continue
        s = io_utils.read_json(summ_path)
        ans_path = art / "answers.parquet"
        acc_ci = _acc_ci(pd.read_parquet(ans_path)) if ans_path.exists() else {}
        rows.append({"model_id": m, "params_b": MODEL_PARAMS_B.get(m, float("nan")),
                     "summary": s, "acc_ci": acc_ci})
    rows.sort(key=lambda r: ORDER.index(r["model_id"]) if r["model_id"] in ORDER
              else len(ORDER))
    return rows


# ----------------------------- figures -----------------------------------

def fig5_scaling_acc(rows, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in rows:
        s = r["summary"]
        ks = sorted(int(k) for k in s["acc_by_k"])
        acc = [s["acc_by_k"][str(k)] for k in ks]
        if r["acc_ci"]:
            lo = [r["acc_ci"].get(k, (acc[i], acc[i], acc[i]))[1] for i, k in enumerate(ks)]
            hi = [r["acc_ci"].get(k, (acc[i], acc[i], acc[i]))[2] for i, k in enumerate(ks)]
            yerr = [np.array(acc) - np.array(lo), np.array(hi) - np.array(acc)]
        else:
            yerr = None
        ax.errorbar(ks, acc, yerr=yerr, marker="o", capsize=3,
                    label=f"Gemma-2-{r['model_id']}")
    ax.set_xlabel("k (distractor sessions)")
    ax.set_ylabel("accuracy (Wilson 95% CI)")
    ax.set_title("Accuracy vs fill, by model size", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_scaling_acc.pdf")
    plt.close(fig)


def fig6_beta_scaling(rows, out):
    x = [r["params_b"] for r in rows]
    b_attn = [r["summary"].get("collapse_beta", float("nan")) for r in rows]
    b_feat = [r["summary"].get("feature_beta", float("nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(x, b_attn, marker="o", label=r"attention-mass $\beta_k$")
    ax.plot(x, b_feat, marker="s", ls="--", label=r"SAE-feature $\beta_k$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{r['model_id']}" for r in rows])
    ax.set_xlabel("model size (params, log)")
    ax.set_ylabel(r"H1 slope $\beta_k$")
    ax.set_title("Signal-decline slope vs scale", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_beta_scaling.pdf")
    plt.close(fig)


def fig7_h2_scaling(rows, out):
    labels = [r["model_id"] for r in rows]
    rho = [r["summary"].get("rho_flip_collapse", float("nan")) for r in rows]
    rho_f = [r["summary"].get("rho_flip_collapse_feat", float("nan")) for r in rows]
    xi = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.bar(xi - w / 2, rho, w, label=r"attn $\rho$")
    ax.bar(xi + w / 2, rho_f, w, label=r"feature $\rho$")
    ax.set_xticks(xi, [f"Gemma-2-{m}" for m in labels])
    ax.set_ylabel(r"H2 Spearman $\rho$ (collapse vs flip)")
    ax.set_title("Paired association vs scale", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig7_h2_scaling.pdf")
    plt.close(fig)


def fig8_causal_scaling(rows, out):
    labels = [r["model_id"] for r in rows]
    drop = [r["summary"].get("ablation_drop", float("nan")) for r in rows]
    rand = [r["summary"].get("ablation_drop_random", float("nan")) for r in rows]
    rec = [r["summary"].get("recovery_frac", float("nan")) for r in rows]
    rec_lo = [r["summary"].get("recovery_ci_lo", float("nan")) for r in rows]
    rec_hi = [r["summary"].get("recovery_ci_hi", float("nan")) for r in rows]
    xi = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    w = 0.38
    ax1.bar(xi - w / 2, drop, w, label="retrieval heads")
    ax1.bar(xi + w / 2, rand, w, label="random control")
    ax1.set_xticks(xi, labels)
    ax1.set_ylabel("accuracy drop at k=0")
    ax1.set_title("H3a ablation vs scale", fontsize=9)
    ax1.legend(fontsize=8)
    yerr = [np.array(rec) - np.array(rec_lo), np.array(rec_hi) - np.array(rec)]
    ax2.bar(xi, rec, 0.5, color="tab:green", yerr=yerr, capsize=4)
    ax2.set_xticks(xi, labels)
    ax2.set_ylabel("flipped answers recovered")
    ax2.set_title("H3b recovery vs scale", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig8_causal_scaling.pdf")
    plt.close(fig)


def run(model_ids):
    rows = load_models(model_ids)
    if not rows:
        io_utils.die("compare", "no models with a results_summary.json were found")

    # machine-readable scaling table
    summary = {"models": []}
    for r in rows:
        s = r["summary"]
        summary["models"].append({
            "model_id": r["model_id"], "params_b": r["params_b"],
            "n_questions": s.get("n_questions"),
            "acc_by_k": s.get("acc_by_k"),
            "acc_ci_by_k": {str(k): {"acc": v[0], "lo": v[1], "hi": v[2]}
                            for k, v in r["acc_ci"].items()},
            "collapse_beta": s.get("collapse_beta"),
            "feature_beta": s.get("feature_beta"),
            "rho_flip_collapse": s.get("rho_flip_collapse"),
            "rho_flip_collapse_feat": s.get("rho_flip_collapse_feat"),
            "ablation_drop": s.get("ablation_drop"),
            "ablation_drop_random": s.get("ablation_drop_random"),
            "recovery_frac": s.get("recovery_frac"),
            "recovery_ci": [s.get("recovery_ci_lo"), s.get("recovery_ci_hi")],
            "sae_layers": s.get("sae_layers"),
        })
    cmp_ana = paths.scratch_root() / "analysis" / "compare"
    cmp_ana.mkdir(parents=True, exist_ok=True)
    io_utils.write_json(cmp_ana / "scaling_summary.json", summary)

    out = paths.scratch_root() / "artifacts" / "compare" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig5_scaling_acc(rows, out)
    fig6_beta_scaling(rows, out)
    fig7_h2_scaling(rows, out)
    fig8_causal_scaling(rows, out)
    io_utils.status("compare", True,
                    f"{len(rows)} model(s): {[r['model_id'] for r in rows]} -> "
                    f"{cmp_ana/'scaling_summary.json'}, figures in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="2b 9b 27b",
                    help="space/comma separated model ids to include")
    args = ap.parse_args()
    ids = [m for m in re.split(r"[,\s]+", args.models.strip()) if m]
    run(ids)


if __name__ == "__main__":
    main()
