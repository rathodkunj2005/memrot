"""Figures 1-4 -> artifacts/figures/*.pdf (matplotlib only, no seaborn)."""
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import io_utils
import paths
from analyze_collapse import _signal_from_features
from load_artifacts import retrieval_head_signal


def _mean_ci(g):
    m = g.mean()
    se = g.std() / np.sqrt(len(g))
    return m, 1.96 * se


def fig1_collapse(cfg, a, recall_feats, out):
    head = retrieval_head_signal(a)
    feat = _signal_from_features(a, recall_feats)
    acc = a.answers.groupby("k")["correct"].mean()

    fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    for df, label, color in [(head, "retrieval-head mass on gold", "tab:blue"),
                             (feat, "recall-feature activation", "tab:green")]:
        norm = df.merge(df[df.k == 0][["qid", "signal"]].rename(
            columns={"signal": "s0"}), on="qid")
        norm["rel"] = norm["signal"] / (norm["s0"] + 1e-9)
        ks = sorted(norm.k.unique())
        ms, cis = zip(*[_mean_ci(norm.loc[norm.k == k, "rel"]) for k in ks])
        ax1.errorbar(ks, ms, yerr=cis, marker="o", label=label, color=color)
    ax1.set_xlabel("k (distractor sessions)")
    ax1.set_ylabel("signal relative to k=0")
    ax2 = ax1.twinx()
    ax2.plot(acc.index, acc.values, marker="s", color="tab:red", ls="--",
             label="accuracy")
    ax2.set_ylabel("accuracy", color="tab:red")
    ax1.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig1_collapse_curves.pdf")
    plt.close(fig)


def fig2_paired(out):
    pq = pd.read_parquet(paths.analysis_dir() / "h2_per_question.parquet")
    pq = pq[pq.signal == "retrieval_head_mass"]
    both = pq[pq.flipped & pq.collapsed]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(both.k_collapse + rng.uniform(-.25, .25, len(both)),
               both.k_flip + rng.uniform(-.25, .25, len(both)),
               s=14, alpha=0.5)
    lim = [0, max(both.k_flip.max(), both.k_collapse.max()) + 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("k_collapse (signal < 50% of k=0)")
    ax.set_ylabel("k_flip (first wrong answer)")
    fig.tight_layout()
    fig.savefig(out / "fig2_flip_vs_collapse.pdf")
    plt.close(fig)


def fig3_causal(out):
    abl = pd.read_parquet(paths.analysis_dir() / "h3_ablation.parquet")
    inter = pd.read_parquet(paths.analysis_dir() / "h3_intervention.parquet")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    ax1.bar(abl["condition"], abl["drop"], color=["tab:blue", "tab:gray"])
    ax1.set_ylabel("accuracy drop at k=0 (ablated)")
    ax1.set_title(f"H3a {abl['mode'].iloc[0]}-ablation", fontsize=9)
    r = inter.iloc[0]
    ax2.bar(["recovery"], [r.recovery_frac], color="tab:green",
            yerr=[[r.recovery_frac - r.recovery_ci_lo],
                  [r.recovery_ci_hi - r.recovery_frac]], capsize=4)
    ax2.set_ylabel(f"flipped answers recovered at k={int(r.k)}")
    ax2.set_title(f"H3b {r['mode']} intervention (n={int(r.n_flipped)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_causal_bars.pdf")
    plt.close(fig)


def fig4_heatmap(cfg, a, recall_feats, out):
    ks = sorted(a.answers.k.unique())
    layers = sorted(int(x) for x in recall_feats)
    M = np.zeros((len(layers), len(ks)))
    with h5py.File(a.h5, "r") as f:
        for i, layer in enumerate(layers):
            ids = {x["feat_id"] for x in recall_feats[str(layer)]}
            for j, k in enumerate(ks):
                vals = []
                for r in a.answers[a.answers.k == k].itertuples():
                    g = f[f"/q/{r.qid}/k{r.k}/layer{layer}"]
                    row = dict(zip(g["feat_ids"][:].tolist(), g["acts"][:].tolist()))
                    vals.append(np.mean([row.get(x, 0.0) for x in ids]))
                M[i, j] = np.mean(vals)
    M = M / (M[:, :1] + 1e-9)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ks)), ks)
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("k")
    ax.set_ylabel("SAE layer")
    fig.colorbar(im, label="recall-feature act (rel. k=0)")
    fig.tight_layout()
    fig.savefig(out / "fig4_feature_heatmap.pdf")
    plt.close(fig)


def run(cfg, a, recall_feats):
    out = paths.figures_dir()
    fig1_collapse(cfg, a, recall_feats, out)
    fig2_paired(out)
    fig3_causal(out)
    fig4_heatmap(cfg, a, recall_feats, out)
    io_utils.status("figures", True, str(out))
