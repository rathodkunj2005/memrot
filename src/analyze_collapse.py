"""H1: signals decline monotonically in k, controlling for gold position.

Mixed-effects regression (statsmodels MixedLM), question as random intercept:
    signal ~ k + gold_pos_frac + (1 | qid)
fit separately for (a) retrieval-head gold attention mass and (b) mean activation
of recall-associated SAE features.

Recall features (also consumed by intervention.py, written to
recall_features.json): per SAE layer, the analysis.recall_feature_top_n features
whose stored activation correlates most positively with answer correctness
across all (qid, k) samples; target_act = mean activation among correct k=0
samples. Activations absent from the stored top-64 are treated as 0 (documented
approximation).
"""
import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import io_utils
import paths
from load_artifacts import retrieval_head_signal


def _feature_matrix(a, layer):
    """(qids, ks, sparse dict samples) -> dense [n_samples, n_feats_seen]."""
    rows, keys = [], []
    with h5py.File(a.h5, "r") as f:
        for r in a.answers.itertuples():
            g = f[f"/q/{r.qid}/k{r.k}/layer{layer}"]
            rows.append(dict(zip(g["feat_ids"][:].tolist(), g["acts"][:].tolist())))
            keys.append((r.qid, r.k))
    feats = sorted({fid for row in rows for fid in row})
    X = np.zeros((len(rows), len(feats)), dtype=np.float32)
    fidx = {f: i for i, f in enumerate(feats)}
    for i, row in enumerate(rows):
        for fid, v in row.items():
            X[i, fidx[fid]] = v
    return X, np.array(feats), keys


def recall_features(cfg, a, force=False):
    out_path = paths.analysis_dir() / "recall_features.json"
    if out_path.exists() and not force:
        return io_utils.read_json(out_path)

    y = a.answers.set_index(["qid", "k"])["correct"]
    k0_correct = a.answers[(a.answers.k == 0) & a.answers.correct]
    top_n = cfg["analysis"]["recall_feature_top_n"]
    result = {}
    for layer in a.sae_layers:
        X, feats, keys = _feature_matrix(a, layer)
        yy = np.array([bool(y.loc[q, k]) for q, k in keys], dtype=float)
        Xc = X - X.mean(0)
        denom = X.std(0) * yy.std() + 1e-9
        r = (Xc * (yy - yy.mean())[:, None]).mean(0) / denom
        order = np.argsort(-r)[:top_n]
        k0_idx = [i for i, (q, k) in enumerate(keys)
                  if k == 0 and (q in set(k0_correct.qid))]
        result[str(layer)] = [
            {"feat_id": int(feats[i]), "corr": round(float(r[i]), 4),
             "target_act": round(float(X[k0_idx, i].mean()), 4)}
            for i in order]
    io_utils.write_json(out_path, result)
    return result


def _signal_from_features(a, rf):
    """Per (qid,k): mean activation of recall features (normalized per layer)."""
    per_layer = []
    with h5py.File(a.h5, "r") as f:
        for layer, feats in rf.items():
            ids = {x["feat_id"] for x in feats}
            vals = []
            for r in a.answers.itertuples():
                g = f[f"/q/{r.qid}/k{r.k}/layer{layer}"]
                row = dict(zip(g["feat_ids"][:].tolist(), g["acts"][:].tolist()))
                vals.append({"qid": r.qid, "k": r.k,
                             "act": float(np.mean([row.get(i, 0.0) for i in ids]))})
            per_layer.append(pd.DataFrame(vals))
    df = pd.concat(per_layer).groupby(["qid", "k"], as_index=False)["act"].mean()
    return df.rename(columns={"act": "signal"})


def _mixedlm(df, name):
    m = smf.mixedlm("signal ~ k + gold_pos_frac", df, groups=df["qid"]).fit()
    return {"signal": name,
            "beta_k": float(m.params["k"]), "p_k": float(m.pvalues["k"]),
            "beta_pos": float(m.params["gold_pos_frac"]),
            "p_pos": float(m.pvalues["gold_pos_frac"]),
            "n_obs": int(len(df))}


def run(cfg, a, force=False):
    out_path = paths.analysis_dir() / "collapse_stats.parquet"
    rf = recall_features(cfg, a, force=force)
    if out_path.exists() and not force:
        io_utils.status("h1_collapse", True, "cached")
        return pd.read_parquet(out_path), rf

    head_df = retrieval_head_signal(a)
    feat_df = _signal_from_features(a, rf).merge(
        head_df[["qid", "k", "gold_pos_frac"]], on=["qid", "k"])
    rows = [_mixedlm(head_df, "retrieval_head_mass"),
            _mixedlm(feat_df, "recall_feature_act")]
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    io_utils.status("h1_collapse", True,
                    f"beta_k(attn)={rows[0]['beta_k']:.4f} (p={rows[0]['p_k']:.2g}); "
                    f"beta_k(feat)={rows[1]['beta_k']:.4f} (p={rows[1]['p_k']:.2g})")
    return df, rf
