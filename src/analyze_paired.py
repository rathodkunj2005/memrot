"""H2 (headline): paired within-question test.

For questions correct at k=0:
  k_flip(q)     = first k > 0 where the answer is wrong (right-censored at k_max
                  if it never flips)
  k_collapse(q) = first k > 0 where the signal drops below
                  analysis.collapse_frac * signal(k=0) (same censoring)
Tests:
  - Spearman rho(k_flip, k_collapse) over questions where BOTH events observed
  - Cox PH (statsmodels PHReg): time = k_flip with censoring; covariates =
    normalized signal AUC (mean_k signal/signal_0; defined for every question)
    and gold_pos_frac — survival handles the censored questions Spearman drops.
Run for both signals (retrieval-head mass, recall-feature activation).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.duration.hazard_regression import PHReg

import io_utils
import paths
from analyze_collapse import _signal_from_features
from load_artifacts import retrieval_head_signal


def _first_event_k(ks, flags):
    for k, f in zip(ks, flags):
        if k > 0 and f:
            return k, True
    return max(ks), False           # censored


def _per_question(ans, sig_df, frac):
    rows = []
    for qid, g in ans.sort_values("k").groupby("qid"):
        if not bool(g.loc[g.k == 0, "correct"].iloc[0]):
            continue                # paired design conditions on k=0 correctness
        s = sig_df[sig_df.qid == qid].sort_values("k")
        s0 = float(s.loc[s.k == 0, "signal"].iloc[0])
        ks = g["k"].tolist()
        k_flip, flipped = _first_event_k(ks, (~g["correct"]).tolist())
        if s0 <= 0:
            k_coll, collapsed = max(ks), False
        else:
            k_coll, collapsed = _first_event_k(
                s["k"].tolist(), (s["signal"] < frac * s0).tolist())
        auc = float(np.mean(s["signal"] / (s0 + 1e-9)))
        rows.append({"qid": qid, "k_flip": k_flip, "flipped": flipped,
                     "k_collapse": k_coll, "collapsed": collapsed,
                     "signal_auc": auc,
                     "gold_pos_frac": float(g["gold_pos_frac"].mean())})
    return pd.DataFrame(rows)


def _h2_for_signal(ans, sig_df, frac, name):
    pq = _per_question(ans, sig_df, frac)
    both = pq[pq.flipped & pq.collapsed]
    if len(both) >= 5:
        rho, p = spearmanr(both["k_flip"], both["k_collapse"])
    else:
        rho, p = float("nan"), float("nan")
    exog = pq[["signal_auc", "gold_pos_frac"]].to_numpy()
    cox = PHReg(pq["k_flip"].to_numpy(), exog,
                status=pq["flipped"].astype(int).to_numpy(),
                ties="breslow").fit()
    return {
        "signal": name,
        "n_k0_correct": int(len(pq)),
        "n_flipped": int(pq.flipped.sum()),
        "n_both_observed": int(len(both)),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "cox_coef_signal_auc": float(cox.params[0]),
        "cox_p_signal_auc": float(cox.pvalues[0]),
        "cox_hr_signal_auc": float(np.exp(cox.params[0])),
    }, pq


def run(cfg, a, recall_feats, force=False):
    out_path = paths.analysis_dir() / "h2_paired.json"
    pq_path = paths.analysis_dir() / "h2_per_question.parquet"
    if out_path.exists() and not force:
        io_utils.status("h2_paired", True, "cached")
        return io_utils.read_json(out_path)

    frac = cfg["analysis"]["collapse_frac"]
    head_sig = retrieval_head_signal(a)
    feat_sig = _signal_from_features(a, recall_feats)

    res_head, pq_head = _h2_for_signal(a.answers, head_sig, frac, "retrieval_head_mass")
    res_feat, pq_feat = _h2_for_signal(a.answers, feat_sig, frac, "recall_feature_act")
    pq_head.assign(signal="retrieval_head_mass") \
        .pipe(lambda d: pd.concat([d, pq_feat.assign(signal="recall_feature_act")])) \
        .to_parquet(pq_path, index=False)

    out = {"collapse_frac": frac, "results": [res_head, res_feat]}
    io_utils.write_json(out_path, out)
    io_utils.status("h2_paired", True,
                    f"rho(head)={res_head['spearman_rho']:.3f} "
                    f"(n={res_head['n_both_observed']}, p={res_head['spearman_p']:.2g}); "
                    f"rho(feat)={res_feat['spearman_rho']:.3f}")
    return out
