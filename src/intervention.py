"""H3b: intervening at high k recovers flipped answers.

Targets: questions correct at k=0 but wrong at k=analysis.flip_high_k in Run 1.

Mode 'feature' (primary): at each selected SAE layer, a forward hook on the
decoder layer re-injects collapsed recall features into the residual stream at
the current last position: for each recall feature f (analysis recall_features),
    x += max(0, target_act_f - act_now_f) * W_dec[f]
where act_now is SAE-encoded in-hook (one vector) and target_act_f is the mean
activation among correct k=0 samples (from recall_features.json). Applied at
every decoding step.

Mode 'head': scale identified retrieval heads' o_proj input slice by
analysis.intervention_alpha (amplifies whatever those heads did attend to).

Recovery fraction = recovered / flipped, with a bootstrap 95% CI over targets.
"""
import numpy as np
import pandas as pd
import torch

import grade
import io_utils
import model_loader
import paths
import sae_loader
from haystack import HaystackBuilder


class FeatureInjector:
    def __init__(self, model, cfg, recall_feats):
        self.model = model
        self.handles = []
        self.spec = {}      # layer -> (sae, feat_ids tensor, targets tensor)
        for layer_s, feats in recall_feats.items():
            layer = int(layer_s)
            sae, _ = sae_loader.load_sae(cfg, layer)
            ids = torch.tensor([f["feat_id"] for f in feats], device="cuda")
            tgt = torch.tensor([f["target_act"] for f in feats], device="cuda")
            self.spec[layer] = (sae, ids, tgt)

    def attach(self):
        def make(layer):
            sae, ids, tgt = self.spec[layer]

            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                x = h[0, -1, :]
                acts = sae.encode(x)[ids]
                delta = torch.clamp(tgt - acts, min=0) @ sae.W_dec[ids]
                h = h.clone()
                h[0, -1, :] = (x.float() + delta).to(h.dtype)
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return hook
        for layer in self.spec:
            self.handles.append(
                self.model.model.layers[layer].register_forward_hook(make(layer)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class HeadAmplifier:
    def __init__(self, model, heads, alpha):
        self.model, self.alpha = model, alpha
        self.head_dim = model.config.head_dim
        self.by = {}
        for l, h in heads:
            self.by.setdefault(l, []).append(h)
        self.handles = []

    def attach(self):
        def make(layer):
            def pre(module, args):
                x = args[0].clone()
                for h in self.by[layer]:
                    sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    x[..., sl] = x[..., sl] * self.alpha
                return (x,) + args[1:]
            return pre
        for layer in self.by:
            m = self.model.model.layers[layer].self_attn.o_proj
            self.handles.append(m.register_forward_pre_hook(make(layer)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _bootstrap_ci(successes: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(successes)
    fracs = [successes[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(fracs, 2.5)), float(np.percentile(fracs, 97.5))


def run(cfg, a, model, tok, recall_feats, force=False):
    out_path = paths.analysis_dir() / "h3_intervention.parquet"
    if out_path.exists() and not force:
        io_utils.status("h3_intervention", True, "cached")
        return pd.read_parquet(out_path)

    k_hi = cfg["analysis"]["flip_high_k"]
    ans = a.answers
    ok0 = set(ans[(ans.k == 0) & ans.correct].qid)
    flipped = ans[(ans.k == k_hi) & ~ans.correct & ans.qid.isin(ok0)] \
        .sort_values("qid")["qid"].tolist()
    if not flipped:
        io_utils.die("h3_intervention", f"no questions flipped at k={k_hi}")
    recs = a.records[a.records.qid.isin(flipped)].sort_values("qid").to_dict("records")
    builder = HaystackBuilder(cfg, tok, a.sessions)

    mode = cfg["analysis"]["intervention_mode"]
    if mode == "feature":
        dev = FeatureInjector(model, cfg, recall_feats)
    else:
        heads = [(h["layer"], h["head"]) for h in a.retrieval_heads]
        dev = HeadAmplifier(model, heads, cfg["analysis"]["intervention_alpha"])

    succ = []
    for rec in recs:
        hs = builder.build(rec, k_hi)        # identical seed path as Run 1
        dev.attach()
        try:
            pred, _ = model_loader.greedy_generate(model, tok, hs["input_ids"], cfg)
        finally:
            dev.remove()
        succ.append(bool(grade.grade(pred, rec["answer"], [], cfg)["correct"]))
    succ = np.array(succ)
    lo, hi = _bootstrap_ci(succ, cfg["analysis"]["n_boot"], cfg["run"]["seed"])
    df = pd.DataFrame([{
        "mode": mode, "k": k_hi, "n_flipped": len(succ),
        "n_recovered": int(succ.sum()),
        "recovery_frac": float(succ.mean()),
        "recovery_ci_lo": lo, "recovery_ci_hi": hi,
    }])
    df.to_parquet(out_path, index=False)
    io_utils.status("h3_intervention", True,
                    f"{mode}: recovered {succ.sum()}/{len(succ)} "
                    f"({succ.mean():.2f}, 95% CI [{lo:.2f},{hi:.2f}])")
    return df
