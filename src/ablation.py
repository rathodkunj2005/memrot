"""H3a: ablating retrieval heads at k=0 reproduces the failure pattern.

Mean-ablation (default; zero-ablation behind analysis.zero_ablation / --zero):
each identified retrieval head's contribution is replaced by its dataset-mean
output, estimated from analysis.ablation_calibration_n unablated k=0 prompts
(mean over all token positions). Implemented as a forward_pre_hook on
self_attn.o_proj, slicing the per-head segment of its input
[B, T, n_heads*head_dim] — version-robust (no attention internals touched).

Control: an equally sized set of random non-retrieval heads (same seed),
mean-ablated the same way. Evaluated on analysis.ablation_n questions that were
correct at k=0 in Run 1, so unablated accuracy = 1.0 by construction.
"""
import numpy as np
import pandas as pd
import torch

import grade
import io_utils
import model_loader
import paths
from haystack import HaystackBuilder


class HeadAblator:
    def __init__(self, model, heads, mode="mean"):
        """heads: list[(layer, head)]; mode: 'mean' | 'zero'."""
        self.model = model
        self.heads = heads
        self.mode = mode
        self.head_dim = model.config.head_dim
        self.means = {}     # (layer, head) -> tensor [head_dim]
        self._sums, self._counts = {}, {}
        self.handles = []

    def _by_layer(self):
        by = {}
        for l, h in self.heads:
            by.setdefault(l, []).append(h)
        return by

    # ---- pass 1: calibration (record mean o_proj input per head) ----
    def calibrate_hooks(self):
        def make(layer):
            def pre(module, args):
                x = args[0][0]                       # [T, n_heads*head_dim]
                for h in self._by_layer()[layer]:
                    seg = x[:, h * self.head_dim:(h + 1) * self.head_dim].float()
                    key = (layer, h)
                    self._sums[key] = self._sums.get(key, 0) + seg.sum(0).cpu()
                    self._counts[key] = self._counts.get(key, 0) + seg.shape[0]
            return pre
        for layer in self._by_layer():
            m = self.model.model.layers[layer].self_attn.o_proj
            self.handles.append(m.register_forward_pre_hook(make(layer)))

    def finish_calibration(self):
        self.remove()
        for key, s in self._sums.items():
            self.means[key] = (s / self._counts[key]).to("cuda", torch.bfloat16)

    # ---- pass 2: ablation ----
    def ablate_hooks(self):
        by = self._by_layer()

        def make(layer):
            def pre(module, args):
                x = args[0].clone()
                for h in by[layer]:
                    sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    if self.mode == "zero":
                        x[..., sl] = 0
                    else:
                        x[..., sl] = self.means[(layer, h)]
                return (x,) + args[1:]
            return pre
        for layer in by:
            m = self.model.model.layers[layer].self_attn.o_proj
            self.handles.append(m.register_forward_pre_hook(make(layer)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _eval_set(cfg, model, tok, builder, recs, ablator=None):
    n_correct = 0
    for rec in recs:
        hs = builder.build(rec, 0)
        if ablator:
            ablator.ablate_hooks()
        try:
            pred, _ = model_loader.greedy_generate(model, tok, hs["input_ids"], cfg)
        finally:
            if ablator:
                ablator.remove()
        n_correct += grade.grade(pred, rec["answer"], [], cfg)["correct"]
    return n_correct / len(recs)


def run(cfg, a, model, tok, zero=False, force=False):
    out_path = paths.analysis_dir() / "h3_ablation.parquet"
    if out_path.exists() and not force:
        io_utils.status("h3_ablation", True, "cached")
        return pd.read_parquet(out_path)

    mode = "zero" if (zero or cfg["analysis"]["zero_ablation"]) else "mean"
    k0_ok = a.answers[(a.answers.k == 0) & a.answers.correct].sort_values("qid")
    qids = k0_ok["qid"].head(cfg["analysis"]["ablation_n"]).tolist()
    recs = a.records[a.records.qid.isin(qids)].sort_values("qid").to_dict("records")
    builder = HaystackBuilder(cfg, tok, a.sessions)

    r_heads = [(h["layer"], h["head"]) for h in a.retrieval_heads]
    rng = np.random.default_rng(cfg["run"]["seed"])
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
            if (l, h) not in set(r_heads)]
    ctrl = [pool[i] for i in rng.choice(len(pool), size=len(r_heads), replace=False)]

    rows = []
    for cond, heads in [("retrieval", r_heads), ("random_control", ctrl)]:
        ab = HeadAblator(model, heads, mode=mode)
        if mode == "mean":
            ab.calibrate_hooks()
            for rec in recs[: cfg["analysis"]["ablation_calibration_n"]]:
                hs = builder.build(rec, 0)
                with torch.no_grad():
                    model(torch.tensor([hs["input_ids"]], device="cuda"), use_cache=False)
            ab.finish_calibration()
        acc = _eval_set(cfg, model, tok, builder, recs, ablator=ab)
        rows.append({"condition": cond, "mode": mode, "n_heads": len(heads),
                     "n_questions": len(recs), "acc_unablated": 1.0,
                     "acc_ablated": acc, "drop": 1.0 - acc})
        io_utils.status("h3_ablation", True, f"{cond}({mode}): acc {acc:.3f}")
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return df
