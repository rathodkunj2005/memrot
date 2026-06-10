"""Gemma Scope JumpReLU SAEs, loaded directly from google/gemma-scope-2b-pt-res.

Deviation from the v1 plan's sae_lens dependency (documented in README): the
params.npz format is stable and a JumpReLU encoder is ~10 lines, so we avoid the
sae_lens/transformer_lens dependency stack entirely. For each layer we pick the
width_16k SAE whose average_l0 is nearest sae.target_l0 (the canonical
convention); the chosen l0 per layer is logged and written into the manifest.

Hook point: SAEs are trained on the post-layer residual stream
(resid_post of layer L) = output[0] of model.model.layers[L].
"""
import re

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

import io_utils
import paths


class JumpReLUSAE:
    def __init__(self, npz_path, device="cuda"):
        p = np.load(npz_path)
        self.W_enc = torch.tensor(p["W_enc"], dtype=torch.float32, device=device)
        self.b_enc = torch.tensor(p["b_enc"], dtype=torch.float32, device=device)
        self.W_dec = torch.tensor(p["W_dec"], dtype=torch.float32, device=device)
        self.b_dec = torch.tensor(p["b_dec"], dtype=torch.float32, device=device)
        self.threshold = torch.tensor(p["threshold"], dtype=torch.float32, device=device)
        self.d_sae = self.W_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] (any dtype) -> activations [..., d_sae] (float32)."""
        pre = x.to(torch.float32) @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decoder_dirs(self, feat_ids) -> torch.Tensor:
        return self.W_dec[feat_ids]                      # [n, d_model]


def available_l0s(cfg) -> dict:
    """{layer: {l0: filename}} for width_16k; HF listing cached on disk."""
    cache = paths.data_dir() / "gemma_scope_listing.json"
    if cache.exists():
        raw = io_utils.read_json(cache)
    else:
        files = HfApi().list_repo_files(cfg["sae"]["repo"])
        raw = [f for f in files if f.endswith("params.npz")]
        io_utils.write_json(cache, raw)
    out = {}
    pat = re.compile(rf"layer_(\d+)/width_{cfg['sae']['width']}/average_l0_(\d+)/params\.npz")
    for f in raw:
        m = pat.fullmatch(f)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
    return out


def pick_l0(l0s: dict, target: int) -> int:
    return min(l0s, key=lambda x: abs(x - target))


def load_sae(cfg, layer: int, device="cuda"):
    """Returns (JumpReLUSAE, chosen_l0)."""
    table = available_l0s(cfg)
    if layer not in table:
        raise RuntimeError(f"no width_{cfg['sae']['width']} SAE for layer {layer}")
    l0 = pick_l0(table[layer], cfg["sae"]["target_l0"])
    path = hf_hub_download(cfg["sae"]["repo"], table[layer][l0])
    return JumpReLUSAE(path, device=device), l0


def sae_layers(cfg) -> list:
    """Resolve sae.layers: explicit list, or 'auto' from layer_scan.json."""
    layers = cfg["sae"]["layers"]
    if layers != "auto":
        return [int(x) for x in layers]
    scan = paths.artifacts_dir() / "layer_scan.json"
    if not scan.exists():
        raise RuntimeError("sae.layers=auto but layer_scan.json missing; run layer scan")
    return [int(x) for x in io_utils.read_json(scan)["chosen_layers"]]


def topk_acts(sae: JumpReLUSAE, resid_vec: torch.Tensor, topk: int):
    """(feat_ids int64[topk], acts float32[topk]) for one residual vector."""
    acts = sae.encode(resid_vec)
    vals, idx = torch.topk(acts, k=min(topk, acts.shape[-1]))
    return idx.cpu().numpy(), vals.cpu().numpy()
