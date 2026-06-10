"""Gemma-2-2B loader: bf16, EAGER attention (non-negotiable for hooks), determinism."""
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from io_utils import die, status


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # "where ops allow"


def load(cfg):
    mcfg = cfg["model"]
    assert mcfg["attn_implementation"] == "eager", \
        "config.model.attn_implementation must be 'eager' (hooks need attention weights)"
    set_determinism(cfg["run"]["seed"])

    if not torch.cuda.is_available():
        die("model_load", "CUDA not available on this node")
    if not torch.cuda.is_bf16_supported():
        die("model_load",
            f"GPU {torch.cuda.get_device_name(0)} lacks bf16; submit with "
            "--constraint='3090|a100'")

    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        torch_dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"],
    ).to("cuda").eval()
    assert model.config.max_position_embeddings >= mcfg["max_ctx"], \
        f"model ctx {model.config.max_position_embeddings} < config max_ctx"

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # Gemma-2 interleaves sliding-window (even idx) and global (odd idx) attention
    sliding = [bool(getattr(model.model.layers[i].self_attn, "sliding_window", None))
               or (i % 2 == 0) for i in range(n_layers)]
    status("model_load", True,
           f"{mcfg['name']} bf16 eager on {torch.cuda.get_device_name(0)} "
           f"({n_layers}L x {n_heads}H)")
    return model, tok, {"n_layers": n_layers, "n_heads": n_heads, "sliding": sliding}


@torch.no_grad()
def greedy_generate(model, tok, input_ids, cfg):
    """Greedy continuation; returns (answer_text, generated_ids)."""
    ids = torch.tensor([input_ids], device="cuda")
    out = model.generate(
        ids,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0, len(input_ids):]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, gen.tolist()
