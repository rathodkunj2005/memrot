"""Login-node dry run (CPU, no gated repos): exercises data_prep, haystack
determinism/budget/spans, io_utils round-trip, and the Gemma Scope npz loader
with a stand-in (gpt2) tokenizer. Run with SCRATCH pointed at a temp dir so the
real artifact tree is untouched:

    SCRATCH=$(mktemp -d) HF_HOME=$SCRATCH/hf_cache python src/dryrun_login.py

Not part of the three-run contract; pure pre-flight insurance.
"""
import copy
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

import data_prep
import io_utils
import paths
import sae_loader
from haystack import HaystackBuilder
from io_utils import status

_s = os.environ.get("SCRATCH", "")
_real = f"/scratch/general/vast/{os.environ.get('USER', '')}"
assert _s and os.path.realpath(_s) != os.path.realpath(_real), \
    "refusing to run against the real $SCRATCH; use SCRATCH=$(mktemp -d)"

cfg = io_utils.load_config(paths.config_path())
cfg = copy.deepcopy(cfg)
cfg["benchmark"]["n_questions"] = 12

tok = AutoTokenizer.from_pretrained("gpt2")   # stand-in; Gemma tokenizer is gated
records, sessions = data_prep.prepare(cfg, tok)
assert len(records) == 12 and len(sessions) > 100
assert records["gold_tokens"].max() <= cfg["benchmark"]["gold_max_tokens"]

builder = HaystackBuilder(cfg, tok, sessions)
rec = records.iloc[0].to_dict()
for k in cfg["benchmark"]["k_levels"]:
    h1 = builder.build(rec, k)
    h2 = builder.build(rec, k)
    assert h1["input_ids"] == h2["input_ids"], f"non-deterministic build at k={k}"
    assert h1["ctx_tokens"] <= cfg["model"]["haystack_token_budget"]
    s, e = h1["gold_span"]
    gold_ids = tok(rec["gold_text"] + "\n\n", add_special_tokens=False)["input_ids"]
    assert h1["input_ids"][s:e] == gold_ids, f"gold span mis-indexed at k={k}"
    assert len(h1["distractor_uids"]) == k
    print(f"  k={k}: ctx={h1['ctx_tokens']} gold_pos_frac={h1['gold_pos_frac']:.2f}")

# different questions get different distractors/positions
other = records.iloc[1].to_dict()
assert builder.build(other, 4)["input_ids"] != builder.build(rec, 4)["input_ids"]

# io round-trip
art = paths.artifacts_dir()
io_utils.jsonl_append(art / "t.jsonl", {"qid": "b", "k": 1, "x": 2.0})
io_utils.jsonl_append(art / "t.jsonl", {"qid": "a", "k": 0, "x": 1.0})
df = io_utils.compact_to_parquet(art / "t.jsonl", art / "t.parquet",
                                 ["qid", "k", "x"], ["qid", "k"])
assert df.iloc[0]["qid"] == "a"
io_utils.h5_write_acts(art / "t.h5", "qa", 0,
                       {12: (np.arange(64), np.random.rand(64))})
assert io_utils.h5_has(art / "t.h5", "qa", 0)
fids, acts = io_utils.h5_read_acts(art / "t.h5", "qa", 0, 12)
assert fids.shape == (64,)

# real Gemma Scope npz (open repo), CPU
sae, l0 = sae_loader.load_sae(cfg, 12, device="cpu")
assert sae.d_sae == 16384
a = sae.encode(torch.randn(3, sae.W_enc.shape[0]))
assert a.shape == (3, 16384) and (a >= 0).all()
ids, vals = sae_loader.topk_acts(sae, torch.randn(sae.W_enc.shape[0]), 64)
assert ids.shape == (64,)
print(f"  SAE layer 12: chose average_l0_{l0}, encode OK")

status("dryrun_login", True, "data_prep + haystack + io + sae_loader all verified")
