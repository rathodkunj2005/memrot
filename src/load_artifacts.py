"""Run 2/3 entry gate: validate schema_version, load Run 1 artifacts as dataframes."""
from types import SimpleNamespace

import pandas as pd

import io_utils
import paths


def load(cfg) -> SimpleNamespace:
    art = paths.artifacts_dir()
    manifest = io_utils.check_manifest(art / "run1_manifest.json")

    a = SimpleNamespace()
    a.manifest = manifest
    a.records = pd.read_parquet(art / "longmemeval_records.parquet")
    a.sessions = pd.read_parquet(art / "longmemeval_sessions.parquet")
    a.answers = pd.read_parquet(art / "answers.parquet")
    a.attn = pd.read_parquet(art / "attn_mass.parquet")
    a.layer_scan = io_utils.read_json(art / "layer_scan.json")
    a.retrieval_heads = io_utils.read_json(art / "retrieval_heads.json")["retrieval_heads"]
    a.sae_layers = [int(x) for x in manifest["sae_layers"]]
    a.h5 = art / "sae_acts.h5"

    n_q = a.answers["qid"].nunique()
    n_k = a.answers["k"].nunique()
    if len(a.answers) != n_q * n_k:
        io_utils.die("artifact-check",
                     f"answers.parquet not complete grid: {len(a.answers)} rows "
                     f"!= {n_q} qids x {n_k} k-levels")
    io_utils.status("load_artifacts", True,
                    f"{n_q} questions x {n_k} k-levels, "
                    f"{len(a.retrieval_heads)} retrieval heads, "
                    f"SAE layers {a.sae_layers}")
    return a


def retrieval_head_signal(a) -> pd.DataFrame:
    """Per (qid,k): mean gold-span attention mass over identified retrieval heads."""
    keys = sorted({(h["layer"], h["head"]) for h in a.retrieval_heads})
    sub = a.attn.merge(pd.DataFrame(keys, columns=["layer", "head"]),
                       on=["layer", "head"])
    return (sub.groupby(["qid", "k"])
            .agg(signal=("attn_mass_gold", "mean"),
                 gold_pos_frac=("gold_pos_frac", "first"),
                 ctx_tokens=("ctx_tokens", "first"))
            .reset_index())
