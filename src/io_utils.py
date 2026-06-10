"""Artifact contract: schemas, incremental append + resume, manifests, STATUS lines.

Frozen contracts (bump SCHEMA_VERSION on any change):
  answers.parquet   (qid, k, pred, gold, f1, correct, error_type, gold_pos_frac, ctx_tokens)
  attn_mass.parquet (qid, k, layer, head, attn_mass_gold, gold_pos_frac, ctx_tokens)
  sae_acts.h5       /q/<qid>/k<k>/layer<L> -> datasets feat_ids[int32 topk], acts[float32 topk]
Streaming format during the sweep is JSONL (append-only, resumable); compacted to
sorted parquet at the end so a re-executed Run 1 is byte-identical.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
import yaml

SCHEMA_VERSION = 2

ANSWER_COLS = ["qid", "k", "pred", "gold", "f1", "correct", "error_type",
               "gold_pos_frac", "ctx_tokens"]
ATTN_COLS = ["qid", "k", "layer", "head", "attn_mass_gold", "gold_pos_frac",
             "ctx_tokens"]


def status(stage: str, ok: bool, detail: str = "") -> None:
    print(f"STATUS: {stage} {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def die(stage: str, detail: str) -> None:
    status(stage, False, detail)
    sys.exit(1)


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("schema_version") != SCHEMA_VERSION:
        die("config", f"config schema_version {cfg.get('schema_version')} != {SCHEMA_VERSION}")
    return cfg


def stable_hash(s: str) -> int:
    """Deterministic across processes (python hash() is salted)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % (2 ** 31)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- JSONL stream
def jsonl_append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def jsonl_load(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def done_keys(answers_jsonl: Path) -> set:
    """(qid, k) pairs already completed — resume point for the sweep."""
    return {(r["qid"], r["k"]) for r in jsonl_load(answers_jsonl)}


def compact_to_parquet(jsonl_path: Path, parquet_path: Path, cols, sort_by) -> pd.DataFrame:
    """Deduplicate (last write wins), sort deterministically, write parquet."""
    rows = jsonl_load(jsonl_path)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=sort_by, keep="last")
    df = df[cols].sort_values(sort_by).reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)
    return df


# --------------------------------------------------------------------- HDF5
def h5_group_name(qid: str, k: int, layer: int) -> str:
    return f"/q/{qid}/k{k}/layer{layer}"


def h5_has(h5path: Path, qid: str, k: int) -> bool:
    if not h5path.exists():
        return False
    with h5py.File(h5path, "r") as f:
        return f"/q/{qid}/k{k}" in f


def h5_write_acts(h5path: Path, qid: str, k: int, layer_acts: dict) -> None:
    """layer_acts: {layer: (feat_ids ndarray, acts ndarray)}; overwrite-safe."""
    with h5py.File(h5path, "a") as f:
        for layer, (fids, acts) in layer_acts.items():
            g = h5_group_name(qid, k, layer)
            if g in f:
                del f[g]
            grp = f.create_group(g)
            grp.create_dataset("feat_ids", data=fids.astype("int32"))
            grp.create_dataset("acts", data=acts.astype("float32"))


def h5_read_acts(h5path: Path, qid: str, k: int, layer: int):
    with h5py.File(h5path, "r") as f:
        g = f[h5_group_name(qid, k, layer)]
        return g["feat_ids"][:], g["acts"][:]


# ------------------------------------------------------------------ manifest
def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                              capture_output=True, check=True).stdout.strip()
    except Exception:
        return "no-git"


def _env_hash() -> str:
    try:
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                                capture_output=True, check=True).stdout
        return hashlib.sha256(frozen.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def write_manifest(path: Path, cfg: dict, repo: Path, extra: dict | None = None) -> None:
    m = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "git_sha": _git_sha(repo),
        "env_hash": _env_hash(),
        "seed": cfg["run"]["seed"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }
    if extra:
        m.update(extra)
    write_json(path, m)


def check_manifest(path: Path) -> dict:
    if not path.exists():
        die("artifact-check", f"missing manifest {path}; run Run 1 first")
    m = read_json(path)
    if m.get("schema_version") != SCHEMA_VERSION:
        die("artifact-check",
            f"manifest schema_version {m.get('schema_version')} != code {SCHEMA_VERSION}")
    return m
