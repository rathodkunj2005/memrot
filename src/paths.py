"""Single source of truth for all filesystem locations.

Resolution rules:
  - Repo root: $MEMROT_ROOT if set (sbatch sets it to $SLURM_SUBMIT_DIR),
    else the parent of this file's directory.
  - Scratch root: $SCRATCH/memrot. $SCRATCH must be set (it is on all CHPC nodes).
All other modules import from here; no path literals anywhere else.
"""
import os
from pathlib import Path


def repo_root() -> Path:
    root = os.environ.get("MEMROT_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parent.parent


def scratch_root() -> Path:
    s = os.environ.get("SCRATCH")
    if not s:
        raise RuntimeError(
            "SCRATCH environment variable is not set. On CHPC it should be "
            "/scratch/general/vast/$USER. Export it or run under SLURM."
        )
    p = Path(s) / "memrot"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(parent: Path, name: str) -> Path:
    p = parent / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _model_seg() -> str:
    """Per-model namespace segment from $MEMROT_MODEL_ID (e.g. '2b'/'9b'/'27b').

    Lets several models' artifacts coexist under $SCRATCH/memrot without
    clobbering each other (the model-scaling study). Empty -> legacy un-namespaced
    layout, preserved for backward compatibility.
    """
    return os.environ.get("MEMROT_MODEL_ID", "").strip()


def artifacts_dir() -> Path:
    base = _sub(scratch_root(), "artifacts")
    seg = _model_seg()
    return _sub(base, seg) if seg else base


def smoke_dir() -> Path:
    """Smoke-test artifacts are quarantined so they never pollute the real run."""
    base = _sub(scratch_root(), "smoke_artifacts")
    seg = _model_seg()
    return _sub(base, seg) if seg else base


def analysis_dir() -> Path:
    base = _sub(scratch_root(), "analysis")
    seg = _model_seg()
    return _sub(base, seg) if seg else base


def figures_dir() -> Path:
    return _sub(artifacts_dir(), "figures")


def data_dir() -> Path:
    return _sub(scratch_root(), "data")


def paper_dir() -> Path:
    return _sub(repo_root(), "paper")


def config_path() -> Path:
    return repo_root() / "config" / "config.yaml"
