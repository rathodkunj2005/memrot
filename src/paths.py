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


def artifacts_dir() -> Path:
    return _sub(scratch_root(), "artifacts")


def smoke_dir() -> Path:
    """Smoke-test artifacts are quarantined so they never pollute the real run."""
    return _sub(scratch_root(), "smoke_artifacts")


def analysis_dir() -> Path:
    return _sub(scratch_root(), "analysis")


def figures_dir() -> Path:
    return _sub(artifacts_dir(), "figures")


def data_dir() -> Path:
    return _sub(scratch_root(), "data")


def paper_dir() -> Path:
    return _sub(repo_root(), "paper")


def config_path() -> Path:
    return repo_root() / "config" / "config.yaml"
