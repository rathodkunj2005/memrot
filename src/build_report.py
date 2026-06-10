"""RUN 3 MAIN: fill report_template.md from results_summary.json, copy figures.

    python -u src/build_report.py --config config/config.yaml
"""
import argparse
import re
import shutil

import io_utils
import paths
from io_utils import die, status


def _fmt(v):
    if isinstance(v, float):
        if v != v:                  # NaN
            return "n/a"
        if 0 < abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:.3f}"
    if isinstance(v, (list, dict)):
        return re.sub(r"[\"']", "", str(v))
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.config_path()))
    args = ap.parse_args()
    cfg = io_utils.load_config(args.config)

    summ_path = paths.analysis_dir() / "results_summary.json"
    if not summ_path.exists():
        die("report", f"missing {summ_path}; run Run 2 first")
    s = io_utils.read_json(summ_path)
    if s["run1_manifest"]["schema_version"] != io_utils.SCHEMA_VERSION:
        die("report", "schema_version mismatch between summary and code")

    values = dict(s)
    values.update(s["run1_manifest"])          # git_sha, env_hash, seed, schema_version
    template = (paths.repo_root() / "src" / "report_template.md").read_text()

    unfilled = []
    def sub(m):
        key = m.group(1)
        if key in values:
            return _fmt(values[key])
        unfilled.append(key)
        return m.group(0)
    paper = re.sub(r"\{\{(\w+)\}\}", sub, template)
    if unfilled:
        die("report", f"unfilled placeholders: {sorted(set(unfilled))}")

    out = paths.paper_dir()
    figs_src = paths.figures_dir()
    (out / "figures").mkdir(exist_ok=True)
    n_figs = 0
    for f in sorted(figs_src.glob("*.pdf")):
        shutil.copy2(f, out / "figures" / f.name)
        n_figs += 1
    (out / "paper.md").write_text(paper)
    status("run3", True, f"{out/'paper.md'} written ({n_figs} figures embedded)")


if __name__ == "__main__":
    main()
