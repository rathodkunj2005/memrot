"""RUN 1 MAIN: smoke test -> data prep -> layer scan -> head ID -> k-sweep -> verify.

Every stage is idempotent (skips if its artifact exists; --force overrides) and
the sweep resumes from the last completed (qid, k). Usage:
    python -u src/run_experiment.py --config config/config.yaml --smoke-test
    python -u src/run_experiment.py --config config/config.yaml --full
"""
import argparse
import time

import numpy as np
import torch
from huggingface_hub import HfApi

import data_prep
import grade
import instrument
import io_utils
import layer_scan
import model_loader
import paths
import retrieval_heads
import sae_loader
from haystack import HaystackBuilder
from io_utils import die, status

SMOKE_SAE_LAYERS = [12]   # machinery check only; real layers come from layer_scan


def check_auth(cfg):
    try:
        HfApi().model_info(cfg["model"]["name"])
        status("auth", True, f"token can access {cfg['model']['name']}")
    except Exception as e:
        die("auth",
            f"cannot access {cfg['model']['name']} ({type(e).__name__}). "
            "Human action: (1) accept the license at "
            f"https://huggingface.co/{cfg['model']['name']} and (2) put a token at "
            "$HF_HOME/token  (huggingface-cli login)")


def evaluate_one(cfg, model, tok, builder, instr, saes, rec, k, dist_text_by_uid):
    """One instrumented pass. Returns (answer_row, attn_rows, layer_acts)."""
    hs = builder.build(rec, k)
    pred, gen_ids = model_loader.greedy_generate(model, tok, hs["input_ids"], cfg)
    attn, resid = instr.run(hs["input_ids"], gen_ids, hs["gold_span"])

    layer_acts = {}
    for layer, sae in saes.items():
        fids, acts = sae_loader.topk_acts(sae, resid[layer], cfg["sae"]["topk_store"])
        layer_acts[layer] = (fids, acts)

    dtexts = [dist_text_by_uid[u] for u in hs["distractor_uids"]]
    gr = grade.grade(pred, rec["answer"], dtexts, cfg)
    ans_row = {"qid": rec["qid"], "k": int(k), "pred": pred, "gold": rec["answer"],
               "f1": gr["f1"], "correct": bool(gr["correct"]),
               "error_type": gr["error_type"],
               "gold_pos_frac": round(hs["gold_pos_frac"], 6),
               "ctx_tokens": int(hs["ctx_tokens"])}
    attn_rows = [{"qid": rec["qid"], "k": int(k), "layer": int(l), "head": int(h),
                  "attn_mass_gold": round(float(attn[l, h]), 6),
                  "gold_pos_frac": round(hs["gold_pos_frac"], 6),
                  "ctx_tokens": int(hs["ctx_tokens"])}
                 for l in range(attn.shape[0]) for h in range(attn.shape[1])]
    return ans_row, attn_rows, layer_acts


def smoke_test(cfg, model, tok, records_df, sessions_df):
    t0 = time.time()
    sdir = paths.smoke_dir()
    builder = HaystackBuilder(cfg, tok, sessions_df)
    instr = instrument.Instrumentor(model, SMOKE_SAE_LAYERS)
    saes = {}
    for L in SMOKE_SAE_LAYERS:
        saes[L], l0 = sae_loader.load_sae(cfg, L)
        assert saes[L].d_sae == 16384, f"unexpected SAE width {saes[L].d_sae}"

    # hook sanity: mass over the FULL key range must be ~1 for every head
    rec0 = records_df.iloc[0].to_dict()
    hs = builder.build(rec0, 1)
    full_mass, _ = instr.run(hs["input_ids"], [1], (0, hs["ctx_tokens"]))
    # sliding layers can't see the whole range at long ctx; smoke ctx is short, so all ~1
    if not (np.abs(full_mass - 1.0) < 0.02).all():
        die("smoke", f"attention mass over full range != 1 (min {full_mass.min():.3f}); "
                     "hook reduction is mis-indexed")

    dist = dict(zip(sessions_df["session_uid"], sessions_df["text"]))
    recs = records_df.head(cfg["run"]["smoke_test_n"]).to_dict("records")
    n_k0_correct, n_pass = 0, 0
    for rec in recs:
        for k in cfg["benchmark"]["k_levels"]:
            ans, attn_rows, layer_acts = evaluate_one(
                cfg, model, tok, builder, instr, saes, rec, k, dist)
            io_utils.jsonl_append(sdir / "answers.jsonl", ans)
            io_utils.h5_write_acts(sdir / "sae_acts.h5", rec["qid"], k, layer_acts)
            n_pass += 1
            if k == 0 and ans["correct"]:
                n_k0_correct += 1
            print(f"  smoke {rec['qid']} k={k} ctx={ans['ctx_tokens']} "
                  f"correct={ans['correct']} pred={ans['pred'][:60]!r}", flush=True)

    acc0 = n_k0_correct / len(recs)
    gate = cfg["analysis"]["k0_accuracy_gate"]
    detail = (f"{n_pass} passes, k0 acc {acc0:.2f}, {time.time()-t0:.0f}s")
    if acc0 < gate:
        die("smoke", detail + f" — k0 accuracy < {gate}. Diagnosis hints: inspect "
            "smoke answers.jsonl preds; if the base model rambles, switch "
            "model.name to google/gemma-2-2b-it; also check gold-session filter "
            "and grading thresholds.")
    status("smoke", True, detail)


def sweep(cfg, model, tok, records_df, sessions_df, force=False):
    art = paths.artifacts_dir()
    ans_jl, attn_jl = art / "answers.jsonl", art / "attn_mass.jsonl"
    h5 = art / "sae_acts.h5"
    if (art / "answers.parquet").exists() and not force:
        status("sweep", True, "cached (answers.parquet exists)")
        return

    layers = sae_loader.sae_layers(cfg)
    saes = {L: sae_loader.load_sae(cfg, L)[0] for L in layers}
    builder = HaystackBuilder(cfg, tok, sessions_df)
    instr = instrument.Instrumentor(model, layers)
    dist = dict(zip(sessions_df["session_uid"], sessions_df["text"]))
    done = io_utils.done_keys(ans_jl)
    k_levels = cfg["benchmark"]["k_levels"]
    total = len(records_df) * len(k_levels)
    n_done = len(done)
    status("sweep_resume", True, f"{n_done}/{total} passes already complete")

    pace_warned = False
    for rec in records_df.to_dict("records"):
        for k in k_levels:
            if (rec["qid"], k) in done:
                continue
            t0 = time.time()
            ans, attn_rows, layer_acts = evaluate_one(
                cfg, model, tok, builder, instr, saes, rec, k, dist)
            # h5 first, answers last: a (qid,k) in answers.jsonl implies its h5 group exists
            io_utils.h5_write_acts(h5, rec["qid"], k, layer_acts)
            for r in attn_rows:
                io_utils.jsonl_append(attn_jl, r)
            io_utils.jsonl_append(ans_jl, ans)
            n_done += 1
            dt = time.time() - t0
            if dt > cfg["run"]["max_sec_per_pass"] and not pace_warned:
                pace_warned = True
                status("sweep_pace", True,
                       f"WARNING {dt:.1f}s/pass > {cfg['run']['max_sec_per_pass']}s "
                       "guardrail — consider n_questions=200 and dropping k=2")
            if n_done % 50 == 0:
                print(f"  sweep {n_done}/{total} ({dt:.1f}s/pass)", flush=True)

    io_utils.compact_to_parquet(ans_jl, art / "answers.parquet",
                                io_utils.ANSWER_COLS, ["qid", "k"])
    io_utils.compact_to_parquet(attn_jl, art / "attn_mass.parquet",
                                io_utils.ATTN_COLS, ["qid", "k", "layer", "head"])
    status("sweep", True, f"{n_done}/{total} passes; parquets compacted")


def verify(cfg, records_df):
    import pandas as pd
    art = paths.artifacts_dir()
    rep = {"checks": {}, "ok": True}

    def chk(name, ok, detail=""):
        rep["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        rep["ok"] &= bool(ok)

    ans = pd.read_parquet(art / "answers.parquet")
    attn = pd.read_parquet(art / "attn_mass.parquet")
    n_q, n_k = len(records_df), len(cfg["benchmark"]["k_levels"])
    n_heads_total = attn.groupby(["layer", "head"]).ngroups
    chk("answers_rows", len(ans) == n_q * n_k, f"{len(ans)} vs {n_q * n_k}")
    chk("attn_rows", len(attn) == n_q * n_k * n_heads_total,
        f"{len(attn)} vs {n_q * n_k * n_heads_total}")
    chk("answers_nan", not ans[["f1", "gold_pos_frac"]].isna().any().any())
    chk("attn_nan", not attn["attn_mass_gold"].isna().any())

    layers = sae_loader.sae_layers(cfg)
    missing = sum(1 for r in ans.itertuples()
                  if not io_utils.h5_has(art / "sae_acts.h5", r.qid, r.k))
    chk("h5_groups", missing == 0, f"{missing} (qid,k) missing in sae_acts.h5")

    acc0 = ans.loc[ans.k == 0, "correct"].mean()
    gate = cfg["analysis"]["k0_accuracy_gate"]
    chk("k0_accuracy", acc0 >= gate,
        f"{acc0:.3f} (gate {gate}); if FAIL: gold-session filter or grading is "
        "broken, or base model needs the -it variant — see README fallbacks")
    rep["k0_accuracy"] = float(acc0)
    rep["acc_by_k"] = ans.groupby("k")["correct"].mean().to_dict()
    rep["sae_layers"] = layers
    io_utils.write_json(art / "run1_verify.json", rep)
    status("verify", rep["ok"], f"k0_acc={acc0:.3f} acc_by_k={rep['acc_by_k']}")
    if not rep["ok"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.config_path()))
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = io_utils.load_config(args.config)

    check_auth(cfg)
    model, tok, info = model_loader.load(cfg)
    records_df, sessions_df = data_prep.prepare(cfg, tok)

    if args.smoke_test:
        smoke_test(cfg, model, tok, records_df, sessions_df)
    if args.full:
        builder = HaystackBuilder(cfg, tok, sessions_df)
        scan = layer_scan.run(cfg, model, tok, builder, records_df, force=args.force)
        rh = retrieval_heads.identify(cfg, model, tok, sessions_df, force=args.force)
        sweep(cfg, model, tok, records_df, sessions_df, force=args.force)
        io_utils.write_manifest(
            paths.artifacts_dir() / "run1_manifest.json", cfg, paths.repo_root(),
            extra={"sae_layers": scan["chosen_layers"],
                   "l0_per_layer": scan["l0_per_layer"],
                   "n_retrieval_heads": len(rh["retrieval_heads"]),
                   "n_questions": len(records_df)})
        verify(cfg, records_df)
        status("run1", True, "all stages complete")


if __name__ == "__main__":
    main()
