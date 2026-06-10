"""LongMemEval-S -> per-question records + distractor session pool.

Selection rules (v2 design):
  - drop abstention questions (qid endswith '_abs'; they have no gold answer)
  - keep questions with EXACTLY ONE gold evidence session (answer_session_ids)
  - gold session must tokenize to <= benchmark.gold_max_tokens
  - take the first benchmark.n_questions in sorted-qid order (determinism)

Distractor pool: every NON-gold haystack session of every kept question. In
LongMemEval-S these filler sessions originate from other users/questions and are
topic-matched by benchmark construction, which is exactly the "other questions'
sessions, topic-matched where possible" pool the design calls for. With
haystack.hard_distractors=true the builder restricts to the owning question's own
non-gold sessions (hardest, fully topic-matched mode).
"""
import json

import pandas as pd
from huggingface_hub import hf_hub_download

import io_utils
import paths


def _session_text(turns) -> str:
    lines = []
    for t in turns:
        role = str(t.get("role", "user")).upper()
        content = str(t.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _load_raw(cfg) -> list:
    b = cfg["benchmark"]
    if b.get("local_json"):
        path = b["local_json"]
    else:
        path = hf_hub_download(repo_id=b["hf_repo"], filename=b["hf_filename"],
                               repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def prepare(cfg, tokenizer, n_questions=None, force=False):
    """Returns (records_df, sessions_df); writes both parquets to artifacts."""
    art = paths.artifacts_dir()
    rec_path = art / "longmemeval_records.parquet"
    ses_path = art / "longmemeval_sessions.parquet"
    if rec_path.exists() and ses_path.exists() and not force:
        io_utils.status("data_prep", True, f"cached ({rec_path})")
        return pd.read_parquet(rec_path), pd.read_parquet(ses_path)

    raw = _load_raw(cfg)
    n_target = n_questions or cfg["benchmark"]["n_questions"]
    gold_max = cfg["benchmark"]["gold_max_tokens"]

    def ntok(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    records, sessions = [], []
    n_abs = n_multi = n_long = n_missing = 0
    for item in sorted(raw, key=lambda x: str(x["question_id"])):
        qid = str(item["question_id"])
        if qid.endswith("_abs"):
            n_abs += 1
            continue
        gold_ids = list(item.get("answer_session_ids", []))
        if len(gold_ids) != 1:
            n_multi += 1
            continue
        hay_ids = list(item["haystack_session_ids"])
        if gold_ids[0] not in hay_ids:
            n_missing += 1
            continue
        gold_idx = hay_ids.index(gold_ids[0])
        sess = item["haystack_sessions"]

        gold_text = _session_text(sess[gold_idx])
        gold_tok = ntok(gold_text)
        if gold_tok > gold_max or gold_tok == 0:
            n_long += 1
            continue

        records.append({
            "qid": qid,
            "question": str(item["question"]),
            "answer": str(item["answer"]),
            "question_type": str(item.get("question_type", "")),
            "question_date": str(item.get("question_date", "")),
            "gold_session_id": gold_ids[0],
            "gold_text": gold_text,
            "gold_tokens": gold_tok,
        })
        for i, (sid, turns) in enumerate(zip(hay_ids, sess)):
            if i == gold_idx:
                continue
            text = _session_text(turns)
            if not text.strip():
                continue
            sessions.append({
                "owner_qid": qid,
                "session_uid": f"{qid}::{sid}",
                "text": text,
                "n_tokens": ntok(text),
            })

    records_df = pd.DataFrame(records).sort_values("qid").reset_index(drop=True)
    if len(records_df) < n_target:
        io_utils.status("data_prep", True,
                        f"WARNING only {len(records_df)} eligible questions "
                        f"(< requested {n_target}); using all of them")
    records_df = records_df.head(n_target)
    keep_qids = set(records_df["qid"])
    sessions_df = (pd.DataFrame(sessions)
                   .loc[lambda d: d["owner_qid"].isin(keep_qids)]
                   .sort_values("session_uid").reset_index(drop=True))

    records_df.to_parquet(rec_path, index=False)
    sessions_df.to_parquet(ses_path, index=False)
    io_utils.status(
        "data_prep", True,
        f"kept {len(records_df)} questions, {len(sessions_df)} pool sessions "
        f"(skipped: {n_abs} abstention, {n_multi} multi/zero-gold, "
        f"{n_long} gold>{gold_max}tok, {n_missing} gold-missing)")
    return records_df, sessions_df
