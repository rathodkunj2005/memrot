"""LongMemEval-S -> per-question records + distractor session pool.

Selection rules (v2 design, amended after measuring the dataset):
  - drop abstention questions (qid endswith '_abs'; they have no gold answer)
  - drop benchmark.exclude_question_types (single-session-preference: their gold
    "answer" is a ~390-char prose preference-rubric, not a literal string; the
    heuristic substring/F1 grader cannot score them — they need an LLM judge)
  - keep questions with EXACTLY ONE gold evidence session (answer_session_ids);
    LongMemEval-S has exactly 170 such questions
  - gold session must fit benchmark.gold_max_tokens; sessions that are longer
    (the median evidence session is ~2.5k Gemma tokens, so a hard filter would
    keep only ~50 questions) are TRIMMED to a turn window centered on the
    has_answer evidence turn(s) — flagged in records as gold_trimmed. Questions
    whose trimming destroys the answer fail at k=0 and drop out of the paired
    analysis by design.
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


def _trim_gold(turns, ntok, cap):
    """Trim a too-long evidence session to a turn window around the has_answer
    turn(s): start from the evidence turns, then alternately add the preceding /
    following neighbor while the rendered session stays under cap."""
    ev = [i for i, t in enumerate(turns) if t.get("has_answer")]
    if not ev:
        return None
    lo, hi = min(ev), max(ev)
    if ntok(_session_text(turns[lo:hi + 1])) > cap:
        # evidence turns alone exceed cap: hard-truncate (k=0 gate catches damage)
        return turns[lo:hi + 1]
    while True:
        grew = False
        if lo > 0 and ntok(_session_text(turns[lo - 1:hi + 1])) <= cap:
            lo -= 1
            grew = True
        if hi < len(turns) - 1 and ntok(_session_text(turns[lo:hi + 2])) <= cap:
            hi += 1
            grew = True
        if not grew:
            return turns[lo:hi + 1]


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
    exclude_types = set(cfg["benchmark"].get("exclude_question_types", []))

    def ntok(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    records, sessions = [], []
    n_abs = n_multi = n_long = n_missing = n_xtype = 0
    for item in sorted(raw, key=lambda x: str(x["question_id"])):
        qid = str(item["question_id"])
        if qid.endswith("_abs"):
            n_abs += 1
            continue
        if str(item.get("question_type", "")) in exclude_types:
            n_xtype += 1
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

        gold_turns = sess[gold_idx]
        gold_text = _session_text(gold_turns)
        gold_tok = ntok(gold_text)
        gold_trimmed = False
        if gold_tok > gold_max:
            trimmed = _trim_gold(gold_turns, ntok, gold_max)
            if trimmed is None:
                n_long += 1
                continue
            gold_text = _session_text(trimmed)
            # if even the evidence turns alone exceed cap, hard-truncate tokens
            tids = tokenizer(gold_text, add_special_tokens=False)["input_ids"]
            if len(tids) > gold_max:
                gold_text = tokenizer.decode(tids[:gold_max])
            gold_tok = ntok(gold_text)
            gold_trimmed = True
        if gold_tok == 0:
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
            "gold_trimmed": gold_trimmed,
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
        f"kept {len(records_df)} questions "
        f"({int(records_df['gold_trimmed'].sum())} gold-trimmed), "
        f"{len(sessions_df)} pool sessions (skipped: {n_abs} abstention, "
        f"{n_multi} multi/zero-gold, {n_long} untrimmable, {n_missing} gold-missing, "
        f"{n_xtype} excluded-type)")
    return records_df, sessions_df
