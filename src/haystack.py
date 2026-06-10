"""Controlled-haystack builder.

For a question record and fill level k:
  - sample k distractor sessions with a per-(qid,k) seeded RNG,
  - truncate each distractor to a FIXED token cap (auto = budget/k_max), so total
    context grows ~linearly with k and k=k_max lands just under the budget,
  - place the gold session at a uniformly random slot among k+1 (logged as
    gold_pos_frac = gold_start_token / total_tokens),
  - assemble input_ids by concatenating per-segment tokenizations (BOS first),
    which makes the gold token span EXACT by construction.

Determinism: rng seed = [stable_hash(qid), k] (plan's hash(qid) made
process-stable via sha256, see io_utils.stable_hash).
"""
import numpy as np

from io_utils import stable_hash

HEADER = ("You are given excerpts from a user's past chat sessions, followed by "
          "a question. Answer the question using only information from the "
          "sessions. Be concise.\n\n")


def distractor_cap(cfg) -> int:
    h = cfg["haystack"]
    if h["distractor_token_cap"] != "auto":
        return int(h["distractor_token_cap"])
    budget = cfg["model"]["haystack_token_budget"]
    k_max = max(cfg["benchmark"]["k_levels"])
    return (budget - cfg["benchmark"]["gold_max_tokens"] - h["overhead_tokens"]) // k_max


class HaystackBuilder:
    def __init__(self, cfg, tokenizer, sessions_df):
        self.cfg = cfg
        self.tok = tokenizer
        self.cap = distractor_cap(cfg)
        self.budget = cfg["model"]["haystack_token_budget"]
        self.hard = cfg["haystack"]["hard_distractors"]
        # pre-index the pool; sorted uids -> deterministic sampling
        self.pool_uids = sessions_df["session_uid"].to_numpy()
        self.pool_owner = sessions_df["owner_qid"].to_numpy()
        self.pool_text = sessions_df["text"].to_numpy()
        self.uid_order = np.argsort(self.pool_uids)

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _pick_distractors(self, qid, k, rng):
        if k == 0:
            return []
        if self.hard:
            mask = self.pool_owner == qid          # question's own haystack only
        else:
            mask = np.ones(len(self.pool_uids), bool)
        cand = self.uid_order[mask[self.uid_order]]
        if len(cand) < k:
            raise RuntimeError(f"distractor pool too small for {qid} k={k}")
        return list(rng.choice(cand, size=k, replace=False))

    def build(self, record, k):
        qid = record["qid"]
        rng = np.random.default_rng([stable_hash(qid), k])
        idxs = self._pick_distractors(qid, k, rng)
        gold_slot = int(rng.integers(0, k + 1))

        gold_span, duids = None, []
        ids = [self.tok.bos_token_id] + self._ids(HEADER)
        d_iter = iter(idxs)
        for slot in range(k + 1):
            if slot == gold_slot:
                text = record["gold_text"]
                seg = self._ids(f"=== Session {slot + 1} ===\n")
                ids += seg
                body = self._ids(text + "\n\n")
                gold_span = (len(ids), len(ids) + len(body))
                ids += body
            else:
                i = next(d_iter)
                duids.append(str(self.pool_uids[i]))
                body = self._ids(str(self.pool_text[i]))[: self.cap]
                ids += self._ids(f"=== Session {slot + 1} ===\n")
                ids += body + self._ids("\n\n")
        ids += self._ids(
            f"=== Question ===\n{record['question']}\n\n=== Answer ===\n"
            "The answer is")

        if len(ids) > self.budget:
            raise RuntimeError(
                f"haystack for {qid} k={k} is {len(ids)} tokens > budget {self.budget}")
        return {
            "input_ids": ids,
            "gold_span": gold_span,
            "gold_pos_frac": gold_span[0] / len(ids),
            "ctx_tokens": len(ids),
            "distractor_uids": duids,
        }
