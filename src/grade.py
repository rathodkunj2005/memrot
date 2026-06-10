"""Answer grading: exact/F1 + error-type classification.

correct        gold answer (normalized) is a substring of the prediction, OR
               SQuAD-style token F1 >= grading.f1_threshold
abstain        prediction matches refusal/uncertainty patterns
distractor     not correct, and the prediction's content tokens overlap a
               distractor session's text by >= grading.distractor_overlap
hallucination  everything else

These are heuristic proxies for LongMemEval's LLM-judge grading; thresholds live
in config and the limitation is stated in the report template.
"""
import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}
_ABSTAIN = re.compile(
    r"(i\s+(do\s*n.?t|don.t)\s+know|not\s+(mentioned|stated|provided|specified)|"
    r"no\s+information|cannot\s+(answer|determine)|unable\s+to|i.m\s+not\s+sure|"
    r"there\s+is\s+no\s+)", re.I)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c if c not in string.punctuation else " " for c in s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def _overlap_frac(pred_toks, text_norm_tokens: set) -> float:
    if not pred_toks:
        return 0.0
    return sum(t in text_norm_tokens for t in pred_toks) / len(pred_toks)


def grade(pred: str, gold: str, distractor_texts, cfg) -> dict:
    g = cfg["grading"]
    f1 = token_f1(pred, gold)
    gold_n, pred_n = normalize(gold), normalize(pred)
    correct = bool(gold_n and gold_n in pred_n) or f1 >= g["f1_threshold"]

    if correct:
        etype = "correct"
    elif _ABSTAIN.search(pred):
        etype = "abstain"
    else:
        etype = "hallucination"
        pred_toks = [t for t in pred_n.split() if len(t) > 2]
        for dt in distractor_texts:
            if _overlap_frac(pred_toks, set(normalize(dt).split())) >= g["distractor_overlap"]:
                etype = "distractor"
                break
    return {"f1": round(f1, 4), "correct": correct, "error_type": etype}
