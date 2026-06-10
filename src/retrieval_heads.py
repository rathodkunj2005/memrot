"""Retrieval-head identification via the copy-score test (Wu et al., 2024 style).

Synthetic needles built from LongMemEval vocabulary: a "secret access code" of
needle_words random words is hidden in ~probe_haystack_tokens of distractor text;
the prompt asks the model to repeat it. While greedily decoding, a head scores a
copy event when the emitted token equals the next needle token AND that head's
attention argmax points at that needle token's position. copy_score = copy
events / needle length, averaged over probe_examples probes.

Probes are kept short (<= ~2k tokens) so generate(output_attentions=True) is
memory-safe; the main 7.5k sweep never does this (it uses reduce-in-hook).
"""
import numpy as np
import torch

import io_utils
import paths


def _vocab(sessions_df, rng, n=4000):
    words = set()
    for text in sessions_df["text"].head(500):
        for w in str(text).split():
            w = "".join(c for c in w if c.isalpha())
            if 4 <= len(w) <= 10:
                words.add(w.lower())
    words = sorted(words)
    return [words[i] for i in rng.choice(len(words), size=min(n, len(words)),
                                         replace=False)]


def _build_probe(tok, vocab, rng, cfg):
    rcfg = cfg["retrieval_heads"]
    code = list(rng.choice(vocab, size=rcfg["needle_words"], replace=False))
    needle_text = " " + " ".join(code) + "."
    filler_words = rng.choice(vocab, size=rcfg["probe_haystack_tokens"], replace=False
                              if len(vocab) >= rcfg["probe_haystack_tokens"] else True)

    def ids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    pre_n = int(rng.integers(50, rcfg["probe_haystack_tokens"] - 50))
    pre = " ".join(filler_words[:pre_n // 2])
    post = " ".join(filler_words[pre_n // 2: rcfg["probe_haystack_tokens"] // 2])

    out = [tok.bos_token_id] + ids("Some notes:\n" + pre + "\nThe secret access code is")
    needle_ids = ids(needle_text)
    needle_pos = list(range(len(out), len(out) + len(needle_ids)))
    out += needle_ids
    out += ids("\n" + post + "\nQuestion: What is the secret access code?\n"
               "Answer: The secret access code is")
    return out, needle_ids, needle_pos


@torch.no_grad()
def identify(cfg, model, tok, sessions_df, force=False):
    art = paths.artifacts_dir()
    out_path = art / "retrieval_heads.json"
    if out_path.exists() and not force:
        io_utils.status("retrieval_heads", True, f"cached ({out_path})")
        return io_utils.read_json(out_path)

    rcfg = cfg["retrieval_heads"]
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    rng = np.random.default_rng(cfg["run"]["seed"])
    vocab = _vocab(sessions_df, rng)
    scores = np.zeros((L, H))

    for _ in range(rcfg["probe_examples"]):
        ids, needle_ids, needle_pos = _build_probe(tok, vocab, rng, cfg)
        inp = torch.tensor([ids], device="cuda")
        out = model.generate(
            inp, max_new_tokens=len(needle_ids) + 4, do_sample=False, num_beams=1,
            output_attentions=True, return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = out.sequences[0, len(ids):].tolist()
        probe_scores = np.zeros((L, H))
        nxt = 0   # next needle token expected to be copied
        for step, tok_id in enumerate(gen):
            if nxt >= len(needle_ids):
                break
            if tok_id != needle_ids[nxt]:
                continue
            tgt = needle_pos[nxt]
            for layer in range(L):
                w = out.attentions[step][layer]          # [1,H,T,T] or [1,H,1,T]
                row = w[0, :, -1, :]                     # [H, T_keys]
                probe_scores[layer] += (row.argmax(dim=-1) == tgt).float().cpu().numpy()
            nxt += 1
        scores += probe_scores / len(needle_ids)         # per-probe normalization
        del out
    scores /= rcfg["probe_examples"]

    flat = [{"layer": int(l), "head": int(h), "copy_score": round(float(scores[l, h]), 4),
             "is_sliding_layer": bool(l % 2 == 0)}
            for l in range(L) for h in range(H)]
    heads = [r for r in flat if r["copy_score"] >= rcfg["copy_score_threshold"]]
    heads = sorted(heads, key=lambda r: -r["copy_score"])[: rcfg["top_n_heads"]]
    result = {"retrieval_heads": heads, "all_scores": flat,
              "threshold": rcfg["copy_score_threshold"]}
    io_utils.write_json(out_path, result)
    io_utils.status("retrieval_heads", True,
                    f"{len(heads)} heads >= {rcfg['copy_score_threshold']} "
                    f"(top: {[(r['layer'], r['head']) for r in heads[:5]]})")
    return result
