"""Stage: pick SAE layers empirically (replaces v1's guessed [5,9,12,17,20]).

On layer_scan.n_questions questions at k=0 and k=k_high:
  - capture resid_post at the last prompt position for ALL layers (cheap,
    output_hidden_states), generate + grade each condition,
  - for each layer with an available Gemma Scope SAE (all 26 by construction),
    encode the captured vectors and score the layer by correct-vs-wrong
    separability: mean |Cohen's d| of the top-N features ranked by |d|.
  - Fallback: if either outcome class has < 5 samples, score by the k0-vs-khigh
    activation shift instead (same statistic, conditions as classes).
Writes layer_scan.json {layer_scores, chosen_layers, l0_per_layer, mode}.
"""
import numpy as np
import torch

import instrument
import io_utils
import model_loader
import paths
import sae_loader
from grade import grade


def _cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-feature |d| between sample groups a [na, F] and b [nb, F]."""
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a.std(0), b.std(0)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2) + 1e-6
    return np.abs(ma - mb) / pooled


def run(cfg, model, tok, builder, records_df, force=False):
    art = paths.artifacts_dir()
    out_path = art / "layer_scan.json"
    if out_path.exists() and not force:
        io_utils.status("layer_scan", True, f"cached ({out_path})")
        return io_utils.read_json(out_path)

    n = cfg["layer_scan"]["n_questions"]
    k_high = cfg["layer_scan"]["k_high"]
    recs = records_df.head(n).to_dict("records")

    vecs, labels_correct, labels_cond = [], [], []   # vecs: list of {layer: vec}
    for rec in recs:
        for k in (0, k_high):
            hs = builder.build(rec, k)
            pred, _ = model_loader.greedy_generate(model, tok, hs["input_ids"], cfg)
            gr = grade(pred, rec["answer"], [], cfg)
            vecs.append(instrument.residuals_all_layers(model, hs["input_ids"]))
            labels_correct.append(gr["correct"])
            labels_cond.append(k == 0)

    yc = np.array(labels_correct)
    use_correct = yc.sum() >= 5 and (~yc).sum() >= 5
    y = yc if use_correct else np.array(labels_cond)
    mode = "correct_vs_wrong" if use_correct else "k0_vs_khigh_shift"

    table = sae_loader.available_l0s(cfg)
    top_n = cfg["layer_scan"]["top_features_per_layer"]
    scores, l0s = {}, {}
    for layer in sorted(table):
        sae, l0 = sae_loader.load_sae(cfg, layer)
        X = torch.stack([v[layer] for v in vecs]).to("cuda")
        A = sae.encode(X).cpu().numpy()
        d = _cohens_d(A[y], A[~y])
        scores[layer] = float(np.sort(d)[-top_n:].mean())
        l0s[layer] = l0
        del sae
        torch.cuda.empty_cache()

    chosen = sorted(sorted(scores, key=scores.get, reverse=True)
                    [: cfg["layer_scan"]["top_n_layers"]])
    out = {"layer_scores": {str(k): v for k, v in scores.items()},
           "chosen_layers": chosen,
           "l0_per_layer": {str(k): v for k, v in l0s.items()},
           "mode": mode, "n_questions": n, "k_high": k_high}
    io_utils.write_json(out_path, out)
    io_utils.status("layer_scan", True, f"chosen layers {chosen} (mode={mode})")
    return out
