"""Hooks for the instrumented forward pass.

Memory contract: full attention matrices are NEVER retained. We run the forward
with output_attentions=True (eager) and register forward hooks on every
self_attn module that (a) reduce the [B,H,Tq,Tk] weights to per-head attention
mass on the gold span, averaged over the answer-emission query rows, and
(b) return the output tuple with the weights replaced by None so the only full
matrix alive at any moment is the current layer's (transient ~1.9 GB fp32 at
T=7.5k — the stated eager peak).

Residual capture: forward hooks on selected decoder layers store the resid_post
vector at the last prompt position (the position that emits the first answer
token); SAE encoding happens after the pass.

The instrumented pass runs over [prompt + generated answer] tokens, so
"answer-emission rows" = positions [prompt_len-1 .. T-1) — the queries that
produced each answer token.
"""
import torch

from io_utils import die


class Instrumentor:
    def __init__(self, model, sae_layer_ids):
        self.model = model
        self.sae_layer_ids = list(sae_layer_ids)
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

    @torch.no_grad()
    def run(self, prompt_ids, gen_ids, gold_span):
        """Returns (attn_mass [n_layers, n_heads] float32 numpy,
                    {layer: resid_vec tensor [d_model]})."""
        if not gen_ids:
            die("instrument", "empty gen_ids: no answer-emission rows to reduce over")
        full = list(prompt_ids) + list(gen_ids)
        q_start = len(prompt_ids) - 1            # first answer-emission query row
        gs, ge = gold_span
        attn_mass = torch.zeros(self.n_layers, self.n_heads)
        resid = {}
        handles = []

        def attn_hook(layer_idx):
            def hook(module, args, output):
                if not isinstance(output, tuple) or len(output) < 2:
                    die("instrument", f"unexpected attn output type at layer {layer_idx}; "
                                      "transformers version drift — repin per env/environment.yml")
                w = output[1]
                if w is None or w.ndim != 4:
                    die("instrument", f"attention weights not exposed at layer {layer_idx}; "
                                      "is attn_implementation really 'eager'?")
                # w: [1, H, Tq, Tk] post-softmax. Emission rows are q_start..T-2:
                # row p predicts token p+1, and row T-1's prediction is unused,
                # so including it averages in a non-emission row (and broke the
                # smoke full-range==1 check, since its self-key lies past ge).
                attn_mass[layer_idx] = (
                    w[0, :, q_start:-1, gs:ge].sum(dim=-1).mean(dim=-1).float().cpu())
                return (output[0], None, *output[2:])   # free the full matrix
            return hook

        def resid_hook(layer_idx):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                resid[layer_idx] = h[0, q_start, :].detach().float().clone()
            return hook

        for i, layer in enumerate(self.model.model.layers):
            handles.append(layer.self_attn.register_forward_hook(attn_hook(i)))
            if i in self.sae_layer_ids:
                handles.append(layer.register_forward_hook(resid_hook(i)))
        try:
            ids = torch.tensor([full], device="cuda")
            self.model(ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if torch.isnan(attn_mass).any():
            die("instrument", "NaN in reduced attention mass")
        return attn_mass.numpy(), resid


@torch.no_grad()
def residuals_all_layers(model, prompt_ids):
    """Cheap path for the layer scan: resid_post at the last prompt position for
    EVERY layer via output_hidden_states (hidden_states[L+1] = resid_post of L)."""
    ids = torch.tensor([prompt_ids], device="cuda")
    out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # tuple len n_layers+1
    return {L: hs[L + 1][0, -1, :].detach().float().clone()
            for L in range(model.config.num_hidden_layers)}
