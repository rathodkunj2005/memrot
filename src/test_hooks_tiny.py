"""Hook-mechanics test on a tiny randomly-initialized Gemma-2 (no gated weights).

Verifies on whatever CUDA device is present (fp32, so even a login-node GT 1030
works): Instrumentor's in-hook reduction + weights-freeing, residual capture,
greedy_generate, HeadAblator calibrate/ablate, and HeadAmplifier. Run:

    python src/test_hooks_tiny.py
"""
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

import io_utils
import paths
from ablation import HeadAblator
from instrument import Instrumentor, residuals_all_layers
from intervention import HeadAmplifier

cfg = io_utils.load_config(paths.config_path())
torch.manual_seed(0)
config = Gemma2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, sliding_window=64,
                      attn_implementation="eager")
model = Gemma2ForCausalLM(config).to("cuda").eval()

prompt = list(range(3, 103))                      # 100 fake tokens
gen = [5, 6, 7]
gold = (10, 30)

instr = Instrumentor(model, sae_layer_ids=[1, 3])
mass, resid = instr.run(prompt, gen, gold)
assert mass.shape == (4, 4) and (mass >= 0).all() and (mass <= 1).all()
full, _ = instr.run(prompt, gen, (0, len(prompt) + len(gen)))
assert (abs(full - 1.0) < 1e-3).all(), f"full-range mass != 1: {full}"
assert set(resid) == {1, 3} and resid[1].shape == (64,)

r_all = residuals_all_layers(model, prompt)
assert set(r_all) == {0, 1, 2, 3}
assert torch.allclose(r_all[1], resid[1], atol=1e-4), "hook vs hidden_states mismatch"

with torch.no_grad():
    base = model(torch.tensor([prompt], device="cuda")).logits[0, -1].clone()

ab = HeadAblator(model, [(1, 0), (2, 3)], mode="mean")
ab.calibrate_hooks()
with torch.no_grad():
    model(torch.tensor([prompt], device="cuda"), use_cache=False)
ab.finish_calibration()
assert set(ab.means) == {(1, 0), (2, 3)} and ab.means[(1, 0)].shape == (16,)
ab.ablate_hooks()
with torch.no_grad():
    ablated = model(torch.tensor([prompt], device="cuda")).logits[0, -1]
ab.remove()
assert not torch.allclose(base, ablated, atol=1e-5), "ablation had no effect"
with torch.no_grad():
    restored = model(torch.tensor([prompt], device="cuda")).logits[0, -1]
assert torch.allclose(base, restored), "hooks not cleanly removed"

amp = HeadAmplifier(model, [(0, 1)], alpha=3.0)
amp.attach()
with torch.no_grad():
    amped = model(torch.tensor([prompt], device="cuda")).logits[0, -1]
amp.remove()
assert not torch.allclose(base, amped, atol=1e-5), "amplification had no effect"

out = model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=4,
                     do_sample=False, pad_token_id=0)
assert out.shape[1] == len(prompt) + 4

io_utils.status("test_hooks_tiny", True,
                "instrument + ablation + amplifier + generate verified on tiny Gemma2")
