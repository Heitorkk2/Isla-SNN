"""Full end-to-end verification of the Isla-SNN framework."""
import torch
import tempfile
from isla.model import IslaModel, KVCache
from isla.model.neurons import LIFNeuron, spike_fn
from isla.model.attention import SpikeSyncAttention, StandardAttention, RotaryEmbedding
from isla.config import ModelConfig, TrainConfig, CheckpointConfig, WandbConfig
from isla.training.trainer import IslaTrainer
from isla.inference.generate import generate_stream
from isla.inference.speed import setup_speed

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# small config for all tests
cfg = ModelConfig(vocab_size=256, hidden_dim=64, num_layers=2, num_heads=2,
                  num_timesteps=2, max_seq_len=32, dropout=0.0,
                  target_spike_rate=0.3, spike_reg_lambda=0.01)

print("=== Neurons ===")
out = spike_fn(torch.randn(10), threshold=0.0)
check("spikes are binary", set(out.unique().tolist()).issubset({0.0, 1.0}))

v = torch.randn(10, requires_grad=True)
spike_fn(v, threshold=0.0).sum().backward()
check("surrogate gradient flows", v.grad is not None and (v.grad != 0).any())

lif = LIFNeuron(32)
s, m = lif.step(torch.randn(4, 32))
check("LIF shapes", s.shape == (4, 32) and m.shape == (4, 32))
check("beta in (0,1)", (lif.beta > 0).all() and (lif.beta < 1).all())

# multi_step vectorised
spike_sum, mem, rate_per_unit = lif.multi_step(torch.randn(4, 16, 32), T=4)
check("multi_step spike_sum shape", spike_sum.shape == (4, 16, 32))
check("multi_step rate_per_unit shape", rate_per_unit.shape == (32,))
check("multi_step rates in [0,1]", (rate_per_unit >= 0).all() and (rate_per_unit <= 1).all())

print("\n=== RoPE ===")
rope = RotaryEmbedding(32, max_seq_len=64)
cos, sin = rope(seq_len=16, offset=0)
check("RoPE shape", cos.shape == (1, 1, 16, 32))
cos_full, _ = rope(seq_len=32, offset=0)
cos_off, _ = rope(seq_len=16, offset=16)
check("RoPE offset consistency", torch.allclose(cos_full[:, :, 16:, :], cos_off, atol=1e-6))

print("\n=== Attention ===")
attn = SpikeSyncAttention(64, 2)
out, _ = attn(torch.randn(2, 8, 64))
check("sync attention shape", out.shape == (2, 8, 64))
check("tau clamped", 0.1 <= attn.tau.item() <= 10.0)

std = StandardAttention(64, 2)
out2, _ = std(torch.randn(2, 8, 64))
check("standard attention shape", out2.shape == (2, 8, 64))

print("\n=== Model ===")
model = IslaModel(cfg)
ids = torch.randint(0, 256, (2, 16))
logits, metrics, caches = model(ids)
check("forward shape", logits.shape == (2, 16, 256))
check("caches None (no cache)", caches is None)
check("spike rate in [0,1]", 0 <= metrics["mean_spike_rate"].item() <= 1)
check("spike_rate_std present", "spike_rate_std" in metrics)
check("no pos_emb (uses RoPE)", not hasattr(model, "pos_emb"))

logits.sum().backward()
check("backward works", all(p.grad is not None for p in model.parameters() if p.requires_grad))

check("weight tying", model.lm_head.weight is model.token_emb.weight)

print("\n=== Gradient Checkpointing ===")
model2 = IslaModel(cfg)
model2.enable_gradient_checkpointing()
model2.train()
logits2, _, _ = model2(ids)
logits2.sum().backward()
check("grad ckpt forward+backward OK", True)

print("\n=== KV Cache ===")
model.eval()
caches = [KVCache() for _ in range(cfg.num_layers)]
logits_full, _, _ = model(ids[:1])
logits_cached, _, caches = model(ids[:1], caches=caches)
diff = (logits_full - logits_cached).abs().max().item()
check(f"prefill matches full (diff={diff:.2e})", diff < 1e-4)

new_tok = torch.randint(0, 256, (1, 1))
logits_step, _, caches = model(new_tok, caches=caches)
all_ids = torch.cat([ids[:1], new_tok], dim=1)
logits_full2, _, _ = model(all_ids)
diff2 = (logits_full2[:, -1] - logits_step[:, -1]).abs().max().item()
check(f"cached step matches full (diff={diff2:.2e})", diff2 < 1e-3)

print("\n=== Save/Load ===")
with torch.no_grad():
    out_before, _, _ = model(ids[:1])
with tempfile.TemporaryDirectory() as d:
    model.save_pretrained(d)
    model_loaded = IslaModel.from_pretrained(d)
    with torch.no_grad():
        out_after, _, _ = model_loaded(ids[:1])
check("save/load roundtrip", torch.allclose(out_before, out_after, atol=1e-5))

print("\n=== Ablation ===")
cfg_abl = ModelConfig(vocab_size=256, hidden_dim=64, num_layers=2, num_heads=2,
                      num_timesteps=2, max_seq_len=32, use_standard_attention=True)
m_abl = IslaModel(cfg_abl)
check("ablation uses StandardAttention", isinstance(m_abl.blocks[0].attn, StandardAttention))
logits_abl, _, _ = m_abl(ids)
check("ablation forward OK", logits_abl.shape == (2, 16, 256))

print("\n=== Overfit ===")
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()
data = torch.randint(0, 256, (8, 16))
losses = []
for _ in range(50):
    out, _, _ = model(data)
    loss = crit(out[:, :-1].contiguous().view(-1, 256), data[:, 1:].contiguous().view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
first5 = sum(losses[:5]) / 5
last5 = sum(losses[-5:]) / 5
check(f"overfit test ({first5:.2f} -> {last5:.2f})", last5 < first5 * 0.8)

print("\n=== Config ===")
check("target_spike_rate", cfg.target_spike_rate == 0.3)
check("WandbConfig", WandbConfig().enabled == False)
check("CheckpointConfig.resume_from", CheckpointConfig().resume_from == "")
tc = TrainConfig(gradient_checkpointing=True)
check("gradient_checkpointing flag", tc.gradient_checkpointing is True)

# bf16/fp16 mutual exclusion
try:
    TrainConfig(bf16=True, fp16=True)
    check("bf16+fp16 rejected", False)
except ValueError:
    check("bf16+fp16 rejected", True)

print("\n=== Speed ===")
m_speed = setup_speed(compile_model=False, model=model)
check("setup_speed returns model", m_speed is model)

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL CHECKS PASSED!")
else:
    print("SOME CHECKS FAILED!")
