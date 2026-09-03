<p align="center">
  <img src="assets/islalogo.png" alt="Isla-SNN" height="200">
</p>

# Isla-SNN

A lightweight spiking neural network framework for language modeling.

Isla explores a different path for neural attention: instead of dot-product similarity, it measures **spike timing synchrony** - how closely neurons fire together. This gives the model a biologically-inspired inductive bias while keeping everything trainable with standard backpropagation via surrogate gradients.

## Highlights

- **Spike synchrony attention** — a novel RBF-kernel attention based on spike timing similarity
- **LIF neurons** — Leaky Integrate-and-Fire with learnable per-unit decay and vectorised multi-step
- **Stable gated residuals** — identity-preserving MLP residual with legacy checkpoint compatibility
- **Surrogate Gradient Control** — adjustable gradient steepness (slope) for stable and flexible learning
- **Rotary Position Embeddings** — RoPE applied before timing mapping for clean positional encoding
- **Simple API** — `isla.train()`, `isla.generate()`, `isla.load_model()`
- **Fast inference** — KV cache + streaming generation
- **Token packing** — zero-waste concatenate-and-chunk for efficient training

## Quick Start

### Train

```python
import isla

model_config = isla.ModelConfig(
    hidden_dim=512,
    num_layers=8,
    num_heads=8,
    num_timesteps=4,
    max_seq_len=2048,
    target_spike_rate=0.3,
)

train_config = isla.TrainConfig(
    lr=3e-4,
    batch_size=16,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
)

data_config = isla.DataConfig(
    dataset_path="./data/my_dataset_tokenized",
    pack_sequences=True,  # concatenate+chunk, no padding waste
)

model, tokenizer = isla.train(model_config, train_config, data_config)
```

### Generate

```python
model, tokenizer = isla.load_model("./outputs/checkpoints/final", device="cuda")

# full generation
print(isla.generate(model, tokenizer, "Once upon a time"))

# streaming (token by token)
print("Hello", end="")
for piece in isla.generate_stream(model, tokenizer, "Hello"):
    print(piece, end="", flush=True)
```

### Install

```bash
pip install -e .            # core only
pip install -e ".[dev]"     # with pytest, wandb, matplotlib
```

## Project Structure

```
Isla-SNN/
├── isla/                          # framework package
│   ├── __init__.py                # public API (train, generate, load_model)
│   ├── config.py                  # all config dataclasses (with validation)
│   ├── model/
│   │   ├── neurons.py             # LIF neuron + SFA + surrogate gradient
│   │   ├── attention.py           # spike sync attention + RoPE + KV cache
│   │   └── architecture.py        # IslaModel, SpikingBlock, identity-preserving gated residual
│   ├── data/
│   │   └── loader.py              # HF datasets + tokenizer + packing + caching
│   ├── training/
│   │   └── trainer.py             # training loop + best model + W&B + diagnostics
│   └── inference/
│       ├── generate.py            # generate + stream with KV cache
│       └── speed.py               # torch.compile + CUDA tuning
├── main.py                        # CLI entry point
├── chat.py                        # interactive chat REPL
├── pyproject.toml                 # packaging (pip install -e .)
├── examples/
│   ├── config_nano.json           # ~4M params (testing)
│   ├── config_50m.json            # ~50M params
│   ├── config_150m.json           # ~150M params
│   └── prepare_data.py            # multi-source dataset preparation
├── notebooks/
│   ├── 00_dataset_caching.ipynb   # pre-tokenize datasets
│   ├── 01_train.ipynb             # Colab training
│   ├── 02_inference.ipynb         # inference + spike analysis
│   └── 03_finetune_chat.ipynb     # instruction fine-tuning
└── tests/
    ├── test_model.py
    └── verify_all.py              # full verification suite
```

## How It Works

**Spike synchrony attention** replaces the standard dot-product with an RBF kernel on sigmoid-mapped projections:

```
Standard:   score(i,j) = Q_i · K_j / √d
Isla:       score(i,j) = -‖σ(Q_i) - σ(K_j)‖² / τ
```

The sigmoid `σ` maps projections to a `[0,1]` timing space. Tokens with *similar* timing profiles attend to each other, regardless of magnitude. `τ` is a learnable temperature (clamped to `[0.1, 10]` for stability).

**Rotary Position Embeddings (RoPE)** are applied to Q and K *before* the sigmoid mapping, so positional information modulates which timing region each token falls into while preserving the RBF structure.

**LIF neurons** replace GELU/ReLU in the feed-forward blocks:

```
V[t] = β · V[t-1] + I[t]         # leaky integration
S[t] = Θ(V[t] - θ)               # spike if above threshold
V[t] = V[t] · (1 - S[t])         # reset after spike
```

Each neuron has a learnable decay `β ∈ (0,1)`. The output is the mean spike rate across `T` timesteps, giving a smooth signal back to the residual stream. The `multi_step()` method computes T timesteps efficiently and returns per-unit spike rates for fine-grained diagnostics.

**Token packing** concatenates all tokenized texts and re-chunks them into fixed-length blocks. This eliminates padding waste, typically yielding 2-5× more effective training tokens per batch compared to padding-based approaches.

New models use an identity-preserving gated MLP residual, `h = h + gate · mlp(h)`, which keeps a clean gradient path through deep stacks. Checkpoints created with the earlier spike-first residual are detected while loading and retain their original behavior.

## Training Features

| Feature | Detail |
|---|---|
| Precision | bf16 / fp16 via `torch.cuda.amp` (mutually exclusive, validated) |
| VRAM | Gradient checkpointing (trade compute for memory) |
| Optimizer | AdamW with warmup + cosine decay |
| Training | Epoch-based (auto-computes steps) or step-based, with progress bar |
| Data | Token packing (zero-waste) or padding (fallback) |
| Fine-Tuning | Native instruction prompt-masking (`-100` label suppression via `response_template`) |
| Positions | Rotary Position Embeddings (RoPE) |
| Slope | Configurable Surrogate Gradient slope via `TrainConfig` |
| Logging | JSONL + W&B (optional), single-line progress bar |
| Diagnostics | τ, β per layer, spike rates ± std, dead/saturated neuron %, grad norm |
| Checkpoints | Periodic (`step_N/`), best model (`best/`), latest (`latest/`), final (`final/`) |
| Resume | Full state restore: optimizer, scaler, step, tokens_seen, best_val_loss |
| Interrupt-safe | `latest/` always saved on exit (Ctrl+C, Colab disconnect, crash) |
| Speed | torch.compile, cudnn.benchmark, matmul precision |
| Ablation | Swap to standard dot-product attention via config flag |
| Packaging | `pip install -e .` via pyproject.toml |

## Measured Findings

These come from auditing the trained 134M checkpoint (2B PT-BR tokens) against
`nicholasKluge/TeenyTinyLlama-160m`, which shares the same tokenizer, on 295
held-out Portuguese Wikipedia articles. Scripts are in `experiments/`.

### Energy: the FFN is not event-driven

`SpikingMLP` computes `down(rate + alpha * membrane)`. The `rate` half is
event-driven, but `membrane` is continuous, so by linearity the second half is a
full dense matmul over every unit. A small `alpha` does not avoid it. Counting
only the accumulates — as the profiler originally did — overstates the saving:

| accounting | pJ/token | vs dense |
|---|---|---|
| dense FFN baseline | 6,794,772 | — |
| FFN, accumulates only (original claim) | 3,558,026 | +47.6% |
| FFN, membrane term included | 6,955,412 | **−2.4%** |

Removing `alpha` is not a free fix. Sweeping it on the trained model:

| alpha scale | ×1.5 | ×1.0 | ×0.75 | ×0.5 | ×0.25 | ×0 |
|---|---|---|---|---|---|---|
| perplexity | 365 | **304** | 337 | 715 | 2,764 | 7,276 |

The trained value is a genuine optimum, and losing membrane is 6.7× more
damaging than having too much of it and the degradation is smooth rather than a
cliff, so this is information loss, not a calibration shock. The continuous term
is load-bearing.

Compensating with more timesteps does not work either, because spike events
scale with `T`:

| T | levels | bits | energy vs dense |
|---|---|---|---|
| 4 | 5 | 2.3 | 0.19× |
| 8 | 9 | 3.2 | 0.38× |
| 16 | 17 | 4.1 | 0.76× |
| 21 | 22 | 4.5 | **1.00×** |

Event-driven stops beating dense at `T ≈ 21`, where rate coding still carries
only ~4.5 bits. Rate coding is efficient only when the task tolerates low
precision, and language modeling does not.

**Whole-model ceiling, if `alpha` could be removed: ~25%, not 91%.** The FFN is
52% of per-token MACs; attention projections and the LM head are the rest.

### Capability: a 1.76× gap not explained by data

| | bits/char | perplexity |
|---|---|---|
| Isla-134M (2B tokens) | 1.7488 | 304.0 |
| TeenyTinyLlama-160m (~6.2B tokens) | 0.8898 | 18.3 |

Use bits/char — token perplexity is only comparable because the tokenizers are
identical here. Scaling laws attribute roughly 15% of the gap to TTL's 3.1×
larger token budget and 19% more parameters, leaving ~1.76× unexplained.

### Likely cause: the residual gate starves the MLP

`SpikingBlock` computes `h = h + gate * mlp_out`, with `gate_raw` initialised at
−2.0. After 2B tokens the gate had moved from 0.119 to only 0.131. Because the
gate also scales the gradient reaching the MLP, the branch trains ~7.7× slower
than an ungated residual, never becomes useful, and the gate never opens.

Measuring `‖mlp‖ / ‖h‖` per layer, averaged over layers 1-11 (layer 0 is an
outlier in both models):

| | after gate | gate divided out | TTL-160m |
|---|---|---|---|
| mean contribution | 0.1412 | **1.0748** | 0.4445 |

The apparent 3.1× deficit inverts once the gate is removed: the spiking MLP
emits 2.4× *more* signal than the baseline's, and the gate discards it. The
deficit survives only in the last third of the stack (layers 9-11), where
contribution falls to 0.19× of the first third while TTL's *rises* to 1.37×.

This is evidence, not proof that large output norm is not the same as useful
output, and a model trained at gate 0.13 is optimal at 0.13 by construction, so
sweeping a trained checkpoint cannot settle it. `experiments/nano_grid.py` runs
the controlled comparison at nano scale.

## Known Limitations

| Limitation | Detail |
|---|---|
| **Energy claims need care** | The FFN is not event-driven while `alpha > 0`; see Measured Findings. `profile_efficiency.py` reports FFN only, on modelled ASIC costs, not measured hardware. |
| **Capability gap** | ~1.76× worse bits/char than a size-matched Llama, cause not yet isolated. |
| **O(L²) attention** | Spike sync attention uses custom RBF kernel — not compatible with FlashAttention or xformers. Sequence lengths above 2048 will be slow. |
| **Sequential LIF timesteps** | Each spiking MLP runs T timesteps sequentially (T=4 by default), making FFN blocks ~4× slower than standard GELU/SiLU. On GPU the model is slower than a dense transformer of the same size. |
| **Single GPU** | No distributed training (DDP/FSDP) support yet. |

## Requirements

```text
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0
```

**For Analytics & Evaluation:**
```bash
pip install -q pandas matplotlib wandb
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

## References

- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). **Surrogate Gradient Learning in Spiking Neural Networks**: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks. *IEEE Signal Processing Magazine*.
- Zenke, F., & Ganguli, S. (2018). **SuperSpike**: Supervised Learning in Multilayer Spiking Neural Networks. *Neural Computation*.
- Zhu, R.-J., Zhao, Q., Li, H., & Wu, P. (2023). **SpikeGPT**: Generative Pre-trained Language Model with Spiking Neural Networks. *arXiv*.

## License

MIT — The Isla-SNN framework code is released under the MIT License. Trained model weights are released under the same license. Training data retains its original licensing (see acknowledgements above).
