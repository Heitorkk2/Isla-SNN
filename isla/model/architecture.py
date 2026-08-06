"""Isla-SNN language model.

Token Embed
→ N × SpikingBlock (sync attention w/ RoPE + spiking MLP)
→ RMSNorm → tied linear head

Each SpikingBlock is a pre-norm residual block:
    h → Norm → SpikeSyncAttention → +h
    h → Norm → SpikingMLP(LIF)   → +h

The output head reads continuous membrane potentials (not binary
spikes), which gives a smooth distribution over the vocabulary.

Position encoding uses Rotary Position Embeddings (RoPE) applied
inside the attention heads, not absolute learned embeddings.
"""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

from isla.config import ModelConfig
from .neurons import LIFNeuron
from .attention import SpikeSyncAttention, SpikeStateSpaceAttention, StandardAttention, KVCache, SSMCache


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SpikingMLP(nn.Module):
    """Feed-forward block whose nonlinearity is a bank of LIF neurons.

    v3: Uses both binary spike AND continuous membrane potential as features.
    x → up-project → LIF(T steps) → spike + α·membrane → down-project

    The α parameter is learnable, letting the model decide how much
    continuous voltage information to mix with binary spikes.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_ff = config.hidden_dim * config.ff_mult
        self.up = nn.Linear(config.hidden_dim, d_ff, bias=False)
        self.down = nn.Linear(d_ff, config.hidden_dim, bias=False)
        self.lif = LIFNeuron(d_ff, config.beta_init, config.threshold, config.surrogate_slope)
        self.T = config.num_timesteps
        self.dropout = nn.Dropout(config.dropout)
        # learnable membrane mixing coefficient
        # sigmoid keeps it in (0,1); init at -2 → sigmoid(-2) ≈ 0.12 (subtle start)
        self.alpha_raw = nn.Parameter(torch.full((1,), -2.0))
        # R-STDP: set to True by trainer to enable spike trace storage
        self._track_traces = False
        self._stdp_decay_pre = 0.95   # overridden by trainer when R-STDP active
        self._stdp_decay_post = 0.95  # overridden by trainer when R-STDP active

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x):
        h = self.up(x)
        if self._track_traces:
            spike_sum, membrane, rate_per_unit, stdp_data = self.lif.multi_step(
                h, self.T, track_traces=True,
                stdp_decay_pre=self._stdp_decay_pre,
                stdp_decay_post=self._stdp_decay_post,
            )
            self._stdp_pre = x.detach()            # (B, L, D_in)
            self._stdp_ltp_accum = stdp_data['ltp_accum']  # (B, L, D_ff) — compact!
            self._stdp_ltd_accum = stdp_data['ltd_accum']  # (B, L, D_ff)
        else:
            spike_sum, membrane, rate_per_unit = self.lif.multi_step(h, self.T)
        rate = spike_sum / self.T
        # mix spike (binary) + alpha * membrane (continuous)
        mixed = rate + self.alpha * membrane.clamp(-1, 1)
        self._last_rates = rate_per_unit.detach()  # for diagnostics
        return self.dropout(self.down(mixed)), rate_per_unit


class SpikingBlock(nn.Module):
    """Pre-norm residual block with spike-synchrony attention and spiking MLP.

    New models preserve the identity path and gate the transformed branch:
        h = h_pre + gate * spike_out

    Legacy checkpoints keep the original spike-first topology through
    config.mlp_residual_mode="spike_first".
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        if config.use_standard_attention:
            attn_cls = StandardAttention
        elif config.use_spike_ssm:
            attn_cls = SpikeStateSpaceAttention
        else:
            attn_cls = SpikeSyncAttention
            
        self.attn = attn_cls(
            config.hidden_dim, config.num_heads,
            config.dropout, config.sync_tau_init,
            config.max_seq_len,
        )
        self.mlp_norm = RMSNorm(config.hidden_dim)
        self.mlp = SpikingMLP(config)
        self.mlp_residual_mode = config.mlp_residual_mode
        # sigmoid(-2) ≈ 0.12: introduce the transformed branch gradually
        self.gate_raw = nn.Parameter(torch.full((1,), -2.0))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_raw)

    def forward(self, h, mask=None, cache=None):
        attn_out, cache = self.attn(self.attn_norm(h), mask, cache=cache)
        h = h + attn_out  # attention residual: kept
        h_pre = h
        mlp_out, spike_rate = self.mlp(self.mlp_norm(h))
        if self.mlp_residual_mode == "identity":
            h = h_pre + self.gate * mlp_out
        else:
            # Exact v3 behaviour for backward-compatible checkpoint loading.
            h = mlp_out + self.gate * h_pre
        return h, spike_rate, cache


class IslaModel(nn.Module):
    """Isla-SNN causal language model.

    Typical usage:
        model = IslaModel(config)             # from config
        model = IslaModel.from_pretrained(d)  # from checkpoint dir
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._grad_ckpt = False
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([SpikingBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)

        # tied output head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, caches: Optional[List[KVCache]] = None):
        """
        Args:
            input_ids: (B, L)
            caches:    list of KVCache (one per layer), or None for training
        Returns:
            logits:  (B, L, V)
            metrics: dict with mean_spike_rate, spike_rate_std, and per-layer rates
            caches:  updated caches list (None during training)
        """
        B, L = input_ids.shape
        use_cache = caches is not None

        h = self.emb_drop(self.token_emb(input_ids))

        # SSM causality is implicit in its recurrence. Avoiding a dense mask is
        # required for its memory usage to remain linear in sequence length.
        if self.config.use_spike_ssm:
            mask = None
        else:
            # Generated on-the-fly (no persistent O(L²) buffer).
            pos_offset = caches[0].seq_len if use_cache else 0
            L_total = pos_offset + L
            mask = torch.triu(
                torch.full((L, L_total), float("-inf"), device=input_ids.device,
                           dtype=h.dtype),
                diagonal=1 + pos_offset,
            )

        spike_rates = []
        for i, block in enumerate(self.blocks):
            layer_cache = caches[i] if use_cache else None
            if self._grad_ckpt and self.training:
                h, rate, _ = ckpt_fn(block, h, mask, use_reentrant=False)
            else:
                h, rate, layer_cache = block(h, mask, cache=layer_cache)
                if use_cache:
                    caches[i] = layer_cache
            spike_rates.append(rate)

        logits = self.lm_head(self.final_norm(h))

        # per-unit rates stacked across layers → richer diagnostics
        stacked = torch.stack(spike_rates)  # (num_layers, d_ff)
        return logits, {
            "mean_spike_rate": stacked.mean(),
            "spike_rate_std": stacked.std(),
            "spike_rates_per_layer": spike_rates,
        }, caches if use_cache else None

    def enable_gradient_checkpointing(self):
        """Trade extra compute for lower VRAM by not storing block activations."""
        self._grad_ckpt = True

    def count_params(self):
        """Trainable parameters (tied weights counted once)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    def save_pretrained(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "model_config.json")
        torch.save(self.state_dict(), d / "model.pth")

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        d = Path(directory)
        config = ModelConfig.load(d / "model_config.json")
        model = cls(config)
        state = torch.load(d / "model.pth", map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)
