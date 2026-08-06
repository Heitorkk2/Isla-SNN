"""Spike synchrony attention with RoPE and optional KV cache.

Standard:   score(i,j) = Qᵢ · Kⱼ / √d
Sync:       score(i,j) = -‖σ(Qᵢ) - σ(Kⱼ)‖² / τ

The sigmoid σ maps projections to a [0,1] pseudo-timing space, and τ
is a learnable temperature. This is a negative squared-distance (RBF)
kernel: tokens with *similar* timing profiles attend strongly.

Efficient form:
    score = (2·q·kᵀ - ‖q‖² - ‖k‖²) / τ

KV cache stores the timing-mapped K (kt = σ(K)) and V across calls,
so during generation only the new token's projections are computed.

Rotary Position Embeddings (RoPE) are applied to Q and K *before* the
sigmoid mapping in sync attention, and directly in standard attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Rotary Position Embeddings ---------- #

class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE (Su et al., 2021).

    Supports positional offset for KV-cache inference.
    """

    def __init__(self, head_dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # pre-build for common lengths; extended lazily if needed
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, d)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len, offset=0):
        """Return (cos, sin) each of shape (1, 1, L, d) for positions [offset, offset+L)."""
        end = offset + seq_len
        if end > self._cached_len:
            self._build_cache(end * 2)  # double to amortise
        cos = self.cos_cached[offset:end].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:end].unsqueeze(0).unsqueeze(0)
        return cos, sin


def _rotate_half(x):
    """Split x into two halves and rotate: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply RoPE to tensor x of shape (B, H, L, d)."""
    return x * cos + _rotate_half(x) * sin


# ---------- KV Cache ---------- #

class KVCache:
    """Stores cached kt and v tensors for incremental decoding.

    Each layer gets its own KVCache instance. On each step the new
    token's kt and v are appended, and the full cached tensors are
    returned for the attention computation.
    """

    def __init__(self):
        self.kt: Optional[torch.Tensor] = None  # (B, H, L_cached, d)
        self.v: Optional[torch.Tensor] = None

    @property
    def seq_len(self):
        return 0 if self.kt is None else self.kt.shape[2]

    def update(self, new_kt, new_v):
        """Append new keys/values and return full cache."""
        if self.kt is None:
            self.kt, self.v = new_kt, new_v
        else:
            self.kt = torch.cat([self.kt, new_kt], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.kt, self.v


# ---------- Spike Synchrony Attention~ ---------- #

class SpikeSyncAttention(nn.Module):
    """Multi-head spike-synchrony attention with RoPE and optional KV cache.

    RoPE is applied to Q and K *before* the sigmoid timing mapping, so
    positional information modulates which timing region each token falls
    into while preserving the [0,1] RBF structure.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, tau_init=1.0, max_seq_len=4096):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        self.tau_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(tau_init) - 1.0))
        )
        self.attn_drop = nn.Dropout(dropout)

    @property
    def tau(self):
        return F.softplus(self.tau_raw).clamp(0.1, 10.0)

    def forward(self, x, mask=None, cache: Optional[KVCache] = None):
        """
        Args:
            x:     (B, L, D) — full sequence or single new token
            mask:  (L_q, L_kv) additive mask
            cache: KVCache instance for incremental decoding (None during training)
        Returns:
            output: (B, L, D)
            cache:  updated KVCache (or None if not caching)
        """
        B, L, _ = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # RoPE: apply before sigmoid so position modulates timing
        pos_offset = cache.seq_len if cache is not None else 0
        cos, sin = self.rotary(L, offset=pos_offset)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # sigmoid timing mapping
        qt = torch.sigmoid(q)
        kt = torch.sigmoid(k)

        if cache is not None:
            kt, v = cache.update(kt, v)

        # RBF kernel: score = (2·qt·ktᵀ - ‖qt‖² - ‖kt‖²) / τ
        qk = torch.matmul(qt, kt.transpose(-2, -1))
        q_sq = (qt * qt).sum(-1, keepdim=True)
        k_sq = (kt * kt).sum(-1, keepdim=True)
        scores = (2.0 * qk - q_sq - k_sq.transpose(-2, -1)) / self.tau

        if mask is not None:
            scores = scores + mask

        weights = self.attn_drop(F.softmax(scores, dim=-1))

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), cache


# ---------- Standard Attention (ablation baseline) ---------- #

class StandardAttention(nn.Module):
    """Standard scaled dot-product attention with RoPE (ablation baseline).

    Same interface as SpikeSyncAttention so models can swap between them
    via config.use_standard_attention without any other code changes.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1, tau_init=1.0, max_seq_len=4096):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, cache: Optional[KVCache] = None):
        B, L, _ = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # RoPE
        pos_offset = cache.seq_len if cache is not None else 0
        cos, sin = self.rotary(L, offset=pos_offset)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        # Dispatches to FlashAttention or another fused SDPA kernel when the
        # current device, dtype, and mask support it.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), cache


# ---------- SSM Cache ---------- #

class SSMCache:
    """Recurrent state cache for SpikeStateSpaceAttention.

    Instead of storing full K/V history (O(L) memory), stores a
    compressed (d, d) state matrix per head — O(1) memory per layer.
    """

    def __init__(self):
        self.state: Optional[torch.Tensor] = None   # (B, H, d, d)
        self.norm: Optional[torch.Tensor] = None     # (B, H, d, 1)
        self._seq_len: int = 0

    @property
    def seq_len(self):
        return self._seq_len

    def update(self, state_delta, norm_delta, decay):
        """Accumulate new key-value into the recurrent state."""
        if self.state is None:
            self.state = state_delta
            self.norm = norm_delta
        else:
            self.state = decay * self.state + state_delta
            self.norm = decay * self.norm + norm_delta
        self._seq_len += 1
        return self.state, self.norm


# ---------- Spike State Space Attention ---------- #

class SpikeStateSpaceAttention(nn.Module):
    """O(N) spike synchrony via recurrent state accumulation.

    Replaces the O(N²) RBF kernel of SpikeSyncAttention with a
    linear-time recurrent formulation that preserves spike timing
    semantics:

        state[t] = λ · state[t-1] + σ(K[t])ᵀ ⊗ V[t]
        output[t] = σ(Q[t]) · state[t] / normalizer[t]

    The sigmoid σ maps to [0,1] timing space (same as SpikeSyncAttention).
    The decay λ acts as membrane leak — recent spikes matter more.

    Training:  sequential scan over L positions (O(L·d²) per head)
    Inference: true recurrent with SSMCache (O(d²) per step)

    Same interface as SpikeSyncAttention for drop-in replacement.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1,
                 tau_init=1.0, max_seq_len=4096):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        # learnable decay per head: sigmoid keeps it in (0, 1)
        # init near 0.95 for moderate memory retention
        raw_init = torch.log(torch.tensor(0.95 / (1.0 - 0.95)))
        self.decay_raw = nn.Parameter(
            torch.full((num_heads, 1, 1), raw_init.item())
        )

        self.attn_drop = nn.Dropout(dropout)

    @property
    def decay(self):
        """Per-head decay λ ∈ (0, 1)."""
        return torch.sigmoid(self.decay_raw)

    def forward(self, x, mask=None, cache: Optional[SSMCache] = None):
        """
        Args:
            x:     (B, L, D) — full sequence or single new token
            mask:  ignored (causality is implicit in the recurrence)
            cache: SSMCache for incremental decoding (None during training)
        Returns:
            output: (B, L, D)
            cache:  updated SSMCache (or None)
        """
        B, L, _ = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # RoPE before sigmoid (same as SpikeSyncAttention)
        pos_offset = cache.seq_len if cache is not None else 0
        cos, sin = self.rotary(L, offset=pos_offset)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # sigmoid timing mapping
        qt = torch.sigmoid(q)  # (B, H, L, d)
        kt = torch.sigmoid(k)

        decay = self.decay  # (H, 1, 1)

        if cache is not None:
            if L > 1:
                # Prefill: sequential scan over L tokens, then update cache
                state = cache.state if cache.state is not None else torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
                norm_state = cache.norm if cache.norm is not None else torch.zeros(B, H, d, 1, device=x.device, dtype=x.dtype)
                outputs = []

                for t in range(L):
                    k_t = kt[:, :, t:t+1, :]              # (B, H, 1, d)
                    v_t = v[:, :, t:t+1, :]                # (B, H, 1, d)
                    k_col = k_t.transpose(-1, -2)          # (B, H, d, 1)

                    state = decay * state + k_col @ v_t    # (B, H, d, d)
                    norm_state = decay * norm_state + k_col # (B, H, d, 1)

                    q_t = qt[:, :, t:t+1, :]               # (B, H, 1, d)
                    o_t = q_t @ state                       # (B, H, 1, d)
                    n_t = q_t @ norm_state + 1e-6           # (B, H, 1, 1)
                    outputs.append(o_t / n_t)

                cache.state = state
                cache.norm = norm_state
                cache._seq_len += L

                out = self.attn_drop(torch.cat(outputs, dim=2))  # (B, H, L, d)
            else:
                # Inference: single-token recurrent update (L == 1)
                k_col = kt.transpose(-1, -2)       # (B, H, d, 1)
                kv = k_col @ v                     # (B, H, d, d)
                state, norm = cache.update(kv, k_col, decay)

                out = qt @ state                    # (B, H, 1, d)
                denom = qt @ norm + 1e-6            # (B, H, 1, 1)
                out = self.attn_drop(out / denom)
        else:
            # Training: sequential scan (O(L·d²) per head)
            state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
            norm_state = torch.zeros(B, H, d, 1, device=x.device, dtype=x.dtype)
            outputs = []

            for t in range(L):
                k_t = kt[:, :, t:t+1, :]              # (B, H, 1, d)
                v_t = v[:, :, t:t+1, :]                # (B, H, 1, d)
                k_col = k_t.transpose(-1, -2)          # (B, H, d, 1)

                state = decay * state + k_col @ v_t    # (B, H, d, d)
                norm_state = decay * norm_state + k_col # (B, H, d, 1)

                q_t = qt[:, :, t:t+1, :]               # (B, H, 1, d)
                o_t = q_t @ state                       # (B, H, 1, d)
                n_t = q_t @ norm_state + 1e-6           # (B, H, 1, 1)
                outputs.append(o_t / n_t)

            out = self.attn_drop(torch.cat(outputs, dim=2))  # (B, H, L, d)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out), cache

