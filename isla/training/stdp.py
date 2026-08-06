"""Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP).

Applies a biologically-inspired local learning rule to spiking MLP
synapses (up_proj weights). The rest of the network trains via
standard backpropagation — this is a hybrid approach.

STDP rule (per timestep t):
    pre_trace[t]  = decay_pre  * pre_trace[t-1]  + input_current
    post_trace[t] = decay_post * post_trace[t-1] + post_spike[t]

    ΔW += A₊ · post_spike[t] ⊗ pre_trace[t]   (LTP)
    ΔW -= A₋ · input       ⊗ post_trace[t]    (LTD)

Weight update (after all T timesteps):
    W_up += η · R · ΔW
    where R = -loss (reward signal from cross-entropy)

v3 optimisation: STDP accumulation is performed ONLINE inside the
LIF integration loop (see neurons.py). The SpikingMLP stores only
two compact accumulators (B, L, D_ff) instead of the full spike
history tensor (T, B, L, D_ff), cutting memory by ~50%.

This module just reads the pre-accumulated sums and performs the
final two matmuls per layer — no temporal loop needed.

References:
    Florian, R. V. (2007). Reinforcement Learning Through Modulation
    of Spike-Timing-Dependent Synaptic Plasticity.
    Tavanaei et al. (2019). BP-STDP: Approximating Backpropagation
    using STDP for Training Spiking Neural Networks.
"""

import torch


class RSTDPRule:
    """Computes and applies R-STDP weight updates to spiking MLP layers.

    Usage (inside trainer):
        stdp = RSTDPRule(cfg)
        # ... forward + backward pass ...
        reward = -loss.item()
        stdp.apply_to_model(model, reward)
    """

    def __init__(self, lr=1e-3, tau_plus=20.0, tau_minus=20.0,
                 a_plus=0.01, a_minus=0.0105):
        self.lr = lr
        self.decay_pre = 1.0 - 1.0 / tau_plus     # exp(-1/τ₊)
        self.decay_post = 1.0 - 1.0 / tau_minus    # exp(-1/τ₋)
        self.a_plus = a_plus
        self.a_minus = a_minus

    @torch.no_grad()
    def _compute_update(self, pre_input, ltp_accum, ltd_accum):
        """Compute STDP weight delta from pre-accumulated online traces.

        The temporal accumulation was already done inside the LIF
        integration loop (see neurons.py multi_step), so this method
        just performs TWO matrix multiplies — no temporal loop.

        Args:
            pre_input: (B, L, D_in)  — input to up_proj
            ltp_accum: (B, L, D_ff) — Σₜ geo_ltp[t] · spike[t]
            ltd_accum: (B, L, D_ff) — Σₜ geo_ltd[t] · spike[t]

        Returns:
            delta_w: (D_ff, D_in) — weight update for up_proj
        """
        # flatten batch and sequence dims
        pre = pre_input.reshape(-1, pre_input.shape[-1])      # (N, D_in)
        ltp = ltp_accum.reshape(-1, ltp_accum.shape[-1])      # (N, D_ff)
        ltd = ltd_accum.reshape(-1, ltd_accum.shape[-1])      # (N, D_ff)
        N = pre.shape[0]

        # LTP: post fired (weighted), pre was active → strengthen
        delta_ltp = self.a_plus * (ltp.T @ pre) / N           # (D_ff, D_in)

        # LTD: pre active, post fired recently (weighted) → weaken
        delta_ltd = self.a_minus * (ltd.T @ pre) / N          # (D_ff, D_in)

        return delta_ltp - delta_ltd

    @torch.no_grad()
    def apply_to_model(self, model, reward):
        """Walk all SpikingMLP layers and apply R-STDP updates.

        Only affects layers that have stored STDP accumulators
        (i.e., _track_traces was True during forward).
        """
        for block in model.blocks:
            mlp = block.mlp
            pre = getattr(mlp, '_stdp_pre', None)
            ltp = getattr(mlp, '_stdp_ltp_accum', None)
            ltd = getattr(mlp, '_stdp_ltd_accum', None)

            if pre is None or ltp is None or ltd is None:
                continue

            delta_w = self._compute_update(pre, ltp, ltd)

            # reward-modulated update
            mlp.up.weight.data += self.lr * reward * delta_w

            # clear stored traces
            mlp._stdp_pre = None
            mlp._stdp_ltp_accum = None
            mlp._stdp_ltd_accum = None
