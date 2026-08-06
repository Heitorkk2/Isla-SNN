"""Spiking neurons with surrogate gradients.

The forward pass uses a hard Heaviside step (binary spikes), while the
backward pass uses a smooth surrogate so gradients can flow through the
spike operation.

LIF dynamics (per timestep):
    V[t] = β · V[t-1] + I[t]
    S[t] = Θ(V[t] - θ)
    V[t] = V[t] · (1 - S[t])       # hard reset

Surrogate (fast sigmoid):
    ∂S/∂V ≈ 1 / (1 + k·|V - θ|)²

References:
    Neftci et al., "Surrogate Gradient Learning in SNNs", 2019
    Zenke & Ganguli, "SuperSpike", 2018
"""

import torch
import torch.nn as nn


def spike_fn(membrane, threshold=1.0, slope=5.0):
    """Differentiable spike function via surrogate gradient.
    Implemented purely in PyTorch (Straight-Through Estimator).
    This bypasses torch.autograd.Function entirely, preventing
    massive Graph Breaks when used with torch.compile().
    """
    # Forward: Hard Heaviside step.
    # .to(membrane.dtype) rather than .float(): a hard-coded fp32 here would
    # promote the whole STE chain and force the LIF to run in fp32 even under
    # bf16 autocast, doubling activation memory.
    hard_spike = (membrane >= threshold).to(membrane.dtype)
    
    # Backward: Surrogate Fast Sigmoid (f' = 1 / (1 + slope * |x|)^2)
    # The pure function f(x) = x / (1 + slope * |x|) yields exactly this derivative.
    x = membrane - threshold
    surrogate = x / (1.0 + slope * x.abs())
    
    # STE: forward uses hard_spike, backward flows through surrogate
    return (hard_spike - surrogate).detach() + surrogate


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with per-unit learnable decay.

    v3 additions:
    - Spike Frequency Adaptation (SFA): threshold rises after firing,
      forcing diverse neuronal activity. Controlled by learnable
      adaptation_strength (decays at fixed rate adaptation_decay).
    - multi_step returns final_membrane for use as continuous feature.
    - Online STDP accumulation: when track_traces=True, pre-computes
      weighted spike sums for LTP/LTD inline, avoiding storage of the
      full (T, B, L, D_ff) spike history tensor.

    The decay β is parameterised as sigmoid(raw) so it stays in (0, 1)
    without explicit clamping, and gradients flow freely.
    """

    def __init__(self, dim, beta=0.9, threshold=1.0, slope=5.0,
                 adaptation_decay=0.9):
        super().__init__()
        self.base_threshold = threshold
        self.slope = slope
        self.adaptation_decay = adaptation_decay
        # inverse-sigmoid of the initial beta
        raw = torch.log(torch.tensor(beta / (1.0 - beta)))
        self.beta_raw = nn.Parameter(torch.full((dim,), raw.item()))
        # SFA: learnable adaptation strength (init near zero = subtle)
        self.adaptation_strength = nn.Parameter(torch.full((dim,), 0.1))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def step(self, current, membrane=None, adaptation=None):
        """Integrate one timestep. Returns (spikes, new_membrane, new_adaptation)."""
        if membrane is None:
            membrane = torch.zeros_like(current)
        if adaptation is None:
            adaptation = torch.zeros_like(current)
        # dynamic threshold: rises when neuron fires (spike frequency adaptation)
        threshold = self.base_threshold + adaptation
        membrane = self.beta * membrane + current
        spikes = spike_fn(membrane, threshold, self.slope)
        membrane = membrane * (1.0 - spikes.detach())  # hard reset
        # adaptation: rises on spike, decays otherwise
        adaptation = self.adaptation_decay * adaptation + \
                     torch.relu(self.adaptation_strength) * spikes.detach()
        return spikes, membrane, adaptation

    def multi_step(self, current, T, track_traces=False, stdp_decay_pre=None,
                   stdp_decay_post=None, valid_mask=None):
        """Integrate T timesteps with the same input current.

        Returns (spike_sum, final_membrane, mean_rate_per_unit).

        valid_mask, shaped (B, L, 1), excludes padding positions from
        mean_rate_per_unit. Without it the reported rate is diluted by
        <pad> tokens, which biases the spike-rate regulariser.

        If track_traces=True, returns a 4th element: a dict with
        pre-accumulated STDP tensors ('ltp_accum', 'ltd_accum') of
        shape (*input_shape) — NOT the full (T, *) spike history.
        This cuts STDP memory usage by ~(T-2)/T compared to storing
        the complete spike_history tensor.

        final_membrane is exposed so SpikingMLP can use it as a
        continuous feature alongside binary spikes (v3 improvement).
        """
        # beta and adaptation_strength are fp32 parameters; multiplying them
        # against a bf16 current promotes the whole LIF chain to fp32, which
        # doubles activation memory and drops off the tensor-core path. Casting
        # to the input dtype keeps the integration in whatever AMP selected.
        beta = self.beta.to(current.dtype)
        adapt_strength = torch.relu(self.adaptation_strength).to(current.dtype)
        membrane = torch.zeros_like(current)
        adaptation = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)

        # Online STDP accumulators (only allocated when needed)
        if track_traces:
            ltp_accum = torch.zeros_like(current)
            ltd_accum = torch.zeros_like(current)
            # Pre-compute geometric weights for all T timesteps
            device, dtype = current.device, current.dtype
            # LTP weights: geo[t] = cumsum of [1, decay_pre, decay_pre², ...]
            dp = stdp_decay_pre if stdp_decay_pre is not None else 0.95
            dq = stdp_decay_post if stdp_decay_post is not None else 0.95
            geo_ltp = torch.zeros(T, device=device, dtype=dtype)
            geo_ltd = torch.zeros(T, device=device, dtype=dtype)
            cumsum = 0.0
            for t in range(T):
                cumsum += dp ** t
                geo_ltp[t] = cumsum
            # LTD weights: reversed geometric of post decay
            cumsum = 0.0
            for t in range(T):
                cumsum += dq ** t
                geo_ltd[t] = cumsum
            geo_ltd = geo_ltd.flip(0)

        for t in range(T):
            threshold = self.base_threshold + adaptation
            membrane = beta * membrane + current
            spikes = spike_fn(membrane, threshold, self.slope)
            membrane = membrane * (1.0 - spikes.detach())
            adaptation = self.adaptation_decay * adaptation + \
                         adapt_strength * spikes.detach()
            spike_sum = spike_sum + spikes

            if track_traces:
                s_det = spikes.detach()
                ltp_accum = ltp_accum + geo_ltp[t] * s_det
                ltd_accum = ltd_accum + geo_ltd[t] * s_det

        # per-unit rate averaged over batch and sequence dims (keep hidden dim)
        reduce_dims = tuple(range(spike_sum.ndim - 1))
        if valid_mask is None:
            rate_per_unit = spike_sum.mean(dim=reduce_dims) / T
        else:
            m = valid_mask.to(spike_sum.dtype)
            n_valid = m.sum().clamp(min=1.0)
            rate_per_unit = (spike_sum * m).sum(dim=reduce_dims) / (n_valid * T)

        if track_traces:
            stdp_data = {
                'ltp_accum': ltp_accum.detach(),  # (B, L, D_ff)
                'ltd_accum': ltd_accum.detach(),   # (B, L, D_ff)
            }
            return spike_sum, membrane, rate_per_unit, stdp_data
        return spike_sum, membrane, rate_per_unit

    def forward(self, currents):
        """Process a full time-series. currents: (T, *, dim)."""
        membrane = None
        adaptation = None
        all_spikes = []
        for t in range(currents.shape[0]):
            s, membrane, adaptation = self.step(currents[t], membrane, adaptation)
            all_spikes.append(s)
        return torch.stack(all_spikes, dim=0), membrane
