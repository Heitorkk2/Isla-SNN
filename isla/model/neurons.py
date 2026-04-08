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


class _SurrogateSpike(torch.autograd.Function):
    """Custom autograd: Heaviside forward, fast-sigmoid backward."""

    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane,) = ctx.saved_tensors
        base = 1.0 + ctx.slope * (membrane - ctx.threshold).abs()
        grad = 1.0 / (base * base)
        return grad_output * grad, None, None


def spike_fn(membrane, threshold=1.0, slope=25.0):
    """Differentiable spike function via surrogate gradient."""
    return _SurrogateSpike.apply(membrane, threshold, slope)


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with per-unit learnable decay.

    The decay β is parameterised as sigmoid(raw) so it stays in (0, 1)
    without explicit clamping, and gradients flow freely.
    """

    def __init__(self, dim, beta=0.9, threshold=1.0, slope=25.0):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
        # inverse-sigmoid of the initial beta
        raw = torch.log(torch.tensor(beta / (1.0 - beta)))
        self.beta_raw = nn.Parameter(torch.full((dim,), raw.item()))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def step(self, current, membrane=None):
        """Integrate one timestep. Returns (spikes, new_membrane)."""
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = self.beta * membrane + current
        spikes = spike_fn(membrane, self.threshold, self.slope)
        membrane = membrane * (1.0 - spikes.detach())  # hard reset
        return spikes, membrane

    def multi_step(self, current, T):
        """Integrate T timesteps with the same input current (vectorised).

        When the input is constant across all timesteps (as in SpikingMLP),
        this avoids the Python-loop overhead by unrolling with in-place ops.
        Returns (spike_sum, final_membrane, mean_rate_per_unit).
        """
        beta = self.beta
        membrane = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)

        for _ in range(T):
            membrane = beta * membrane + current
            spikes = spike_fn(membrane, self.threshold, self.slope)
            membrane = membrane * (1.0 - spikes.detach())
            spike_sum = spike_sum + spikes

        # per-unit rate averaged over batch and sequence dims (keep hidden dim)
        rate_per_unit = spike_sum.mean(dim=tuple(range(spike_sum.ndim - 1))) / T
        return spike_sum, membrane, rate_per_unit

    def forward(self, currents):
        """Process a full time-series. currents: (T, *, dim)."""
        membrane = None
        all_spikes = []
        for t in range(currents.shape[0]):
            s, membrane = self.step(currents[t], membrane)
            all_spikes.append(s)
        return torch.stack(all_spikes, dim=0), membrane
