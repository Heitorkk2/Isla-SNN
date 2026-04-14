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


def spike_fn(membrane, threshold=1.0, slope=25.0):
    """Differentiable spike function via surrogate gradient.
    Implemented purely in PyTorch (Straight-Through Estimator).
    This bypasses torch.autograd.Function entirely, preventing
    massive Graph Breaks when used with torch.compile().
    """
    # Forward: Hard Heaviside step
    hard_spike = (membrane >= threshold).float()
    
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

    The decay β is parameterised as sigmoid(raw) so it stays in (0, 1)
    without explicit clamping, and gradients flow freely.
    """

    def __init__(self, dim, beta=0.9, threshold=1.0, slope=25.0,
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

    def multi_step(self, current, T):
        """Integrate T timesteps with the same input current.

        Returns (spike_sum, final_membrane, mean_rate_per_unit).
        final_membrane is exposed so SpikingMLP can use it as a
        continuous feature alongside binary spikes (v3 improvement).
        """
        beta = self.beta
        membrane = torch.zeros_like(current)
        adaptation = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)

        for _ in range(T):
            threshold = self.base_threshold + adaptation
            membrane = beta * membrane + current
            spikes = spike_fn(membrane, threshold, self.slope)
            membrane = membrane * (1.0 - spikes.detach())
            adaptation = self.adaptation_decay * adaptation + \
                         torch.relu(self.adaptation_strength) * spikes.detach()
            spike_sum = spike_sum + spikes

        # per-unit rate averaged over batch and sequence dims (keep hidden dim)
        rate_per_unit = spike_sum.mean(dim=tuple(range(spike_sum.ndim - 1))) / T
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
