"""Neuromorphic Operation and Energy Profiler for Isla-SNN.

Calculates the exact computational savings (MACs vs ACs) and estimates the
energy consumption (in Picojoules per token) based on the model's actual spike
rates during an inference pass.
"""

import os
import sys
import argparse
import torch
import numpy as np

import isla
from isla.model.architecture import IslaModel
from isla.config import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Isla-SNN Neuromorphic Energy Profiler")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="outputs/checkpoints/final",
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="O cerebro humano e uma maquina espetacular que funciona com apenas 20 watts de potencia.",
        help="Prompt text to run the profiling inference on"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on"
    )
    return parser.parse_args()


def format_si(val):
    """Format large numbers with SI prefixes (K, M, B, T)."""
    if val >= 1e12:
        return f"{val / 1e12:.2f} T"
    elif val >= 1e9:
        return f"{val / 1e9:.2f} B"
    elif val >= 1e6:
        return f"{val / 1e6:.2f} M"
    elif val >= 1e3:
        return f"{val / 1e3:.2f} K"
    return f"{val:.2f}"


def run_profiler():
    args = parse_args()
    
    print("=" * 70)
    print("        ISLA-SNN NEUROMORPHIC OPERATION & ENERGY PROFILER")
    print("=" * 70)
    
    # 1. Load model and tokenizer
    if os.path.exists(args.checkpoint):
        print(f"[*] Loading model checkpoint from: {args.checkpoint} ...")
        try:
            model, tokenizer = isla.load_model(args.checkpoint, device=args.device)
            config = model.config
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}. Falling back to default ModelConfig.")
            config = ModelConfig()
            config.vocab_size = 50257
            model = IslaModel(config).to(args.device)
            tokenizer = None
    else:
        print(f"[!] Checkpoint not found at '{args.checkpoint}'.")
        print("[*] Creating a random model from default config for profiling demo...")
        config = ModelConfig()
        config.vocab_size = 50257
        model = IslaModel(config).to(args.device)
        tokenizer = None

    model.eval()
    
    # 2. Prepare input
    if tokenizer is not None:
        input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
        tokens_len = input_ids.shape[1]
        print(f"[*] Input prompt: \"{args.prompt}\"")
        print(f"[*] Tokenized length: {tokens_len} tokens")
    else:
        # Dummy input for demo if tokenizer not available
        tokens_len = 32
        input_ids = torch.randint(0, config.vocab_size, (1, tokens_len), device=args.device)
        print(f"[*] Using dummy input of {tokens_len} tokens for profiling.")

    print(f"[*] Model size: {model.count_params():,} parameters")
    print(f"[*] Configuration: {config.num_layers} Layers, {config.hidden_dim} Hidden Dim, {config.ff_mult}x MLP expansion")
    print(f"[*] LIF Neurons: {config.num_timesteps} integration steps")
    print("-" * 70)

    # Hook to capture layer activations / spike rates per layer
    spike_rates = []
    layer_diagnostics = []
    
    # We will compute:
    # d_in = hidden_dim
    # d_ff = hidden_dim * ff_mult
    # Standard dense FFN layer has 2 linear projections:
    # 1. Up-projection (d_in -> d_ff) -> d_in * d_ff parameters
    # 2. Down-projection (d_ff -> d_in) -> d_ff * d_in parameters
    # Total dense FFN MACs per layer per token = 2 * d_in * d_ff
    
    d_in = config.hidden_dim
    d_ff = d_in * config.ff_mult
    ffn_dense_macs_per_token_per_layer = 2 * d_in * d_ff
    
    # Neuromorphic Reference energy values:
    # Based on standard neuromorphic and CMOS hardware benchmarks:
    # 1 MAC (45nm CMOS floating point) ≈ 120 fJ (0.12 pJ)
    # 1 AC (Accumulate / event-driven weight addition) ≈ 20 fJ (0.02 pJ)
    # Bypassed operation = 0 fJ
    ENERGY_MAC_FJ = 120.0
    ENERGY_AC_FJ = 20.0
    
    with torch.no_grad():
        logits, metrics, _ = model(input_ids)
    
    # Extract actual spike rates per layer from forward pass metrics
    rates_per_layer = metrics.get("spike_rates_per_layer", [])
    if not rates_per_layer:
        # Fallback to config target or default if not logged
        rates_per_layer = [torch.tensor(0.165, device=args.device) for _ in range(config.num_layers)]
        
    print(f"{'LAYER':<10} | {'SPIKE RATE':<12} | {'Sparsity %':<12} | {'Dense MACs (std)':<16} | {'SNN Ops (AC/MAC)':<18}")
    print("-" * 70)
    
    total_dense_macs = 0
    total_snn_macs = 0
    total_snn_acs = 0

    # Per-layer alpha: the learned weight on the continuous membrane term.
    alphas = [b.mlp.alpha.item() for b in model.blocks]

    for i, rate_tensor in enumerate(rates_per_layer):
        rate = rate_tensor.mean().item()
        sparsity = (1.0 - rate) * 100.0
        
        # SNN up-project is dense (continuous input -> LIF): d_in * d_ff MACs
        layer_snn_macs = d_in * d_ff

        # SNN down-project: SpikingMLP feeds it `rate + alpha * membrane`.
        # By linearity that splits into down(rate) + alpha * down(membrane).
        #   down(rate)     -> event-driven, one accumulate per active connection
        #   down(membrane) -> membrane is CONTINUOUS, so this is a full dense MAC
        #                     over every unit whether it spiked or not
        # A small alpha does not help: any nonzero coefficient still requires the
        # dense matmul. Counting only the accumulates (as this profiler did until
        # the alpha ablation was run) overstates the saving by ~2x.
        layer_snn_acs = rate * d_ff * d_in
        alpha = alphas[i] if i < len(alphas) else 0.0
        if alpha > 0:
            layer_snn_macs += d_ff * d_in

        layer_dense_macs = ffn_dense_macs_per_token_per_layer
        
        total_dense_macs += layer_dense_macs
        total_snn_macs += layer_snn_macs
        total_snn_acs += layer_snn_acs
        
        print(f"Layer {i:<4} | {rate*100:<10.2f}% | {sparsity:<10.2f}% | {format_si(layer_dense_macs):<16} | {format_si(layer_snn_macs)} MACs + {format_si(layer_snn_acs)} ACs")

    print("=" * 70)
    print("                     SUMMARY PER LAYER PER TOKEN")
    print("-" * 70)
    
    # Calculate energy per token for standard dense FFN
    # Total dense energy = Dense MACs * 120 fJ
    energy_dense_fj = total_dense_macs * ENERGY_MAC_FJ
    energy_dense_pj = energy_dense_fj / 1000.0
    
    # Calculate energy per token for SNN FFN
    # Total SNN energy = SNN MACs * 120 fJ + SNN ACs * 20 fJ
    energy_snn_fj = (total_snn_macs * ENERGY_MAC_FJ) + (total_snn_acs * ENERGY_AC_FJ)
    energy_snn_pj = energy_snn_fj / 1000.0
    
    energy_saved_pct = (1.0 - (energy_snn_fj / energy_dense_fj)) * 100.0
    
    mean_rate = sum([r.mean().item() for r in rates_per_layer]) / len(rates_per_layer)
    
    print(f"[*] Mean Spiking Rate: {mean_rate*100:.2f}%")
    print(f"[*] FFN Bypassed Connections: {(1.0 - mean_rate)*100:.2f}%")
    print(f"[*] Standard FFN Operations:  {format_si(total_dense_macs)} MACs")
    print(f"[*] Isla-SNN FFN Operations:  {format_si(total_snn_macs)} MACs + {format_si(total_snn_acs)} ACs")
    print("-" * 70)
    print(f"[*] Standard FFN Energy:      {energy_dense_pj:,.2f} pJ / token")
    print(f"[*] Isla-SNN FFN Energy:      {energy_snn_pj:,.2f} pJ / token")
    label = "Energy Saved" if energy_saved_pct > 0 else "Energy COST"
    print(f"[*] Neuromorphic {label}: {abs(energy_saved_pct):.2f}%")

    mean_alpha = sum(alphas) / len(alphas)
    if mean_alpha > 0:
        print("-" * 70)
        print(f"[!] alpha = {mean_alpha:.4f} > 0: the continuous membrane term forces a")
        print("    dense down-projection, so the FFN is NOT event-driven. The figure")
        print("    above already accounts for that. Run experiments/alpha_ablation.py")
        print("    to measure what alpha buys before trying to remove it.")
    print("=" * 70)
    
    # 3. Dynamic scale estimation
    # Let's estimate for generating a full page of text (e.g. 500 tokens)
    tokens_num = 500
    energy_saved_joules = (energy_dense_pj - energy_snn_pj) * 1e-12 * tokens_num
    print(f"[*] Estimated savings for 500 generated tokens:")
    print(f"    - Dense FFN energy:      {energy_dense_pj * tokens_num * 1e-6:.4f} uJ")
    print(f"    - Isla-SNN FFN energy:   {energy_snn_pj * tokens_num * 1e-6:.4f} uJ")
    print(f"    - Absolute Energy Saved: {(energy_dense_pj - energy_snn_pj) * tokens_num * 1e-6:.4f} uJ")
    print(f"    - Neuromorphic Factor:   {energy_dense_pj / energy_snn_pj:.2f}x")

    print("=" * 70)
    print("Scope and assumptions — read before quoting these numbers:")
    print("  * FFN only. Attention projections and the LM head are NOT counted,")
    print("    and together they are ~48% of per-token MACs, so whole-model")
    print("    savings are roughly half of the FFN-only figure above.")
    print("  * 45nm CMOS for MACs (120 fJ) and a neuromorphic ASIC for ACs (20 fJ).")
    print("    No such chip runs this architecture today; this is a model, not a")
    print("    measurement. On GPU the SNN is slower than a dense transformer.")
    print("  * Rate coding trades precision for events: with T timesteps the AC")
    print("    count scales with T, and event-driven stops beating dense at")
    print("    T ~ 21 (only ~4.5 bits of precision). See README.")
    print("=" * 70)


if __name__ == "__main__":
    run_profiler()
