"""Performance utilities for Isla-SNN.

Centralises speed-related setup so it's easy to call from notebooks or
main.py without duplicating CUDA tuning code everywhere.
"""

import torch


def setup_speed(compile_model=False, model=None):
    """Apply speed optimizations. Call once before training or inference.

    Args:
        compile_model: if True and PyTorch >= 2.0, torch.compile() the model
        model: the IslaModel instance to compile (required if compile_model=True)

    Returns:
        The (possibly compiled) model, or None if model was None.
    """
    # tensor-core friendly matmul (trades precision for throughput on ampere+)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # let cuDNN auto-tune convolution algorithms (small overhead on first call)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if compile_model and model is not None:
        if hasattr(torch, "compile"):
            print("[SPEED] torch.compile() applied")
            model = torch.compile(model)
        else:
            print("[SPEED] torch.compile() requires PyTorch >= 2.0, skipping")

    return model
