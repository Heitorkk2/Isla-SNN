"""Configuration dataclasses for the Isla-SNN framework.

All hyperparameters live here as plain dataclasses, serializable to JSON.
The API mirrors HuggingFace conventions so checkpoints are self-contained.
"""

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Defines the model architecture. Saved alongside weights as model_config.json."""

    model_type: str = "isla-snn"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_timesteps: int = 4         # LIF integration steps per spiking MLP
    max_seq_len: int = 1024
    dropout: float = 0.1
    ff_mult: int = 4

    # LIF neuron parameters
    beta_init: float = 0.9         # membrane decay β₀ (learnable, constrained to (0,1))
    threshold: float = 1.0         # spike threshold θ
    surrogate_slope: float = 25.0  # steepness k of the surrogate gradient

    # spike synchrony attention
    sync_tau_init: float = 1.0     # initial temperature τ₀ for the RBF kernel

    # regularization
    spike_reg_lambda: float = 1e-3 # weight of the spike-rate penalty term
    target_spike_rate: float = 0.0 # if > 0, penalise |rate - target|² instead of rate

    # ablation
    use_standard_attention: bool = False  # swap sync attention for dot-product

    # speed
    compile: bool = False  # torch.compile() the model

    # set dynamically from tokenizer before model construction
    vocab_size: int = 0

    # metadata (not used by the model, but stored for reproducibility)
    tokenizer_name: str = "codelion/gpt-2-70m"

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        valid_keys = set(cls.__dataclass_fields__)
        unknown = [k for k in data if k not in valid_keys and not k.startswith("_")]
        if unknown:
            warnings.warn(f"Unknown config keys (ignored): {unknown}. Typo?", stacklevel=2)
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class CheckpointConfig:
    """Controls when and where checkpoints are saved."""

    output_dir: str = "./outputs/checkpoints"
    save_every: int = 2000         # save step_N/ every N steps
    save_best: bool = True         # save best/ when val_loss improves
    save_final: bool = True        # save final/ when training completes
    resume_from: str = ""          # path to checkpoint dir to resume from


@dataclass
class WandbConfig:
    """Weights & Biases logging. Set enabled=False or omit wandb install to skip."""

    enabled: bool = False
    project: str = "isla-snn"
    run_name: str = ""
    log_freq: int = 1  # log every N train-log events (1 = every log_every steps)


@dataclass
class DataConfig:
    """Points the trainer to a dataset and tokenizer."""

    dataset_path: str = ""
    tokenizer_name: str = "codelion/gpt-2-70m"
    max_seq_len: int = 1024
    validation_split: float = 0.001
    num_workers: int = 2
    num_proc: int = 4
    pack_sequences: bool = True  # concatenate+chunk instead of padding (faster training)


@dataclass
class TrainConfig:
    """Training hyperparameters (optimizer, schedule, precision, logging)."""

    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 0           # 0 = auto-compute from num_epochs
    num_epochs: int = 1
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    gradient_checkpointing: bool = False  # trade compute for VRAM

    bf16: bool = True
    fp16: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive. Pick one.")


    log_every: int = 50
    eval_every: int = 500

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
