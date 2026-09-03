"""Configuration dataclasses for the Isla-SNN framework.

All hyperparameters live here as plain dataclasses, serializable to JSON.
The API mirrors HuggingFace conventions so checkpoints are self-contained.
"""

import json
import warnings
from typing import Optional
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
    # Steepness k of the fast-sigmoid surrogate, f'(x) = 1/(1+k|x|)².
    # Lower k passes MORE gradient, not less: measured gradient mass is 0.333
    # at k=2, 0.167 at k=5, 0.063 at k=15, 0.039 at k=25 — so k=25 delivers
    # 4.3x less signal than k=5. Values of 2-10 are the usual range for this
    # surrogate; 5 is the tested default. Raising k narrows the window to
    # neurons near threshold, which is a better match to the true Heaviside
    # derivative but starves everything else.
    surrogate_slope: float = 5.0

    # spike synchrony attention
    sync_tau_init: float = 1.0     # initial temperature τ₀ for the RBF kernel

    # regularization
    spike_reg_lambda: float = 1e-3 # weight of the spike-rate penalty term
    target_spike_rate: float = 0.0 # if > 0, penalise |rate - target|² instead of rate

    # ablation
    use_standard_attention: bool = False  # swap sync attention for dot-product
    use_spike_ssm: bool = False          # swap sync attention for O(N) state space

    # residual topology
    # identity: h = h + gate * mlp(h), stable pre-norm Transformer residual
    # spike_first: h = mlp(h) + gate * h, legacy v3 checkpoint behaviour
    mlp_residual_mode: str = "identity"

    # speed
    compile: bool = False  # torch.compile() the model

    # set dynamically from tokenizer before model construction
    vocab_size: int = 0

    # metadata (not used by the model, but stored for reproducibility)
    tokenizer_name: str = "codelion/gpt-2-70m"

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim and num_heads must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.num_layers <= 0 or self.num_timesteps <= 0:
            raise ValueError("num_layers and num_timesteps must be positive.")
        if self.use_standard_attention and self.use_spike_ssm:
            raise ValueError(
                "use_standard_attention and use_spike_ssm are mutually exclusive."
            )
        if self.mlp_residual_mode not in {"identity", "spike_first"}:
            raise ValueError(
                "mlp_residual_mode must be 'identity' or 'spike_first'."
            )

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        data = json.loads(raw.decode(encoding))
        # Preserve forward semantics for checkpoints created before the
        # identity residual was introduced.
        data.setdefault("mlp_residual_mode", "spike_first")
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
    is_finetune: bool = False    # if True, masks the prompt in the loss calculation
    response_template: str = "<|im_start|>assistant\n" # the boundary token to calculate loss after

    def __post_init__(self):
        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be greater than 1.")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in the [0, 1) interval.")
        if self.num_workers < 0 or self.num_proc <= 0:
            raise ValueError("num_workers must be non-negative and num_proc positive.")


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
    
    # SNN physics override
    surrogate_slope: float = 5.0  # overrides model_config.surrogate_slope if set during training
    surrogate_slope_final: Optional[float] = None  # if set, dynamically schedule slope from surrogate_slope to surrogate_slope_final

    # R-STDP (biological learning for spiking MLP layers)
    use_rstdp: bool = False           # enable hybrid backprop + R-STDP
    rstdp_lr: float = 1e-3            # R-STDP learning rate (separate from backprop lr)
    rstdp_tau_plus: float = 20.0      # LTP trace decay time constant
    rstdp_tau_minus: float = 20.0     # LTD trace decay time constant
    rstdp_a_plus: float = 0.01        # LTP amplitude
    rstdp_a_minus: float = 0.0105     # LTD amplitude (slightly > A+ for stability)

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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
