"""Isla-SNN: Spiking Neural Network Language Model with Spike Synchrony Attention.

v3 features:
    - Gated additive residual (spikes always active + learnable boost)
    - 8 timesteps (256 temporal patterns per neuron)
    - Membrane potential as continuous feature (spike + α·membrane)
    - Spike Frequency Adaptation (dynamic threshold)

Notebook usage:
    import isla

    model_config = isla.ModelConfig(hidden_dim=768, num_layers=12, num_heads=12,
                                    num_timesteps=8, max_seq_len=2048)
    train_config = isla.TrainConfig(lr=3e-4, batch_size=4,
                                    gradient_accumulation_steps=8)
    data_config  = isla.DataConfig(dataset_path="./data/corpus_v3.jsonl",
                                   tokenizer_name="NousResearch/Llama-2-7b-hf")

    model, tokenizer = isla.train(model_config, train_config, data_config)
    text = isla.generate(model, tokenizer, "Hello")

CLI usage:
    python main.py --data ./data/corpus_v3.jsonl --hidden-dim 768 --num-layers 12
"""

from typing import Optional, Tuple

from .config import ModelConfig, TrainConfig, DataConfig, CheckpointConfig, WandbConfig
from .model import IslaModel
from .inference import generate, generate_stream, setup_speed
from .training import IslaTrainer


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    data_config: DataConfig,
) -> Tuple["IslaModel", "object"]:
    """Train from config objects. Returns (model, tokenizer)."""
    from .data import get_tokenizer, load_isla_dataset, create_dataloader
    import torch

    torch.manual_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ISLA] device={device}")

    tokenizer = get_tokenizer(data_config.tokenizer_name)

    # align vocab to 64 for tensor-core efficiency
    model_config.vocab_size = len(tokenizer)
    if model_config.vocab_size % 64 != 0:
        model_config.vocab_size = ((model_config.vocab_size // 64) + 1) * 64
    print(f"[ISLA] vocab={model_config.vocab_size}")

    dataset = load_isla_dataset(data_config.dataset_path, tokenizer,
                                data_config.max_seq_len, data_config.num_proc,
                                pack=data_config.pack_sequences)

    train_dl = create_dataloader(dataset["train"], train_config.batch_size,
                                 shuffle=True, num_workers=data_config.num_workers,
                                 drop_last=True, seed=train_config.seed)
    val_dl = None
    if "validation" in dataset:
        val_dl = create_dataloader(dataset["validation"], train_config.batch_size,
                                   shuffle=False, num_workers=data_config.num_workers,
                                   drop_last=False, seed=train_config.seed)

    model = IslaModel(model_config).to(device)
    print(f"[ISLA] params={model.count_params():,}")

    # optional torch.compile
    if model_config.compile:
        model = setup_speed(compile_model=True, model=model)
    else:
        setup_speed(compile_model=False)

    trainer = IslaTrainer(model, train_config, model_config)
    trainer.train(train_dl, val_dl)
    return model, tokenizer


def load_model(
    checkpoint_dir: str,
    device: str = "cpu",
) -> Tuple["IslaModel", "object"]:
    """Load (model, tokenizer) from a checkpoint directory."""
    from .data import get_tokenizer
    model = IslaModel.from_pretrained(checkpoint_dir, device=device)
    tokenizer = get_tokenizer(model.config.tokenizer_name)
    return model, tokenizer


__version__ = "0.3.0"
__all__ = [
    "ModelConfig", "TrainConfig", "DataConfig", "CheckpointConfig", "WandbConfig",
    "IslaModel", "IslaTrainer",
    "train", "load_model", "generate", "generate_stream", "setup_speed",
]
