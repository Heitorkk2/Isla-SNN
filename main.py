"""Isla-SNN framework entry point.

Run from a notebook or terminal:
    !python main.py                       # uses defaults
    !python main.py --data ./corpus.jsonl  # custom dataset

All heavy lifting is delegated to `isla.train()`.
"""

import argparse
import isla


def main():
    parser = argparse.ArgumentParser(description="Isla-SNN: train a spiking language model")
    parser.add_argument("--data", default="./data/corpus.jsonl", help="dataset path (dir, jsonl, or HF name)")
    parser.add_argument("--output", default="./outputs/checkpoints", help="checkpoint directory")
    parser.add_argument("--tokenizer", default="codelion/gpt-2-70m")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-init", type=float, default=0.9)
    parser.add_argument("--spike-reg-lambda", type=float, default=0.001)
    parser.add_argument("--target-spike-rate", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="reduce VRAM")
    parser.add_argument("--resume", default="", help="checkpoint dir to resume from")
    parser.add_argument("--wandb", action="store_true", help="enable W&B logging")
    parser.add_argument("--wandb-project", default="isla-snn", help="W&B project name")
    parser.add_argument("--wandb-run", default="", help="W&B run name (auto if empty)")
    args = parser.parse_args()

    model_config = isla.ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        tokenizer_name=args.tokenizer,
        threshold=args.threshold,
        beta_init=args.beta_init,
        spike_reg_lambda=args.spike_reg_lambda,
        target_spike_rate=args.target_spike_rate,
    )

    train_config = isla.TrainConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint=isla.CheckpointConfig(
            output_dir=args.output,
            resume_from=args.resume,
        ),
        wandb=isla.WandbConfig(
            enabled=args.wandb,
            project=args.wandb_project,
            run_name=args.wandb_run,
        ),
    )

    data_config = isla.DataConfig(
        dataset_path=args.data,
        tokenizer_name=args.tokenizer,
        max_seq_len=args.seq_len,
    )

    isla.train(model_config, train_config, data_config)


if __name__ == "__main__":
    main()
