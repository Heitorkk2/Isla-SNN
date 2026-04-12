"""Interactive chat REPL for Isla-SNN.

Usage:
    python chat.py                                     # default checkpoint
    python chat.py --ckpt ./outputs/finetuned/final    # custom checkpoint
"""

import torch
from transformers import AutoTokenizer
from isla.config import ModelConfig
from isla.model.architecture import IslaModel
from isla.inference.generate import generate_stream
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./outputs/checkpoints/latest",
                        help="Path to checkpoint dir")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig.load(f"{args.ckpt}/model_config.json")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)

    model = IslaModel(model_cfg).to(device)
    model.load_state_dict(torch.load(f"{args.ckpt}/model.pth",
                                     map_location=device, weights_only=True))
    model.eval()

    print(f"\nModel: {model_cfg.hidden_dim}d, {model_cfg.num_layers}L, "
          f"vocab={model_cfg.vocab_size}")
    print(f"Tokenizer: {model_cfg.tokenizer_name}")
    print(f"Device: {device}")
    print("Type 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            prompt = input("\n[You]: ")
            if prompt.lower() in ["exit", "quit"]:
                break

            print("[Isla]: ", end="", flush=True)
            for piece in generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=40,
                device=device
            ):
                print(piece, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
