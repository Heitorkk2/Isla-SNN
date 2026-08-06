"""Interactive chat REPL for Isla-SNN.

Usage:
    python chat.py                                          # defaults
    python chat.py --ckpt ./outputs                         # custom checkpoint
    python chat.py --temp 0.9 --top-k 50 --max-tokens 200  # sampling knobs
    python chat.py --raw                                    # skip Instruction template
    python chat.py --no-stats                               # hide stat footer

All sampling parameters can be changed mid-conversation with /set:
    /set temp 0.9
    /set top_k 50
    /set max_tokens 200

Commands:
    /help       Show all commands
    /system     Change the system instruction
    /clear      Clear screen (also works without /)
    /params     Show current parameters
    exit/quit   Exit chat
"""

import time
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from isla.config import ModelConfig
from isla.model.architecture import IslaModel
from isla.model.attention import KVCache, SSMCache


# ── ANSI Colors ──────────────────────────────────────────────────────────
PINK    = "\033[38;5;205m"
CYAN    = "\033[36m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
RED     = "\033[31m"
RESET   = "\033[0m"


# ── Generation with metrics ─────────────────────────────────────────────
@torch.no_grad()
def generate_with_stats(model, tokenizer, prompt, max_new_tokens=100,
                        temperature=0.8, top_k=40, top_p=0.9,
                        repetition_penalty=1.1, device="cpu"):
    """Generate text token-by-token, yielding each piece. Returns stats dict."""
    from isla.inference.generate import _filter_logits, _sample_next

    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if ids.numel() == 0:
        return

    generated = ids
    prev_text = tokenizer.decode(ids[0], skip_special_tokens=True)

    # Prefill pass
    cache_cls = SSMCache if getattr(model.config, 'use_spike_ssm', False) else KVCache
    caches = [cache_cls() for _ in range(model.config.num_layers)]
    logits, metrics, caches = model(ids, caches=caches)

    # Stats accumulators
    all_spike_rates = [metrics["mean_spike_rate"].item()]
    all_spike_stds  = [metrics["spike_rate_std"].item()]
    total_logprob   = 0.0
    num_tokens      = 0
    t_start         = time.perf_counter()

    for _ in range(max_new_tokens):
        token = _sample_next(logits, generated, temperature, top_k, top_p, repetition_penalty)
        generated = torch.cat([generated, token], dim=1)

        # Compute token log-prob for perplexity
        probs = F.softmax(logits[:, -1, :], dim=-1)
        token_prob = probs[0, token.item()].clamp(min=1e-12)
        total_logprob += torch.log(token_prob).item()
        num_tokens += 1

        # Decode and yield new text
        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        new_text = full_text[len(prev_text):]
        prev_text = full_text
        yield new_text

        if token.item() == tokenizer.eos_token_id:
            break

        # Next step with KV cache
        logits, metrics, caches = model(token, caches=caches)
        all_spike_rates.append(metrics["mean_spike_rate"].item())
        all_spike_stds.append(metrics["spike_rate_std"].item())

    elapsed = time.perf_counter() - t_start

    # Attach stats to the generator (accessible after exhaustion)
    avg_spike = sum(all_spike_rates) / len(all_spike_rates) if all_spike_rates else 0
    avg_std   = sum(all_spike_stds)  / len(all_spike_stds)  if all_spike_stds else 0
    ppl       = torch.exp(torch.tensor(-total_logprob / max(num_tokens, 1))).item()

    generate_with_stats._last_stats = {
        "tokens":     num_tokens,
        "elapsed_s":  elapsed,
        "tok_per_s":  num_tokens / max(elapsed, 1e-6),
        "perplexity": ppl,
        "spike_rate": avg_spike,
        "spike_std":  avg_std,
        "prompt_len": ids.shape[1],
    }


# ── Pretty-print stats ──────────────────────────────────────────────────
def print_stats():
    s = generate_with_stats._last_stats
    print(f"\n{DIM}{'─' * 55}")
    print(f"  ⚡ {s['tokens']} tokens in {s['elapsed_s']:.2f}s "
          f"({CYAN}{s['tok_per_s']:.1f} tok/s{RESET}{DIM})  "
          f"│  prompt: {s['prompt_len']} tok")
    print(f"  📊 ppl: {YELLOW}{s['perplexity']:.1f}{RESET}{DIM}  "
          f"│  spike: {GREEN}{s['spike_rate']:.3f}{RESET}{DIM} ± {s['spike_std']:.3f}")
    print(f"{'─' * 55}{RESET}")


# ── Wrap prompt in Alpaca template ───────────────────────────────────────
def wrap_prompt(user_input, system_msg=None, inst_tag="### Instruction:", resp_tag="### Response:"):
    """Wrap user text in a configurable template. 
    Injects system msg INSIDE the instruction block for better focus on small models.
    """
    if system_msg:
        full_inst = f"{system_msg}\n\n{user_input}"
    else:
        full_inst = user_input
        
    return f"{inst_tag}\n{full_inst}\n\n{resp_tag}\n"


# ── Help Menu ───────────────────────────────────────────────────────────
def print_help():
    print(f"\n{BOLD}  Available Commands:{RESET}")
    print(f"  {CYAN}/help{RESET}             Show this help menu")
    print(f"  {CYAN}/system <text>{RESET}    Change the system/preamble message")
    print(f"  {CYAN}/set <k> <v>{RESET}      Update sampling (temp, top_k, max_tokens, etc.)")
    print(f"  {CYAN}/params{RESET}           Show current sampling parameters")
    print(f"  {CYAN}/clear{RESET}            Clear terminal screen")
    print(f"  {CYAN}exit / quit{RESET}       End the chat session\n")


# ── /set command handler ────────────────────────────────────────────────
def handle_set_command(args_str, params):
    """Parse /set key value and update params dict."""
    parts = args_str.strip().split(maxsplit=1)
    if len(parts) != 2:
        print(f"  {RED}Usage: /set <key> <value>{RESET}")
        print(f"  {DIM}Keys: temp, top_k, top_p, max_tokens, rep_penalty{RESET}")
        return

    key, val = parts
    key_map = {
        "temp":         ("temperature",        float),
        "temperature":  ("temperature",        float),
        "top_k":        ("top_k",              int),
        "top_p":        ("top_p",              float),
        "max_tokens":   ("max_new_tokens",     int),
        "max_new_tokens": ("max_new_tokens",   int),
        "rep_penalty":  ("repetition_penalty", float),
        "repetition_penalty": ("repetition_penalty", float),
    }

    if key not in key_map:
        print(f"  {RED}Unknown key '{key}'. Available: {', '.join(key_map.keys())}{RESET}")
        return

    param_name, cast_fn = key_map[key]
    try:
        params[param_name] = cast_fn(val)
        print(f"  {GREEN}✓ {param_name} = {params[param_name]}{RESET}")
    except ValueError:
        print(f"  {RED}Invalid value '{val}' for {param_name}{RESET}")


# ── Main REPL ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with Isla-SNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="In-chat commands:\n"
               "  /set <key> <value>   Change a sampling parameter\n"
               "  /params              Show current parameters\n"
               "  /clear               Clear screen\n"
               "  exit / quit          Exit\n"
    )
    parser.add_argument("--ckpt", type=str, default="./outputs",
                        help="Path to checkpoint dir (default: ./outputs)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k filtering (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens to generate (default: 150)")
    parser.add_argument("--rep-penalty", type=float, default=1.15,
                        help="Repetition penalty (default: 1.15)")
    parser.add_argument("--system", type=str, default="Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente a solicitação.",
                        help="System instruction / preamble")
    parser.add_argument("--inst-tag", type=str, default="### Instruction:",
                        help="Tag to start the instruction block (default: ### Instruction:)")
    parser.add_argument("--resp-tag", type=str, default="### Response:",
                        help="Tag to start the response block (default: ### Response:)")
    parser.add_argument("--raw", action="store_true",
                        help="Send prompt as-is (no Instruction template)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Hide stats footer after each response")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────
    print(f"\n{PINK}{BOLD}  ✦ Isla-SNN Chat ✦{RESET}")
    print(f"{DIM}  Loading from {args.ckpt}...{RESET}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    model_cfg = ModelConfig.load(f"{args.ckpt}/model_config.json")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = IslaModel(model_cfg).to(device)
    model.load_state_dict(torch.load(f"{args.ckpt}/model.pth",
                                     map_location=device, weights_only=True))
    model.eval()

    # ── Print banner ─────────────────────────────────────────────────
    param_count = model.count_params()
    print(f"{DIM}{'─' * 55}")
    print(f"  Model:     {CYAN}{param_count:,}{RESET}{DIM} params "
          f"({model_cfg.hidden_dim}d × {model_cfg.num_layers}L × {model_cfg.num_heads}H)")
    print(f"  Tokenizer: {model_cfg.tokenizer_name}")
    print(f"  Device:    {CYAN}{device}{RESET}{DIM} | amp: {amp_dtype}")
    if args.raw:
        print(f"  Template:  {YELLOW}raw (off){RESET}")
    else:
        print(f"  Template:  {BOLD}{args.inst_tag}{RESET} ... {BOLD}{args.resp_tag}{RESET}")
    print(f"{'─' * 55}{RESET}")
    print(f"  {DIM}Type a message, /help for commands, or 'exit' to quit.{RESET}\n")

    # ── Sampling params (mutable via /set) ───────────────────────────
    params = {
        "temperature":        args.temp,
        "top_k":              args.top_k,
        "top_p":              args.top_p,
        "max_new_tokens":     args.max_tokens,
        "repetition_penalty": args.rep_penalty,
    }
    system_msg = args.system

    # ── Chat loop ────────────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}  [You]{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}  Bye! 👋{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"{DIM}  Bye! 👋{RESET}")
            break

        # ── Slash commands ───────────────────────────────────────────
        cmd_lower = user_input.lower()
        if cmd_lower == "/help":
            print_help()
            continue
        if cmd_lower.startswith("/system"):
            new_sys = user_input[7:].strip()
            if new_sys:
                system_msg = new_sys
                print(f"  {GREEN}✓ System message updated.{RESET}")
            else:
                print(f"  {DIM}Current system: {CYAN}{system_msg}{RESET}")
            continue
        if cmd_lower.startswith("/set "):
            handle_set_command(user_input[5:], params)
            continue
        if cmd_lower == "/params":
            print(f"  {DIM}system: {CYAN}{system_msg}{RESET}")
            for k, v in params.items():
                print(f"  {DIM}{k}: {CYAN}{v}{RESET}")
            continue
        if cmd_lower in ("/clear", "clear"):
            import os
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # ── Generate ─────────────────────────────────────────────────
        if not user_input:
            continue
            
        prompt = user_input if args.raw else wrap_prompt(user_input, system_msg, args.inst_tag, args.resp_tag)

        print(f"{BOLD}{PINK}  [Isla]{RESET} ", end="", flush=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            for piece in generate_with_stats(
                model, tokenizer, prompt,
                device=str(device),
                **params,
            ):
                print(piece, end="", flush=True)

        print()  # newline after response

        if not args.no_stats:
            print_stats()

    print()


if __name__ == "__main__":
    main()
