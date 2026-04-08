"""Autoregressive text generation with KV cache and streaming.

generate()        → returns full decoded string
generate_stream() → yields token strings one-by-one (for real-time UIs)
"""

import torch
import torch.nn.functional as F
from isla.model.attention import KVCache


def _filter_logits(logits, top_k=0, top_p=0.0):
    """Zero out low-probability tokens via top-k and/or nucleus sampling."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[..., -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

    return logits


def _sample_next(logits, generated_ids, temperature, top_k, top_p, repetition_penalty):
    """Apply sampling controls and return one token id."""
    nxt = logits[:, -1, :].clone()

    # vectorised repetition penalty (no Python loop over token set)
    if repetition_penalty != 1.0:
        score = torch.gather(nxt, 1, generated_ids)
        score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        nxt.scatter_(1, generated_ids, score)

    if temperature != 1.0:
        nxt = nxt / temperature

    nxt = _filter_logits(nxt, top_k=top_k, top_p=top_p)
    return torch.multinomial(F.softmax(nxt, dim=-1), 1)


@torch.no_grad()
def generate_stream(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8,
                    top_k=40, top_p=0.9, repetition_penalty=1.1, device="cpu",
                    use_cache=True):
    """Yield decoded token strings as they're produced.

    Only generated tokens are yielded (not the prompt itself).

    Usage:
        print(prompt, end="")
        for piece in isla.generate_stream(model, tokenizer, prompt):
            print(piece, end="", flush=True)
    """
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = ids

    caches = [KVCache() for _ in range(model.config.num_layers)] if use_cache else None
    logits, _, caches = model(ids, caches=caches)

    for _ in range(max_new_tokens):
        token = _sample_next(logits, generated, temperature, top_k, top_p, repetition_penalty)
        generated = torch.cat([generated, token], dim=1)

        yield tokenizer.decode(token[0], skip_special_tokens=True)

        if token.item() == tokenizer.eos_token_id:
            break

        if use_cache:
            logits, _, caches = model(token, caches=caches)
        else:
            ctx = generated[:, -model.config.max_seq_len:]
            logits, _, _ = model(ctx)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8,
             top_k=40, top_p=0.9, repetition_penalty=1.1, device="cpu",
             use_cache=True):
    """Generate and return full text string (prompt + generated tokens)."""
    pieces = list(generate_stream(
        model, tokenizer, prompt, max_new_tokens, temperature,
        top_k, top_p, repetition_penalty, device, use_cache,
    ))
    return prompt + "".join(pieces)
