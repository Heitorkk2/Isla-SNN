"""Prepare trilingual + math dataset for Isla-SNN v3 training.

Mistura:
  - 45% English text      (CulturaX EN)
  - 30% Portuguese text   (CulturaX PT)
  - 15% Code              (StarCoderData via Parquet)
  - 10% Matemática        (OpenWebMath + ORCA Math)

Output: ./data/corpus_v3.jsonl  (~1.5B tokens)
"""

import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset

CHARS_PER_TOKEN = 6

def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN

# ─── English / Portuguese ─────────────────────────────────────────────────────

def stream_culturax(lang, target_tokens, output_file, label):
    print(f"\n[{label}] Streaming CulturaX '{lang}' — target: {target_tokens:,} tokens")
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    total_tokens = 0
    doc_count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            tokens = estimate_tokens(text)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_tokens += tokens
            doc_count += 1
            if doc_count % 5000 == 0:
                pct = total_tokens / target_tokens * 100
                print(f"  [{label}] {doc_count:,} docs | {total_tokens:,} tokens ({pct:.1f}%)")
            if total_tokens >= target_tokens:
                break
    print(f"  [{label}] Done: {doc_count:,} docs, {total_tokens:,} tokens")
    return total_tokens

# ─── Code ─────────────────────────────────────────────────────────────────────

def stream_code(languages, target_tokens, output_file, label="CODE"):
    from huggingface_hub import HfFileSystem
    print(f"\n[{label}] Streaming code ({', '.join(languages)}) — target: {target_tokens:,} tokens")

    lang_map = {
        "python": "python", "javascript": "javascript",
        "typescript": "typescript", "java": "java",
        "c": "c", "cpp": "c++", "go": "go", "rust": "rust",
    }
    tokens_per_lang = target_tokens // len(languages)
    total_tokens = 0
    fs = HfFileSystem()

    for lang in languages:
        lang_tokens = 0
        doc_count = 0
        dir_name = lang_map.get(lang.lower(), lang.lower())
        try:
            pattern = f"datasets/bigcode/starcoderdata/{dir_name}/*.parquet"
            files = fs.glob(pattern)
            if not files:
                raise FileNotFoundError(f"No parquet files found for {dir_name}")
            hf_urls = [f"hf://{f}" for f in files[:5]]
            ds = load_dataset("parquet", data_files={"train": hf_urls},
                              split="train", streaming=True)
            source = f"starcoderdata/{dir_name}"
        except Exception as e:
            print(f"  [WARN] Could not load code for {lang}: {e}")
            continue

        print(f"  [{label}] Loading {lang} from {source}...")
        with open(output_file, "a", encoding="utf-8") as f:
            for doc in ds:
                code = doc.get("content", doc.get("code", ""))
                if len(code) < 50 or len(code) > 50000:
                    continue
                text = f"```{lang.lower()}\n{code}\n```"
                tokens = estimate_tokens(text)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                lang_tokens += tokens
                doc_count += 1
                if doc_count % 2000 == 0:
                    pct = lang_tokens / tokens_per_lang * 100
                    print(f"    {lang}: {doc_count:,} files | {lang_tokens:,} tokens ({pct:.1f}%)")
                if lang_tokens >= tokens_per_lang:
                    break

        total_tokens += lang_tokens
        print(f"    {lang} done: {doc_count:,} files, {lang_tokens:,} tokens")

    print(f"  [{label}] Total code: {total_tokens:,} tokens")
    return total_tokens

# ─── Mathematics ──────────────────────────────────────────────────────────────

def stream_math(target_tokens, output_file, label="MATH"):
    """Stream math data from OpenWebMath + ORCA-Math."""
    print(f"\n[{label}] Streaming math — target: {target_tokens:,} tokens")
    total_tokens = 0

    # Source 1: OpenWebMath (high quality web math, ~14.7B tokens avaIslable)
    half_target = target_tokens // 2
    owm_tokens = 0
    try:
        print(f"  [{label}] Loading open-web-math...")
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for doc in ds:
                text = doc.get("text", "")
                if len(text) < 50:
                    continue
                tokens = estimate_tokens(text)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                owm_tokens += tokens
                if owm_tokens % 5_000_000 < tokens:
                    pct = owm_tokens / half_target * 100
                    print(f"    OpenWebMath: {owm_tokens:,} tokens ({pct:.1f}%)")
                if owm_tokens >= half_target:
                    break
        total_tokens += owm_tokens
        print(f"    OpenWebMath done: {owm_tokens:,} tokens")
    except Exception as e:
        print(f"  [WARN] OpenWebMath failed: {e}")

    # Source 2: ORCA Math (word problems + solutions)
    orca_tokens = 0
    orca_target = target_tokens - total_tokens
    try:
        print(f"  [{label}] Loading microsoft/orca-math-word-problems-200k...")
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for doc in ds:
                question = doc.get("question", "")
                answer = doc.get("answer", "")
                if not question or not answer:
                    continue
                text = f"Question: {question}\nAnswer: {answer}"
                tokens = estimate_tokens(text)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                orca_tokens += tokens
                if orca_tokens >= orca_target:
                    break
        total_tokens += orca_tokens
        print(f"    ORCA-Math done: {orca_tokens:,} tokens")
    except Exception as e:
        print(f"  [WARN] ORCA-Math failed: {e}")

    print(f"  [{label}] Total math: {total_tokens:,} tokens")
    return total_tokens

# ─── Shuffle ──────────────────────────────────────────────────────────────────

def shuffle_file(filepath):
    print(f"\n[SHUFFLE] Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"[SHUFFLE] Shuffling {len(lines):,} documents...")
    random.seed(42)
    random.shuffle(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("[SHUFFLE] Done!")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Isla-SNN v3")
    parser.add_argument("--output", default="./data/corpus_v3.jsonl")
    parser.add_argument("--total-tokens", type=int, default=1_500_000_000)
    parser.add_argument("--en-pct",   type=float, default=0.45)
    parser.add_argument("--pt-pct",   type=float, default=0.30)
    parser.add_argument("--code-pct", type=float, default=0.15)
    parser.add_argument("--math-pct", type=float, default=0.10)
    parser.add_argument("--code-langs", nargs="+",
                        default=["python", "javascript", "typescript", "java"])
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")

    en_target   = int(args.total_tokens * args.en_pct)
    pt_target   = int(args.total_tokens * args.pt_pct)
    code_target = int(args.total_tokens * args.code_pct)
    math_target = int(args.total_tokens * args.math_pct)

    print("=" * 60)
    print("Isla-SNN v3 — Dataset Preparation")
    print("=" * 60)
    print(f"Target: {args.total_tokens:,} total tokens")
    print(f"  EN:   {en_target:,}   ({args.en_pct*100:.0f}%)")
    print(f"  PT:   {pt_target:,}   ({args.pt_pct*100:.0f}%)")
    print(f"  Code: {code_target:,} ({args.code_pct*100:.0f}%) — {args.code_langs}")
    print(f"  Math: {math_target:,} ({args.math_pct*100:.0f}%)")
    print(f"Output: {output}")
    print("=" * 60)

    actual_en   = stream_culturax("en", en_target, str(output), "EN")
    actual_pt   = stream_culturax("pt", pt_target, str(output), "PT")
    actual_code = stream_code(args.code_langs, code_target, str(output), "CODE")
    actual_math = stream_math(math_target, str(output), "MATH")

    total = actual_en + actual_pt + actual_code + actual_math

    if not args.no_shuffle:
        shuffle_file(str(output))

    size_mb = output.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("DATASET READY!")
    print("=" * 60)
    print(f"  EN:    {actual_en:,}   tokens ({actual_en/total*100:.1f}%)")
    print(f"  PT:    {actual_pt:,}   tokens ({actual_pt/total*100:.1f}%)")
    print(f"  Code:  {actual_code:,} tokens ({actual_code/total*100:.1f}%)")
    print(f"  Math:  {actual_math:,} tokens ({actual_math/total*100:.1f}%)")
    print(f"  Total: {total:,} tokens")
    print(f"  File:  {output} ({size_mb:.1f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
