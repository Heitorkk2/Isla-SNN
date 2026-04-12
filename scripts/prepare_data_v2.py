"""Prepare trilingual dataset for Isla-SNN v2 training.

Downloads and mixes:
  - 50% English text (CulturaX EN)
  - 35% Portuguese text (CulturaX PT)  
  - 15% Code (Python, JavaScript, TypeScript, Java)

Uses streaming to avoid downloading entire datasets.
Output: ./data/corpus_v2.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset

# Target token counts (rough estimate: 1 token ≈ 4 chars)
CHARS_PER_TOKEN = 6

def estimate_tokens(text):
    """Rough token count estimate."""
    return len(text) // CHARS_PER_TOKEN

def stream_culturax(lang, target_tokens, output_file, label):
    """Stream CulturaX and append documents to output file."""
    print(f"\n[{label}] Streaming CulturaX '{lang}' — target: {target_tokens:,} tokens")
    
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    
    total_tokens = 0
    doc_count = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:  # skip very short docs
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

def stream_code(languages, target_tokens, output_file, label="CODE"):
    """Stream code from bigcode/starcoderdata via direct parquet loading."""
    from huggingface_hub import HfFileSystem
    
    print(f"\n[{label}] Streaming code ({', '.join(languages)}) — target: {target_tokens:,} tokens")
    
    # Map language names to starcoderdata directory names
    lang_map = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "c": "c",
        "cpp": "c++",
        "go": "go",
        "rust": "rust",
    }
    
    tokens_per_lang = target_tokens // len(languages)
    total_tokens = 0
    fs = HfFileSystem()
    
    for lang in languages:
        lang_tokens = 0
        doc_count = 0
        dir_name = lang_map.get(lang.lower(), lang.lower())
        
        # Try to find parquet files for this language
        try:
            pattern = f"datasets/bigcode/starcoderdata/{dir_name}/*.parquet"
            files = fs.glob(pattern)
            if not files:
                raise FileNotFoundError(f"No parquet files found for {dir_name}")
            
            hf_urls = [f"hf://{f}" for f in files[:5]]  # limit files to avoid overload
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
                if len(code) < 50 or len(code) > 50000:  # skip tiny/huge files
                    continue
                
                # Wrap code with language marker for better learning
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


def shuffle_file(filepath):
    """Shuffle lines in a JSONL file (in memory — fine for ~3-5GB)."""
    print(f"\n[SHUFFLE] Reading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"[SHUFFLE] Shuffling {len(lines):,} documents...")
    random.seed(42)
    random.shuffle(lines)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"[SHUFFLE] Done!")


def main():
    parser = argparse.ArgumentParser(description="Prepare trilingual dataset for Isla-SNN v2")
    parser.add_argument("--output", default="./data/corpus_v2.jsonl")
    parser.add_argument("--total-tokens", type=int, default=1_500_000_000,
                        help="Total target tokens (default: 1.5B)")
    parser.add_argument("--en-pct", type=float, default=0.50, help="English percentage")
    parser.add_argument("--pt-pct", type=float, default=0.35, help="Portuguese percentage")
    parser.add_argument("--code-pct", type=float, default=0.15, help="Code percentage")
    parser.add_argument("--code-langs", nargs="+", default=["python", "javascript", "typescript", "java"],
                        help="Programming languages to include")
    parser.add_argument("--no-shuffle", action="store_true", help="Skip shuffling")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file
    output.write_text("")
    
    en_target = int(args.total_tokens * args.en_pct)
    pt_target = int(args.total_tokens * args.pt_pct)
    code_target = int(args.total_tokens * args.code_pct)
    
    print("=" * 60)
    print("Isla-SNN v2 — Dataset Preparation")
    print("=" * 60)
    print(f"Target: {args.total_tokens:,} total tokens")
    print(f"  EN:   {en_target:,} ({args.en_pct*100:.0f}%)")
    print(f"  PT:   {pt_target:,} ({args.pt_pct*100:.0f}%)")
    print(f"  Code: {code_target:,} ({args.code_pct*100:.0f}%) — {args.code_langs}")
    print(f"Output: {output}")
    print("=" * 60)
    
    # Stream each source
    actual_en = stream_culturax("en", en_target, str(output), "EN")
    actual_pt = stream_culturax("pt", pt_target, str(output), "PT")
    actual_code = stream_code(args.code_langs, code_target, str(output), "CODE")
    
    total = actual_en + actual_pt + actual_code
    
    # Shuffle
    if not args.no_shuffle:
        shuffle_file(str(output))
    
    # Stats
    size_mb = output.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("DATASET READY!")
    print("=" * 60)
    print(f"  EN:    {actual_en:,} tokens ({actual_en/total*100:.1f}%)")
    print(f"  PT:    {actual_pt:,} tokens ({actual_pt/total*100:.1f}%)")
    print(f"  Code:  {actual_code:,} tokens ({actual_code/total*100:.1f}%)")
    print(f"  Total: {total:,} tokens")
    print(f"  File:  {output} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
