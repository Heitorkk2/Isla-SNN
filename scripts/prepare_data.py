import json
import random
import os
from pathlib import Path

# Important: CulturaX in streaming mode requires specific huggingface configurations sometimes,
# but load_dataset with streaming=True works smoothly for huggingface subsets.
from datasets import load_dataset

def prepare_bilingual_data(target_tokens=1_600_000_000, en_ratio=0.7, out_file="./data/corpus.jsonl"):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("Initializing streams for CulturaX (en, pt)...")
    # Using streaming=True means we don't download the 27 Petabytes of data, just what we need lazily.
    stream_en = iter(load_dataset("uonlp/CulturaX", "en", split="train", streaming=True))
    stream_pt = iter(load_dataset("uonlp/CulturaX", "pt", split="train", streaming=True))
    
    # Approx 4 chars per token for standard Byte-level BPE (GPT-2 tokenizer)
    target_chars = target_tokens * 4
    
    collected_chars = 0
    saved_docs = 0
    en_docs = 0
    pt_docs = 0
    
    print(f"Mixing {en_ratio*100:.0f}% EN / {(1-en_ratio)*100:.0f}% PT into {out_file}")
    print(f"Targeting ~{target_tokens/1e9:.2f}B tokens (~{target_chars/1e9:.2f}B chars)")
    
    with open(out_file, "w", encoding="utf-8") as f:
        while collected_chars < target_chars:
            use_en = random.random() < en_ratio
            try:
                if use_en:
                    item = next(stream_en)
                else:
                    item = next(stream_pt)
            except StopIteration:
                print("Warning: Stream ended unexpectedly.")
                break
                
            text = item.get("text", "")
            # Skip very short texts or meaningless noise
            if len(text) < 100:
                continue
                
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            collected_chars += len(text)
            saved_docs += 1
            
            if use_en:
                en_docs += 1
            else:
                pt_docs += 1
            
            if saved_docs % 20000 == 0:
                pct = (collected_chars / target_chars) * 100
                print(f"[{pct:.1f}%] {collected_chars/1e9:.2f}B chars | {saved_docs:,} docs (EN: {en_docs:,}, PT: {pt_docs:,})")

    print(f"\nDone! Saved {saved_docs:,} mixed documents.")
    print(f"Total size: {collected_chars/1e9:.2f} GB of text.")
    print(f"Ratio achieved -> EN: {en_docs/(saved_docs)*100:.1f}%, PT: {pt_docs/(saved_docs)*100:.1f}%")
    print("\nDataset ready for tokenization. You can now run the training loop!")

if __name__ == "__main__":
    # 1.6 Billion tokens allows some padding/margin for the 1.5B training requirement
    # We want to use the config EN 70% and PT 30%
    prepare_bilingual_data(target_tokens=1_600_000_000, en_ratio=0.70)
