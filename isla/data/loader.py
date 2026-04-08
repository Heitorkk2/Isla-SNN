"""Dataset loading with HuggingFace caching conventions.

Supports three input modes:
    1. Pre-tokenized directory (HF save_to_disk)  → instant load
    2. Raw JSONL/JSON file                         → tokenize, cache, load
    3. HuggingFace dataset name                    → download, tokenize, cache

Pre-tokenized datasets must contain at least an 'input_ids' column.
If 'labels' is missing, input_ids are used as labels during collation.
Both flat Dataset and DatasetDict formats are supported.

When pack_sequences=True (default), texts are concatenated and chunked
into fixed-length blocks with no padding — the standard approach for
language model pre-training.
"""

from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset as HFDataset
from transformers import AutoTokenizer


def get_tokenizer(name="codelion/gpt-2-70m"):
    """Load tokenizer and ensure a pad token exists."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _tokenize_batch(examples, tokenizer, max_seq_len, is_finetune=False, response_template="<|im_start|>assistant\n"):
    """Tokenize for causal LM. Pads to max_seq_len, labels mask padding with -100.
    If is_finetune=True, also masks the human prompt with -100, leaving only assistant responses.
    """
    texts = examples.get("text") or examples.get("content")
    if texts is None:
        raise ValueError(f"No text column. Available: {list(examples.keys())}")

    enc = tokenizer(texts, max_length=max_seq_len, truncation=True,
                    padding="max_length", return_tensors=None)

    labels = []
    
    if is_finetune and response_template:
        # Encode the template
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        resp_len = len(response_token_ids)
        
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            # Find the starting index of the response template in the tokens
            match_idx = -1
            for i in range(len(ids) - resp_len + 1):
                if ids[i:i+resp_len] == response_token_ids:
                    match_idx = i + resp_len  # Compute loss exactly AFTER the template finishes
                    break
            
            # Mask anything before match_idx or any padding
            cur_labels = []
            for i, (tok_id, mask) in enumerate(zip(ids, attn)):
                if mask == 0:
                    cur_labels.append(-100) # Mask padding
                elif match_idx != -1 and i < match_idx:
                    cur_labels.append(-100) # Mask human prompt
                else:
                    cur_labels.append(tok_id)
            labels.append(cur_labels)
    else:
        # Default Causal LM pre-training logic
        labels = [
            [tok_id if mask == 1 else -100 for tok_id, mask in zip(ids, attn)]
            for ids, attn in zip(enc["input_ids"], enc["attention_mask"])
        ]
        
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def _pack_sequences(dataset, max_seq_len):
    """Concatenate all tokens and re-chunk into fixed-length blocks.

    No padding is needed — every token contributes to training. This is
    2-5× more efficient than padding-based tokenization for typical corpora
    where most texts are shorter than max_seq_len.
    """
    all_ids = []
    all_labels = []
    has_labels = "labels" in dataset.column_names

    for example in dataset:
        ids = example["input_ids"]
        all_ids.extend(list(ids) if not isinstance(ids, list) else ids)
        if has_labels:
            lbls = example["labels"]
            all_labels.extend(list(lbls) if not isinstance(lbls, list) else lbls)


    # chunk into blocks of max_seq_len (discard remainder)
    n_chunks = len(all_ids) // max_seq_len
    if n_chunks == 0:
        raise ValueError(
            f"Dataset has {len(all_ids)} tokens, need at least {max_seq_len} for one chunk."
        )

    total = n_chunks * max_seq_len
    all_ids = all_ids[:total]

    chunks_ids = [all_ids[i * max_seq_len:(i + 1) * max_seq_len] for i in range(n_chunks)]
    if has_labels:
        chunks_labels = [all_labels[i * max_seq_len:(i + 1) * max_seq_len] for i in range(n_chunks)]

    return HFDataset.from_dict({
        "input_ids": chunks_ids,
        "labels": chunks_labels if has_labels else chunks_ids,
    })


def _ensure_dict_with_split(dataset, validation_split=0.001):
    """Wrap flat Dataset in DatasetDict and create val split if needed."""

    # flat Dataset → wrap into DatasetDict
    if isinstance(dataset, HFDataset):
        print(f"[DATA] Wrapping flat Dataset ({len(dataset):,} samples) into DatasetDict")
        dataset = DatasetDict({"train": dataset})

    # create validation split if missing
    if "validation" not in dataset and "test" not in dataset:
        n_val = max(1, int(len(dataset["train"]) * validation_split))
        split = dataset["train"].train_test_split(test_size=n_val, shuffle=True, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
        print(f"[DATA] Created validation split: {len(dataset['validation']):,} samples")

    # rename 'test' to 'validation' if that's what we have
    if "test" in dataset and "validation" not in dataset:
        dataset["validation"] = dataset["test"]

    return dataset


def load_isla_dataset(data_path, tokenizer, max_seq_len=1024, num_proc=4, pack=True, 
                      is_finetune=False, response_template="<|im_start|>assistant\n"):
    """Load (and optionally tokenize + cache) a dataset.

    Args:
        data_path: directory, JSONL file, or HuggingFace dataset name
        tokenizer: pre-loaded tokenizer
        max_seq_len: sequence length for chunking/padding
        num_proc: number of parallel tokenization workers
        pack: if True, concatenate and chunk (no padding waste)
        is_finetune: if True, masks the user prompts with -100
        response_template: identifying tag for the assistant block
    """
    p = Path(data_path)

    # already tokenized directory (HF save_to_disk format)
    if p.is_dir():
        print(f"[DATA] Loading pre-tokenized dataset: {p}")
        ds = load_from_disk(str(p))
        ds = _ensure_dict_with_split(ds)

        # optionally re-pack pre-tokenized data
        if pack and "labels" not in ds["train"].column_names:
            print("[DATA] Re-packing pre-tokenized dataset (no labels column, packing from input_ids)")
            packed = _pack_sequences(ds["train"], max_seq_len)
            ds = _ensure_dict_with_split(packed)

        cols = ds["train"].column_names
        has_labels = "labels" in cols
        print(f"[DATA] Columns: {cols} | labels={'yes' if has_labels else 'auto (=input_ids)'}")
        print(f"[DATA] Ready: {len(ds['train']):,} train, "
              f"{len(ds.get('validation', [])):,} val")
        return ds

    # cached version next to source
    cache_suffix = "_packed" if pack else "_tokenized"
    cached = p.parent / f"{p.stem}{cache_suffix}"
    if cached.is_dir():
        print(f"[DATA] Found cache: {cached}")
        ds = load_from_disk(str(cached))
        ds = _ensure_dict_with_split(ds)
        return ds

    # need to tokenize
    if p.suffix in (".jsonl", ".json"):
        print(f"[DATA] Tokenizing: {p}")
        raw = load_dataset("json", data_files={"train": str(p)})
    else:
        print(f"[DATA] Downloading: {data_path}")
        raw = load_dataset(data_path)

    fn = partial(_tokenize_batch, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                 is_finetune=is_finetune, response_template=response_template)
    tok_ds = raw.map(fn, batched=True, remove_columns=raw["train"].column_names,
                     num_proc=num_proc, desc="Tokenizing")

    if pack:
        # pack after tokenization: concatenate + chunk for zero-waste training
        packed_train = _pack_sequences(tok_ds["train"], max_seq_len)
        tok_ds = _ensure_dict_with_split(packed_train)
    else:
        tok_ds = _ensure_dict_with_split(tok_ds)

    tok_ds.save_to_disk(str(cached))
    print(f"[DATA] Cached at: {cached}")
    return tok_ds


def _to_long_tensor(val):
    """Convert any format (list, numpy, Arrow, tensor) to a LongTensor."""
    if isinstance(val, torch.Tensor):
        return val.long()
    return torch.tensor(list(val), dtype=torch.long)


def _collate(batch):
    ids = torch.stack([_to_long_tensor(b["input_ids"]) for b in batch])
    if "labels" in batch[0]:
        labels = torch.stack([_to_long_tensor(b["labels"]) for b in batch])
    else:
        labels = ids.clone()
    return {"input_ids": ids, "labels": labels}


def create_dataloader(dataset_split, batch_size, shuffle=True, num_workers=2,
                      drop_last=True, seed=42):
    """Create a DataLoader with deterministic shuffling."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(dataset_split, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate, num_workers=num_workers,
                      pin_memory=True, drop_last=drop_last, generator=generator)
