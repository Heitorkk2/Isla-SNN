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

import warnings
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset as HFDataset
from transformers import AutoTokenizer


def get_tokenizer(name="codelion/gpt-2-70m"):
    """Load tokenizer and ensure a pad token exists."""
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _tokenize_batch(examples, tokenizer, max_seq_len, is_finetune=False,
                    response_template="<|im_start|>assistant\n", pack=False):
    """Tokenize for causal LM.

    Examples stay variable-length in both modes; unpacked batches are padded
    to the longest member at collation time, not to max_seq_len here. Packed
    examples additionally receive one EOS boundary token.
    If is_finetune=True, also masks the human prompt with -100, leaving only assistant responses.
    """
    texts = examples.get("text") or examples.get("content")
    if texts is None:
        raise ValueError(f"No text column. Available: {list(examples.keys())}")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    token_limit = max_seq_len - 1 if pack and eos_token_id is not None else max_seq_len
    # Never pad here: fixed-width rows would bake <pad> into the cached dataset
    # and force every batch to max_seq_len, which is quadratically expensive in
    # the sync-attention kernel. _collate pads per batch instead.
    enc = tokenizer(texts, max_length=token_limit, truncation=True,
                    padding=False, return_tensors=None)

    if pack and eos_token_id is not None:
        for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
            if not ids or ids[-1] != eos_token_id:
                ids.append(eos_token_id)
                attn.append(1)

    labels = []
    
    if is_finetune and response_template:
        # Encode the template
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        resp_len = len(response_token_ids)
        n_unmatched = 0

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
            if match_idx == -1:
                n_unmatched += 1

        if n_unmatched:
            # A miss means the prompt was NOT masked and is being trained on as
            # if it were a response. BPE can split the template differently in
            # context, so this must be loud rather than silently degrading.
            warnings.warn(
                f"response_template {response_template!r} not found in "
                f"{n_unmatched}/{len(labels)} examples; their prompts are "
                f"unmasked and will contribute to the loss. Check that the "
                f"template tokenizes identically inside the full text.",
                stacklevel=2,
            )
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
    has_attention_mask = "attention_mask" in dataset.column_names

    for example in dataset:
        ids = list(example["input_ids"])
        labels = list(example["labels"]) if has_labels else ids
        if has_attention_mask:
            keep = [bool(value) for value in example["attention_mask"]]
            ids = [token for token, active in zip(ids, keep) if active]
            labels = [label for label, active in zip(labels, keep) if active]

        all_ids.extend(ids)
        if has_labels:
            all_labels.extend(labels)


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
        if validation_split == 0:
            return dataset
        n_val = max(1, int(len(dataset["train"]) * validation_split))
        split = dataset["train"].train_test_split(test_size=n_val, shuffle=True, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
        print(f"[DATA] Created validation split: {len(dataset['validation']):,} samples")

    # rename 'test' to 'validation' if that's what we have
    if "test" in dataset and "validation" not in dataset:
        dataset["validation"] = dataset["test"]

    return dataset


def load_isla_dataset(data_path, tokenizer, max_seq_len=1024, num_proc=4, pack=True,
                      is_finetune=False, response_template="<|im_start|>assistant\n",
                      validation_split=0.001):
    """Load (and optionally tokenize + cache) a dataset.

    Args:
        data_path: directory, JSONL file, or HuggingFace dataset name
        tokenizer: pre-loaded tokenizer
        max_seq_len: sequence length for chunking/padding
        num_proc: number of parallel tokenization workers
        pack: if True, concatenate and chunk (no padding waste)
        is_finetune: if True, masks the user prompts with -100
        response_template: identifying tag for the assistant block
        validation_split: fraction reserved when the dataset has no validation split
    """
    p = Path(data_path)

    # already tokenized directory (HF save_to_disk format)
    if p.is_dir():
        print(f"[DATA] Loading pre-tokenized dataset: {p}")
        ds = load_from_disk(str(p))
        ds = _ensure_dict_with_split(ds, validation_split)

        # optionally re-pack pre-tokenized data
        if pack and "labels" not in ds["train"].column_names:
            print("[DATA] Re-packing pre-tokenized dataset (no labels column, packing from input_ids)")
            packed = _pack_sequences(ds["train"], max_seq_len)
            ds = _ensure_dict_with_split(packed, validation_split)

        cols = ds["train"].column_names
        has_labels = "labels" in cols
        print(f"[DATA] Columns: {cols} | labels={'yes' if has_labels else 'auto (=input_ids)'}")
        print(f"[DATA] Ready: {len(ds['train']):,} train, "
              f"{len(ds.get('validation', [])):,} val")
        return ds

    # cached version next to source
    # v2 invalidates legacy packed caches that may contain padded input tokens.
    # _tokenized_v2 invalidates legacy caches whose rows were padded to
    # max_seq_len; those would defeat the dynamic padding in _collate.
    cache_suffix = "_packed_v2" if pack else "_tokenized_v2"
    cached = p.parent / f"{p.stem}{cache_suffix}"
    if cached.is_dir():
        print(f"[DATA] Found cache: {cached}")
        ds = load_from_disk(str(cached))
        ds = _ensure_dict_with_split(ds, validation_split)
        return ds

    # need to tokenize
    if p.suffix in (".jsonl", ".json"):
        print(f"[DATA] Tokenizing: {p}")
        raw = load_dataset("json", data_files={"train": str(p)})
    else:
        print(f"[DATA] Downloading: {data_path}")
        raw = load_dataset(data_path)

    fn = partial(_tokenize_batch, tokenizer=tokenizer, max_seq_len=max_seq_len, 
                 is_finetune=is_finetune, response_template=response_template,
                 pack=pack)
    tok_ds = raw.map(fn, batched=True, remove_columns=raw["train"].column_names,
                     num_proc=num_proc, desc="Tokenizing")

    if pack:
        # pack after tokenization: concatenate + chunk for zero-waste training
        packed_train = _pack_sequences(tok_ds["train"], max_seq_len)
        tok_ds = _ensure_dict_with_split(packed_train, validation_split)
    else:
        tok_ds = _ensure_dict_with_split(tok_ds, validation_split)

    tok_ds.save_to_disk(str(cached))
    print(f"[DATA] Cached at: {cached}")
    return tok_ds


def _to_long_tensor(val):
    """Convert any format (list, numpy, Arrow, tensor) to a LongTensor."""
    if isinstance(val, torch.Tensor):
        return val.long()
    return torch.tensor(list(val), dtype=torch.long)


def _collate(batch, pad_id=0, pad_to_multiple_of=64):
    """Pad a batch to its own longest member instead of a global max length.

    Rows already share a length when the dataset was packed, so this is a
    no-op there. For instruction data it is the difference between attending
    over 2048 positions and attending over ~256.
    """
    ids_list = [_to_long_tensor(b["input_ids"]) for b in batch]
    has_labels = "labels" in batch[0]
    labels_list = (
        [_to_long_tensor(b["labels"]) for b in batch] if has_labels
        else [t.clone() for t in ids_list]
    )

    max_len = max(t.numel() for t in ids_list)
    if pad_to_multiple_of > 1:
        # Round up so the attention matmuls stay on tensor-core-friendly shapes.
        max_len = -(-max_len // pad_to_multiple_of) * pad_to_multiple_of

    B = len(ids_list)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, (row_ids, row_labels) in enumerate(zip(ids_list, labels_list)):
        n = row_ids.numel()
        ids[i, :n] = row_ids
        labels[i, :n] = row_labels
        attention_mask[i, :n] = 1

    return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}


class LengthGroupedSampler(Sampler):
    """Yields indices so that each batch holds similarly-sized sequences.

    Dynamic padding only pays off when a batch is homogeneous — one 2048-token
    outlier drags its three short neighbours up with it. Shuffling inside large
    megabatches keeps randomness while bounding the intra-batch length spread.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=42, megabatch_mult=50):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.megabatch_size = batch_size * megabatch_mult

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))

        indices = []
        for start in range(0, n, self.megabatch_size):
            megabatch = order[start:start + self.megabatch_size]
            megabatch.sort(key=lambda i: self.lengths[i], reverse=True)
            indices.extend(megabatch)

        if self.shuffle:
            # Shuffle whole batches so the longest ones are not always first.
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            batches = [indices[i:i + self.batch_size]
                       for i in range(0, len(indices), self.batch_size)]
            perm = torch.randperm(len(batches), generator=g).tolist()
            indices = [i for b in perm for i in batches[b]]

        return iter(indices)


class TokenBudgetBatchSampler(Sampler):
    """Builds batches with a bounded token count instead of a fixed row count.

    Activation memory scales with batch_size × padded_length, so a fixed
    batch_size must be sized for the longest batch and wastes capacity on
    every shorter one. Holding batch_size × L roughly constant instead keeps
    peak VRAM flat: long sequences get few rows per batch, short ones get many.

    This matters more for an SNN than a Transformer — LIF integration stores
    T timesteps of activations, so the peak is T× higher for the same shape.
    """

    def __init__(self, lengths, max_tokens, max_batch_size, shuffle=True,
                 seed=42, megabatch_mult=50, pad_to_multiple_of=64, drop_last=False):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.megabatch_size = max_batch_size * megabatch_mult
        self.pad_to_multiple_of = pad_to_multiple_of
        self.drop_last = drop_last
        self._batches = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = None

    def _padded(self, n):
        m = self.pad_to_multiple_of
        return -(-n // m) * m if m > 1 else n

    def _build(self):
        n = len(self.lengths)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))

        batches, cur, cur_max = [], [], 0
        for start in range(0, n, self.megabatch_size):
            megabatch = order[start:start + self.megabatch_size]
            megabatch.sort(key=lambda i: self.lengths[i], reverse=True)
            for idx in megabatch:
                cand_max = max(cur_max, self._padded(self.lengths[idx]))
                if cur and ((len(cur) + 1) * cand_max > self.max_tokens
                            or len(cur) + 1 > self.max_batch_size):
                    batches.append(cur)
                    cur, cur_max = [idx], self._padded(self.lengths[idx])
                else:
                    cur.append(idx)
                    cur_max = cand_max
        if cur and not self.drop_last:
            batches.append(cur)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        self._batches = batches

    def __iter__(self):
        if self._batches is None:
            self._build()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._build()
        return len(self._batches)


def create_dataloader(dataset_split, batch_size, shuffle=True, num_workers=2,
                      drop_last=True, seed=42, pad_id=0, group_by_length=False,
                      pad_to_multiple_of=64, max_tokens_per_batch=0):
    """Create a DataLoader with dynamic padding and deterministic shuffling.

    group_by_length sorts similar-length sequences into the same batch, which
    cuts padding waste further on instruction datasets. It is ignored for
    packed datasets, where every row already has the same length.

    max_tokens_per_batch, when > 0, switches to token-budget batching: rows per
    batch vary so that batch_size × padded_length stays under the budget, which
    bounds peak VRAM. batch_size then acts only as an upper cap on rows.
    """
    collate = partial(_collate, pad_id=pad_id, pad_to_multiple_of=pad_to_multiple_of)

    lengths = None
    if group_by_length or max_tokens_per_batch > 0:
        lengths = [len(x) for x in dataset_split["input_ids"]]
        if len(set(lengths)) == 1:
            lengths = None  # packed data: every row already the same length

    if lengths is not None and max_tokens_per_batch > 0:
        batch_sampler = TokenBudgetBatchSampler(
            lengths, max_tokens_per_batch, batch_size, shuffle, seed,
            pad_to_multiple_of=pad_to_multiple_of, drop_last=drop_last,
        )
        return DataLoader(dataset_split, batch_sampler=batch_sampler,
                          collate_fn=collate, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=num_workers > 0)

    sampler = None
    if lengths is not None and group_by_length:
        sampler = LengthGroupedSampler(lengths, batch_size, shuffle, seed)

    generator = None
    if shuffle and sampler is None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(dataset_split, batch_size=batch_size,
                      shuffle=shuffle if sampler is None else False,
                      sampler=sampler,
                      collate_fn=collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0,
                      drop_last=drop_last, generator=generator)
