"""Tests for tokenization, packing, and dataset splitting."""

from datasets import Dataset

from isla.config import DataConfig
from isla.data.loader import _ensure_dict_with_split, _pack_sequences, _tokenize_batch


class TinyTokenizer:
    eos_token_id = 99

    def __call__(self, texts, max_length, truncation, padding, return_tensors):
        sequences = [[ord(char) - 96 for char in text] for text in texts]
        sequences = [sequence[:max_length] for sequence in sequences]

        if padding == "max_length":
            masks = [[1] * len(sequence) + [0] * (max_length - len(sequence))
                     for sequence in sequences]
            sequences = [sequence + [0] * (max_length - len(sequence))
                         for sequence in sequences]
        else:
            masks = [[1] * len(sequence) for sequence in sequences]

        return {"input_ids": sequences, "attention_mask": masks}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) - 96 for char in text]


def test_packed_tokenization_uses_variable_lengths_and_eos_boundaries():
    encoded = _tokenize_batch(
        {"text": ["ab", "c"]}, TinyTokenizer(), max_seq_len=8, pack=True,
    )

    assert encoded["input_ids"] == [[1, 2, 99], [3, 99]]
    assert encoded["attention_mask"] == [[1, 1, 1], [1, 1]]
    assert encoded["labels"] == encoded["input_ids"]


def test_packing_removes_tokens_masked_as_padding():
    dataset = Dataset.from_dict({
        "input_ids": [[1, 2, 0, 0], [3, 0, 0, 0]],
        "attention_mask": [[1, 1, 0, 0], [1, 0, 0, 0]],
        "labels": [[1, 2, -100, -100], [3, -100, -100, -100]],
    })

    packed = _pack_sequences(dataset, max_seq_len=3)

    assert packed[0]["input_ids"] == [1, 2, 3]
    assert packed[0]["labels"] == [1, 2, 3]


def test_validation_split_is_respected():
    dataset = Dataset.from_dict({"input_ids": [[value] for value in range(10)]})

    split = _ensure_dict_with_split(dataset, validation_split=0.2)

    assert len(split["train"]) == 8
    assert len(split["validation"]) == 2


def test_validation_can_be_disabled():
    dataset = Dataset.from_dict({"input_ids": [[value] for value in range(3)]})

    split = _ensure_dict_with_split(dataset, validation_split=0.0)

    assert list(split.keys()) == ["train"]
    DataConfig(validation_split=0.0)
