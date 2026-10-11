"""Dataset download, text cleaning, vocabulary, and sequence preparation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


DATASET_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
UNKNOWN_TOKEN = "<unk>"

_START_MARKER = re.compile(
    r"^\*{3}\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_END_MARKER = re.compile(
    r"^\*{3}\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def download_dataset(
    destination: Path, timeout: int = 30, overwrite: bool = False
) -> Path:
    """Download the verified Project Gutenberg text and save it as UTF-8."""
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists at {destination}; pass --overwrite to replace it."
        )
    request = Request(
        DATASET_URL,
        headers={"User-Agent": "LSTM-Text-Generation/1.0 (educational project)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not download Shakespeare from {DATASET_URL}: {exc}") from exc

    if not text.strip():
        raise RuntimeError(f"The download from {DATASET_URL} was empty.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not save the dataset to {destination}: {exc}") from exc
    return destination


def remove_gutenberg_boilerplate(text: str) -> str:
    """Remove the Gutenberg license/header and footer when their markers exist."""
    start_match = _START_MARKER.search(text)
    if start_match:
        text = text[start_match.end() :]

    end_match = _END_MARKER.search(text)
    if end_match:
        text = text[: end_match.start()]
    return text


def clean_text(text: str) -> str:
    """Lowercase text and replace punctuation with spaces before tokenization."""
    text = remove_gutenberg_boilerplate(text).lower()
    return "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )


def tokenize_text(text: str) -> list[str]:
    """Split cleaned text into word tokens."""
    return clean_text(text).split()


def load_tokens(path: Path) -> list[str]:
    """Read and preprocess a UTF-8 dataset, raising a clear error if it is empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read dataset at {path}: {exc}") from exc

    tokens = tokenize_text(text)
    if not tokens:
        raise ValueError(f"No words found in dataset at {path}.")
    return tokens


class Vocabulary:
    """A compact, serializable word-to-ID mapping with an unknown-word token."""

    def __init__(self, word_to_id: dict[str, int]) -> None:
        self.word_to_id = word_to_id
        self.id_to_word = {token_id: word for word, token_id in word_to_id.items()}
        if UNKNOWN_TOKEN not in word_to_id or word_to_id[UNKNOWN_TOKEN] != 0:
            raise ValueError(f"Vocabulary must reserve ID 0 for {UNKNOWN_TOKEN}.")

    @classmethod
    def from_tokens(
        cls, tokens: Iterable[str], max_vocabulary_size: int | None = None
    ) -> "Vocabulary":
        """Build a vocabulary from training-only tokens, optionally keeping common words."""
        counts = Counter(token for token in tokens if token != UNKNOWN_TOKEN)
        words = sorted(counts, key=lambda word: (-counts[word], word))
        if max_vocabulary_size is not None:
            if max_vocabulary_size < 2:
                raise ValueError("Maximum vocabulary size must be at least 2.")
            words = words[: max_vocabulary_size - 1]
        word_to_id = {UNKNOWN_TOKEN: 0}
        word_to_id.update(
            {word: index for index, word in enumerate(sorted(words), start=1)}
        )
        if len(word_to_id) < 2:
            raise ValueError("Training text must contain at least one word.")
        return cls(word_to_id)

    @property
    def size(self) -> int:
        return len(self.word_to_id)

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        """Map words to IDs, using the unknown-token ID for unseen words."""
        return np.fromiter(
            (self.word_to_id.get(word, 0) for word in tokens), dtype=np.int32
        )

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        """Map IDs to words, replacing invalid IDs with the unknown token."""
        return [self.id_to_word.get(int(token_id), UNKNOWN_TOKEN) for token_id in token_ids]

    def save(self, path: Path) -> None:
        """Write the vocabulary mapping to a JSON artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"word_to_id": self.word_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load and validate a vocabulary JSON artifact."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            word_to_id = payload["word_to_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Could not load vocabulary at {path}: {exc}") from exc
        if not isinstance(word_to_id, dict) or any(
            not isinstance(word, str) or type(token_id) is not int
            for word, token_id in word_to_id.items()
        ):
            raise ValueError(f"Vocabulary at {path} has an invalid mapping.")
        if sorted(word_to_id.values()) != list(range(len(word_to_id))):
            raise ValueError(f"Vocabulary at {path} must use contiguous token IDs.")
        return cls(word_to_id)


def make_sequences(
    token_ids: np.ndarray,
    sequence_length: int,
    target_start: int,
    target_end: int,
    max_sequences: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length contexts and their next-token targets.

    Target indices use the half-open range [target_start, target_end); each
    input contains only the sequence_length tokens immediately before its target.
    """
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")
    if target_start < sequence_length or target_end > len(token_ids):
        raise ValueError("Target range does not have enough preceding context.")
    if target_end <= target_start:
        raise ValueError("Target range is empty.")
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("Maximum sequence count must be at least 1.")

    target_indices = np.arange(target_start, target_end, dtype=np.int64)
    if max_sequences is not None and len(target_indices) > max_sequences:
        positions = np.linspace(
            0, len(target_indices) - 1, num=max_sequences, dtype=np.int64
        )
        target_indices = target_indices[positions]

    inputs = np.stack(
        [token_ids[index - sequence_length : index] for index in target_indices]
    ).astype(np.int32, copy=False)
    targets = token_ids[target_indices].astype(np.int32, copy=False)
    return inputs, targets


def prepare_train_validation(
    tokens: list[str],
    sequence_length: int,
    validation_fraction: float,
    max_train_sequences: int | None,
    max_validation_sequences: int | None,
    max_vocabulary_size: int | None = 10000,
) -> tuple[Vocabulary, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically, fit vocabulary on training text, and make samples."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")

    split_index = int(len(tokens) * (1 - validation_fraction))
    if split_index <= sequence_length or split_index >= len(tokens):
        raise ValueError(
            "Dataset is too short for the selected sequence length and validation fraction."
        )

    vocabulary = Vocabulary.from_tokens(
        tokens[:split_index], max_vocabulary_size=max_vocabulary_size
    )
    token_ids = vocabulary.encode(tokens)
    train_inputs, train_targets = make_sequences(
        token_ids,
        sequence_length,
        sequence_length,
        split_index,
        max_train_sequences,
    )
    validation_inputs, validation_targets = make_sequences(
        token_ids,
        sequence_length,
        split_index,
        len(token_ids),
        max_validation_sequences,
    )
    if vocabulary.size < 2 or not len(train_targets) or not len(validation_targets):
        raise ValueError("Not enough vocabulary or sequences to train and validate.")
    return (
        vocabulary,
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
    )
