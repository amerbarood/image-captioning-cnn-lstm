"""Flickr8k captions: loading, cleaning, tokenising and the train/validation split."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET = Path("Flickr8k")
START, END = "startseq", "endseq"


def images_dir(dataset: Path) -> Path:
    return Path(dataset) / "Images"


def load_captions(dataset: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Read captions.txt (columns: image, caption)."""
    return pd.read_csv(Path(dataset) / "captions.txt")


def clean_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)  # keep letters only
    words = [w for w in text.split() if len(w) > 1]  # drop single letters such as "a"
    return f"{START} {' '.join(words)} {END}"


def preprocess(captions: pd.DataFrame) -> pd.DataFrame:
    df = captions.copy()
    df["caption"] = df["caption"].astype(str).map(clean_caption)
    df["word_count"] = df["caption"].str.split().str.len()
    return df


def split_by_image(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that all captions of an image end up in the same subset."""
    images = sorted(df["image"].unique())
    train_images, _ = train_test_split(images, test_size=val_fraction, shuffle=True, random_state=seed)
    in_train = df["image"].isin(set(train_images))
    return df[in_train].reset_index(drop=True), df[~in_train].reset_index(drop=True)


class Tokenizer:
    """Word-level tokenizer: index 0 is padding, words are numbered by frequency
    (the same convention as Keras' Tokenizer)."""

    def __init__(self, word_index: dict[str, int] | None = None):
        self.word_index = word_index or {}
        self.index_word = {i: w for w, i in self.word_index.items()}

    def fit(self, texts: list[str]) -> "Tokenizer":
        counts = Counter(w for t in texts for w in t.split())
        ordered = sorted(counts, key=lambda w: (-counts[w], w))
        self.word_index = {w: i + 1 for i, w in enumerate(ordered)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.word_index) + 1

    def encode(self, text: str) -> list[int]:
        return [self.word_index[w] for w in text.split() if w in self.word_index]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.word_index))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text()))


def strip_tokens(caption: str) -> str:
    words = [w for w in caption.split() if w not in (START, END)]
    return " ".join(words)
