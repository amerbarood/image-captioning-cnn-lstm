"""Merge-architecture caption model, training data generator and greedy decoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from keras import layers, models
from keras.utils import PyDataset, pad_sequences

from .data import END, START, Tokenizer


def build_model(n_image_features: int, vocab_size: int, max_length: int, units: int = 256) -> models.Model:
    """image -> dropout -> dense(256);  caption -> embedding -> dropout -> LSTM(256);
    concatenate -> dense(256) -> softmax over the vocabulary."""
    image_in = layers.Input(shape=(n_image_features,), name="image_features")
    fe = layers.Dense(units, activation="relu")(layers.Dropout(0.5)(image_in))

    caption_in = layers.Input(shape=(max_length,), name="caption_so_far")
    se = layers.Embedding(vocab_size, units, mask_zero=True)(caption_in)
    se = layers.LSTM(units)(layers.Dropout(0.5)(se))

    x = layers.Dense(units, activation="relu")(layers.concatenate([fe, se]))
    out = layers.Dense(vocab_size, activation="softmax")(x)

    model = models.Model(inputs=[image_in, caption_in], outputs=out)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


class CaptionSequence(PyDataset):
    """Builds (image features, partial caption) -> next word pairs on the fly.

    Each caption of n tokens yields n-1 training pairs. Targets are word indices
    (sparse labels), equivalent to the notebook's one-hot targets but far smaller.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, max_length: int, features: dict,
                 batch_size: int = 64, shuffle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.df = df[df["image"].isin(features)].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.df) // self.batch_size

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index: int):
        batch = self.df.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        X1, X2, y = [], [], []
        for image, caption in zip(batch["image"], batch["caption"]):
            feature = self.features[image][0]
            seq = self.tokenizer.encode(caption)
            for i in range(1, len(seq)):
                X1.append(feature)
                X2.append(seq[:i])
                y.append(seq[i])
        X2 = pad_sequences(X2, maxlen=self.max_length)
        return (np.array(X1), X2), np.array(y)


def greedy_caption(model, feature: np.ndarray, tokenizer: Tokenizer, max_length: int) -> str:
    """Generate a caption word by word, always taking the most likely next word."""
    text = START
    for _ in range(max_length):
        seq = pad_sequences([tokenizer.encode(text)], maxlen=max_length)
        word = tokenizer.index_word.get(int(np.argmax(model.predict([feature, seq], verbose=0))))
        if word is None:
            break
        text += " " + word
        if word == END:
            break
    return text
