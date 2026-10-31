"""Fast checks with tiny synthetic data; no dataset or VGG16 download needed."""

import numpy as np
import pandas as pd

from captioning import data
from captioning.model import CaptionSequence, build_model, greedy_caption


def test_cleaning_and_tokenizer():
    assert data.clean_caption("A dog runs, in 2 fields!") == "startseq dog runs in fields endseq"
    tok = data.Tokenizer().fit(["startseq dog runs endseq", "startseq dog sits endseq"])
    assert tok.word_index["dog"] < tok.word_index["runs"]  # more frequent -> smaller index
    assert tok.vocab_size == 6


def test_split_keeps_images_together():
    df = pd.DataFrame({"image": [f"{i // 5}.jpg" for i in range(50)], "caption": ["x"] * 50})
    train, val = data.split_by_image(df)
    assert not set(train["image"]) & set(val["image"])


def test_generator_model_and_decoding():
    captions = data.preprocess(pd.DataFrame({
        "image": ["a.jpg", "a.jpg", "b.jpg", "b.jpg"],
        "caption": ["A dog runs", "Dog running fast", "A cat sleeps", "Cat on sofa"],
    }))
    tok = data.Tokenizer().fit(captions["caption"].tolist())
    max_len = int(captions["word_count"].max())
    features = {"a.jpg": np.random.rand(1, 16).astype("float32"), "b.jpg": np.random.rand(1, 16).astype("float32")}
    seq = CaptionSequence(captions, tok, max_len, features, batch_size=2)
    (x1, x2), y = seq[0]
    assert x1.shape[1] == 16 and x2.shape[1] == max_len and len(y) == len(x1)

    model = build_model(16, tok.vocab_size, max_len, units=8)
    model.fit(seq, epochs=1, verbose=0)
    caption = greedy_caption(model, features["a.jpg"], tok, max_len)
    assert caption.startswith("startseq")
