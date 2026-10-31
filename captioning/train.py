"""Training, evaluation figures and caption generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: E402

from . import data  # noqa: E402
from .features import DEFAULT_FEATURES, image_feature, load_features, load_vgg16  # noqa: E402
from .model import CaptionSequence, build_model, greedy_caption  # noqa: E402

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "caption_model.keras"
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
META_PATH = ARTIFACTS / "meta.json"
OUTPUT_DIR = Path("outputs")


def train(dataset: Path = data.DEFAULT_DATASET, features_path: Path = DEFAULT_FEATURES, epochs: int = 25,
          batch_size: int = 64, patience: int = 5):
    captions = data.preprocess(data.load_captions(dataset))
    tokenizer = data.Tokenizer().fit(captions["caption"].tolist())
    max_length = int(captions["word_count"].max())
    print(f"Vocabulary size = {tokenizer.vocab_size}, max caption length = {max_length} words")

    features = load_features(features_path)
    n_features = next(iter(features.values())).shape[1]
    train_df, val_df = data.split_by_image(captions)
    print(f"Train captions: {len(train_df)}  Validation captions: {len(val_df)}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    META_PATH.write_text(json.dumps({"max_length": max_length, "val_images": sorted(val_df["image"].unique())}))

    model = build_model(n_features, tokenizer.vocab_size, max_length)
    model.summary()
    history = model.fit(
        CaptionSequence(train_df, tokenizer, max_length, features, batch_size),
        validation_data=CaptionSequence(val_df, tokenizer, max_length, features, batch_size, shuffle=False),
        epochs=epochs,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", min_delta=0, patience=patience, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1, factor=0.2, min_lr=1e-8),
        ],
    )
    pd.DataFrame(history.history).to_csv(ARTIFACTS / "history.csv", index_label="epoch")
    plot_history(history.history)
    return model, history


def plot_history(history: dict) -> Path:
    fig = plt.figure(figsize=(12, 5))
    plt.plot(history["loss"], label="training")
    plt.plot(history["val_loss"], label="validation")
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper right")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "training_loss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def load_trained():
    from keras.models import load_model

    model = load_model(MODEL_PATH)
    tokenizer = data.Tokenizer.load(TOKENIZER_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, tokenizer, meta


def caption_validation_samples(dataset: Path = data.DEFAULT_DATASET, features_path: Path = DEFAULT_FEATURES,
                               n: int = 10, seed: int | None = None) -> Path:
    """Caption n random validation images and save them as a grid."""
    model, tokenizer, meta = load_trained()
    features = load_features(features_path)
    rng = random.Random(seed)
    images = rng.sample([i for i in meta["val_images"] if i in features], n)
    results = []
    for name in images:
        caption = data.strip_tokens(greedy_caption(model, features[name], tokenizer, meta["max_length"]))
        print(f"{name}: {caption}")
        results.append((name, caption))
    return plot_captions(data.images_dir(dataset), results)


def caption_files(paths: list[Path]) -> list[tuple[str, str]]:
    """Caption arbitrary image files (features are computed on the fly)."""
    model, tokenizer, meta = load_trained()
    vgg = load_vgg16()
    results = []
    for path in paths:
        caption = data.strip_tokens(greedy_caption(model, image_feature(vgg, path), tokenizer, meta["max_length"]))
        print(f"{path}: {caption}")
        results.append((str(path), caption))
    return results


def plot_captions(images_dir: Path, results: list[tuple[str, str]], cols: int = 5) -> Path:
    rows = (len(results) + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4.5 * rows))
    for i, (name, caption) in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(plt.imread(Path(images_dir) / name))
        ax.set_title("\n".join(wrap(caption, 25)))
        ax.axis("off")
    fig.subplots_adjust(hspace=0.7, wspace=0.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "predicted_captions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path
