"""Image encoder: VGG16 convolutional features (ImageNet weights, no classifier head)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_FEATURES = Path("artifacts/features.npz")
TARGET_SIZE = (224, 224)


def load_vgg16():
    from keras.applications import VGG16

    return VGG16(include_top=False, weights="imagenet")


def image_feature(model, image_path: Path) -> np.ndarray:
    """Flattened VGG16 features for one image, shape (1, 25088)."""
    from keras.applications.vgg16 import preprocess_input
    from keras.utils import img_to_array, load_img

    image = img_to_array(load_img(image_path, target_size=TARGET_SIZE))
    batch = preprocess_input(image[None])
    return model.predict(batch, verbose=0).reshape(1, -1)


def extract_features(images_dir: Path, out: Path = DEFAULT_FEATURES, batch_size: int = 32) -> Path:
    """Run every image through VGG16 once and cache the features in an .npz file."""
    from keras.applications.vgg16 import preprocess_input
    from keras.utils import img_to_array, load_img

    model = load_vgg16()
    names = sorted(p.name for p in Path(images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Extracting VGG16 features for {len(names)} images...")
    feats = []
    for start in range(0, len(names), batch_size):
        chunk = names[start : start + batch_size]
        batch = np.stack([img_to_array(load_img(Path(images_dir) / n, target_size=TARGET_SIZE)) for n in chunk])
        feats.append(model.predict(preprocess_input(batch), verbose=0).reshape(len(chunk), -1))
        print(f"  {min(start + batch_size, len(names))}/{len(names)}", end="\r")
    print()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array(names), features=np.concatenate(feats).astype(np.float32))
    print(f"Saved {out}")
    return out


def load_features(path: Path = DEFAULT_FEATURES) -> dict[str, np.ndarray]:
    """Mapping image file name -> feature array of shape (1, n_features)."""
    data = np.load(path)
    return {str(n): f[None] for n, f in zip(data["names"], data["features"])}
