"""Command-line entry point.

Put Flickr8k in ./Flickr8k/ (Images/ folder + captions.txt), then:

    python main.py extract-features          # VGG16 features for every image (run once)
    python main.py train                     # train the caption model
    python main.py caption --samples 10      # caption random validation images -> outputs/predicted_captions.png
    python main.py caption --images my_photo.jpg other.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from captioning.data import DEFAULT_DATASET
from captioning.features import DEFAULT_FEATURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image captioning with VGG16 + LSTM")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="folder with Images/ and captions.txt")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="cached VGG16 features (.npz)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-features", help="extract and cache VGG16 features for all images")
    p.add_argument("--batch-size", type=int, default=32)

    p = sub.add_parser("train", help="train the caption model")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience")

    p = sub.add_parser("caption", help="generate captions with the trained model")
    p.add_argument("--samples", type=int, default=10, help="number of random validation images")
    p.add_argument("--images", type=Path, nargs="*", help="caption these image files instead")
    p.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "extract-features":
        from captioning.data import images_dir
        from captioning.features import extract_features

        extract_features(images_dir(args.dataset), args.features, args.batch_size)

    elif args.command == "train":
        from captioning.train import train

        train(args.dataset, args.features, args.epochs, args.batch_size, args.patience)

    elif args.command == "caption":
        from captioning import train as t

        if args.images:
            t.caption_files(args.images)
        else:
            t.caption_validation_samples(args.dataset, args.features, args.samples, args.seed)


if __name__ == "__main__":
    main()
