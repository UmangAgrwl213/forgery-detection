"""Prepare a balanced pilot from unpacked GenImage generator directories."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for partition in ("train", "validation", "test"):
        parser.add_argument(f"--{partition}", nargs="+", required=True, metavar="GENERATOR")
    parser.add_argument("--per-class", type=int, default=500, help="Images per class and generator")
    args = parser.parse_args()
    groups = {name: getattr(args, name) for name in ("train", "validation", "test")}
    names = [generator.casefold() for values in groups.values() for generator in values]
    if args.per_class < 1 or len(set(names)) != len(names):
        parser.error("Use a positive sample count and distinct generator folders")
    sources = {}
    for part, generators in groups.items():
        for generator in generators:
            if Path(generator).name != generator or generator in (".", ".."):
                parser.error("Generator names must be direct child folders")
            for label in ("nature", "ai"):
                folder = args.source / generator / ("train" if part == "train" else "val") / label
                files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
                if len(files) < args.per_class:
                    parser.error(f"Need at least {args.per_class} candidate images in {folder}")
                sources[part, generator, label] = files

    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(42)
    seen = set()
    manifest = []
    for (part, generator, label), files in sources.items():
        rng.shuffle(files)
        destination = args.output / part / ("real" if label == "nature" else "fake") / generator
        destination.mkdir(parents=True)
        count = 0
        for path in files:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            target = destination / f"{digest}.jpg"
            image.save(target, quality=90)
            manifest.append({"source": path.relative_to(args.source).as_posix(),
                             "output": target.relative_to(args.output).as_posix(), "rgb_sha256": digest})
            count += 1
            if count == args.per_class:
                break
        if count != args.per_class:
            raise ValueError(f"Insufficient unique images in {part}/{generator}/{label}. Retry in a fresh output directory.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} images")


if __name__ == "__main__":
    main()
