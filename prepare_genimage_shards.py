"""Prepare balanced, generator-disjoint samples from GenImage Parquet shards."""

import argparse
import hashlib
import json
import random
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=128)
    for part in ("train", "validation", "test"):
        parser.add_argument(f"--{part}", nargs="+", required=True)
    args = parser.parse_args()
    protocol = {part: getattr(args, part) for part in ("train", "validation", "test")}
    names = [name for groups in protocol.values() for name in groups]
    if (args.per_class < 1 or len({name.casefold() for name in names}) != len(names)
            or any(Path(name).name != name or name.lower() in {"nature", ".", ".."} for name in names)):
        parser.error("Use distinct generator folder names and a positive per-class count")
    plans = []
    rng = random.Random(42)
    for part, generators in protocol.items():
        prefix = "val" if part == "validation" else part
        paths = sorted(args.source.glob(f"{prefix}-*.parquet"))
        candidates = {name: [] for name in ["nature", *generators]}
        for path in paths:
            for index, name in enumerate(pq.read_table(path, columns=["generator"]).column(0).to_pylist()):
                if name in candidates:
                    candidates[name].append((path, index))
        picked = {}
        for index, (name, records) in enumerate(candidates.items()):
            count = args.per_class
            if name != "nature":
                count, remainder = divmod(count, len(generators))
                count += index <= remainder
            if not count or len(records) < count:
                parser.error(f"{part}/{name} needs {count} examples, found {len(records)}")
            for key in rng.sample(records, count):
                picked[key] = name
        plans.append((part, paths, picked))
    args.output.mkdir(parents=True, exist_ok=False)
    seen = set()
    records = []
    for part, paths, picked in plans:
        for path in paths:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image", "image_path"]):
                for row in batch.to_pylist():
                    generator = picked.get((path, index))
                    index += 1
                    if generator is None:
                        continue
                    with Image.open(BytesIO(row["image"]["bytes"])) as source:
                        image = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"),
                                             (256, 256), Image.Resampling.LANCZOS)
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in seen:
                        raise ValueError(f"Duplicate normalized image: {row['image_path']}")
                    seen.add(digest)
                    folder = args.output / part / ("real" if generator == "nature" else f"fake/{generator}")
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / f"{digest}.jpg"
                    image.save(target, quality=90)
                    records.append({"shard": path.name, "source": row["image_path"],
                                    "output": target.relative_to(args.output).as_posix(), "rgb_sha256": digest})
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} images")


if __name__ == "__main__":
    main()
