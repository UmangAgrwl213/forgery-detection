"""Derive a dimension-checked manifest without reassigning any source groups."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=Path("dataset/CASIA2"))
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    excluded = []
    for name, records in manifest["splits"].items():
        kept = []
        for record in records:
            if record["mask"] is not None:
                with Image.open(args.root / record["image"]) as image, Image.open(
                    args.root / "CASIA 2 Groundtruth" / record["mask"]
                ) as mask:
                    if image.size != mask.size:
                        excluded.append({"partition": name, "image": record["image"],
                                         "image_size": image.size, "mask_size": mask.size})
                        continue
            kept.append(record)
        manifest["splits"][name] = kept
        positives = sum(record["mask"] is not None for record in kept)
        manifest["counts"][name] = [positives, len(kept) - positives]
    manifest["parent_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["excluded_dimension_mismatches"] = excluded
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(manifest["counts"], f"Excluded {len(excluded)} mismatches")


if __name__ == "__main__":
    main()
