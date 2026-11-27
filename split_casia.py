"""Freeze a CASIA split grouped by filename-encoded source IDs."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from data.casia import CASIADataset


def build_split(dataset, authentic_dir, seed):
    parents = {}

    def root(source):
        while parents[source] != source:
            source = parents[source]
        return source

    records = []
    unresolved = []
    for image, mask in dataset.samples:
        sources = image.stem.lower().split("_")[-3:-1]
        if len(sources) != 2 or not all(re.fullmatch(r"[a-z]{3}\d{5}", source) for source in sources):
            unresolved.append(image.name)
            continue
        for source in sources:
            parents.setdefault(source, source)
        parents[root(sources[1])] = root(sources[0])
        records.append(dict(image=f"Tp/{image.name}", mask=mask.name, sources=sorted(set(sources))))

    for image in sorted(authentic_dir.iterdir()):
        if image.suffix.lower() not in dataset.extensions:
            continue
        match = re.fullmatch(r"au_([a-z]{3})_(\d{5})", image.stem.lower())
        if not match:
            unresolved.append(image.name)
            continue
        source = "".join(match.groups())
        parents.setdefault(source, source)
        records.append(dict(image=f"Au/{image.name}", mask=None, sources=[source]))

    groups = defaultdict(list)
    for record in records:
        groups[root(record["sources"][0])].append(record)
    components = list(groups.values())
    random.Random(seed).shuffle(components)
    components.sort(key=lambda group: sum(row["mask"] is not None for row in group), reverse=True)
    splits = {name: [] for name in ("train", "validation", "test")}
    fractions = dict(train=0.70, validation=0.15, test=0.15)
    totals = [sum(row["mask"] is not None for row in records), sum(row["mask"] is None for row in records)]
    counts = {name: [0, 0] for name in splits}
    for group in components:
        positive = sum(row["mask"] is not None for row in group)
        sizes = (positive, len(group) - positive)
        name = min(splits, key=lambda part: sum(
            size * (2 * (counts[part][i] - fractions[part] * totals[i]) + size) / max(1, totals[i]) ** 2
            for i, size in enumerate(sizes)
        ))
        splits[name].extend(group)
        for i, size in enumerate(sizes):
            counts[name][i] += size

    if any(min(count) == 0 for count in counts.values()):
        raise ValueError(f"Source grouping could not produce both classes in every split: {counts}")
    return dict(seed=seed, counts=counts, splits=splits, unresolved=unresolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/CASIA2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = CASIADataset(args.root / "Tp", args.root / "CASIA 2 Groundtruth")
    manifest = build_split(dataset, args.root / "Au", args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print("Counts [tampered, authentic]", manifest["counts"])
    print("Unresolved filenames", len(manifest["unresolved"]))


if __name__ == "__main__":
    main()
