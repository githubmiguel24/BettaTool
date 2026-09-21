"""Normalizes a raw Roboflow/CVAT COCO-Keypoints export into this repo's
expected annotation schema (Build Prompt v2 Sec5.1), so
training.validate_annotations / training.splitting / training.dataset can
read it without every sample failing validation on day one.

A plain COCO Keypoints export does NOT close two schema gaps this codebase
requires:

  1. `fish_box` -- this codebase's per-sample bounding box field. Standard
     COCO annotations use `bbox` instead; this script copies it over.

  2. `specimen_id` -- required by training/splitting.py's grouped split (so
     multiple photos of the same physical fish never leak across
     train/val/test -- see that module's docstring). Standard COCO has no
     such concept. Without real grouping info, this script defaults every
     image to its own specimen (specimen_id = image_id). This is SAFE (no
     leakage), it just means grouping has no protective effect until real
     groups are supplied. If several of your photos really are the same
     physical fish, pass --specimen-map, a CSV of `file_name,specimen_id`
     rows, and those override the default.

Note on visibility flags: COCO's v=0/1/2 (not labeled / occluded / visible)
maps directly onto this repo's Visibility.OCCLUDED(0)/AMBIGUOUS(1)/CLEAR(2)
-- no fix needed there. This repo additionally recognizes v=-1
(OUT_OF_FRAME), which a plain COCO export will never contain; those
landmarks will just read as "occluded" instead of a distinct out-of-frame
bucket. Not a crash, just a slightly less informative
training.validate_annotations report -- worth knowing, not worth blocking on.

Usage:
    python -m training.prepare_annotations \\
        --in data/annotations/raw_export.json \\
        --out data/annotations/annotations.json \\
        [--specimen-map data/annotations/specimen_map.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def prepare(raw: dict[str, Any], specimen_map: dict[str, str] | None) -> dict[str, Any]:
    """Mutates and returns `raw`, filling in `fish_box` and `specimen_id`
    on every annotation that is missing them."""
    images_by_id = {img["id"]: img for img in raw["images"]}
    n_defaulted = 0

    for ann in raw["annotations"]:
        if "fish_box" not in ann:
            if "bbox" not in ann:
                raise ValueError(
                    f"annotation {ann.get('id', '?')} (image_id={ann.get('image_id', '?')}) has "
                    "neither 'fish_box' nor 'bbox' -- cannot proceed."
                )
            ann["fish_box"] = ann["bbox"]

        if "specimen_id" not in ann:
            image_meta = images_by_id.get(ann["image_id"])
            file_name = image_meta.get("file_name") if image_meta else None
            mapped = specimen_map.get(file_name) if (specimen_map and file_name) else None
            if mapped is not None:
                ann["specimen_id"] = mapped
            else:
                ann["specimen_id"] = str(ann["image_id"])
                n_defaulted += 1

    if n_defaulted:
        print(
            f"NOTE: {n_defaulted} annotation(s) had no specimen_id and no --specimen-map entry, "
            "so each was defaulted to its own specimen_id (= its image_id). This is SAFE (no "
            "train/val/test leakage) but means grouped splitting has no effect for them -- if "
            "several of your photos are the SAME physical fish, supply --specimen-map or "
            "leakage protection won't apply to those images."
        )

    return raw


def load_specimen_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    with open(path, newline="") as f:
        return {row["file_name"]: row["specimen_id"] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", required=True, help="Raw Roboflow/CVAT COCO Keypoints export.")
    parser.add_argument("--out", dest="out_path", required=True, help="Where to write the normalized annotations.json.")
    parser.add_argument(
        "--specimen-map", default=None,
        help="Optional CSV with columns file_name,specimen_id, for images that are the same physical fish.",
    )
    args = parser.parse_args()

    raw = json.loads(Path(args.in_path).read_text())
    specimen_map = load_specimen_map(args.specimen_map)
    prepared = prepare(raw, specimen_map)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(prepared, indent=2))
    print(f"Wrote {len(prepared['annotations'])} normalized annotations to {args.out_path}")


if __name__ == "__main__":
    main()
