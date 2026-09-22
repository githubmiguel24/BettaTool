"""One-off diagnostic for the two things training.validate_annotations just
flagged on the real dataset: an out-of-bounds keypoint, and images with more
than one annotation (multiple fish in one photo, OR an accidental duplicate
label -- this script can't tell which, a human has to look).

Usage:
    # Just report -- makes no changes:
    python -m training.diagnose_data_issues --ann data/annotations/annotations.json

    # Report AND fix out-of-bounds points by masking them (safe: sets v=-1
    # so the loss ignores that single landmark, rather than guessing a
    # "corrected" coordinate). Writes a NEW file, never overwrites in place:
    python -m training.diagnose_data_issues --ann data/annotations/annotations.json \\
        --fix --out data/annotations/annotations.fixed.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.perception.keypoints import KEYPOINT_SHORT_CODES, NUM_KEYPOINTS, Visibility


def find_out_of_bounds(coco: dict, images_by_id: dict) -> list[dict]:
    """Returns one entry per (annotation, landmark) that is outside its
    image's bounds and NOT already flagged out-of-frame."""
    hits = []
    for ann in coco["annotations"]:
        image_meta = images_by_id.get(str(ann["image_id"]))
        if image_meta is None:
            continue
        width, height = image_meta.get("width"), image_meta.get("height")
        kpts = ann.get("keypoints")
        if not (isinstance(kpts, list) and len(kpts) == NUM_KEYPOINTS * 3 and width and height):
            continue
        for i in range(NUM_KEYPOINTS):
            x, y, v = kpts[3 * i], kpts[3 * i + 1], kpts[3 * i + 2]
            if v != Visibility.OUT_OF_FRAME and not (0 <= x <= width and 0 <= y <= height):
                hits.append({
                    "ann_id": ann.get("id"),
                    "image_id": ann["image_id"],
                    "file_name": image_meta.get("file_name"),
                    "landmark_index": i,
                    "landmark_name": KEYPOINT_SHORT_CODES[i],
                    "x": x, "y": y, "v": v,
                    "width": width, "height": height,
                })
    return hits


def find_duplicate_images(coco: dict, images_by_id: dict) -> dict[str, list[dict]]:
    """Groups annotations by image_id, keeping only image_ids with >1 annotation."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[str(ann["image_id"])].append(ann)

    dupes = {}
    for image_id, anns in by_image.items():
        if len(anns) > 1:
            image_meta = images_by_id.get(image_id, {})
            dupes[image_id] = {
                "file_name": image_meta.get("file_name"),
                "annotations": [
                    {"ann_id": a.get("id"), "fish_box": a.get("fish_box") or a.get("bbox"),
                     "specimen_id": a.get("specimen_id")}
                    for a in anns
                ],
            }
    return dupes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", required=True, help="Path to annotations.json (post prepare_annotations.py).")
    parser.add_argument("--fix", action="store_true", help="Mask out-of-bounds points instead of just reporting.")
    parser.add_argument("--out", default=None, help="Required with --fix: where to write the fixed file.")
    args = parser.parse_args()

    coco = json.loads(Path(args.ann).read_text())
    images_by_id = {str(img["id"]): img for img in coco["images"]}

    oob = find_out_of_bounds(coco, images_by_id)
    print(f"=== Out-of-bounds keypoints: {len(oob)} ===")
    for hit in oob:
        print(
            f"  ann_id={hit['ann_id']} image_id={hit['image_id']} file={hit['file_name']!r} "
            f"landmark={hit['landmark_name']} ({hit['landmark_index']}) "
            f"coord=({hit['x']}, {hit['y']}) v={hit['v']} image_size={hit['width']}x{hit['height']}"
        )

    dupes = find_duplicate_images(coco, images_by_id)
    print(f"\n=== Images with more than one annotation: {len(dupes)} ===")
    print("(Compare fish_box below for each. Near-identical boxes on the same image usually mean an")
    print(" accidental duplicate label in Roboflow -- delete the redundant one there and re-export.")
    print(" Clearly different boxes mean genuinely different fish in one photo -- no action needed,")
    print(" each becomes its own training crop.)")
    for image_id, info in dupes.items():
        print(f"  image_id={image_id} file={info['file_name']!r}")
        for a in info["annotations"]:
            print(f"      ann_id={a['ann_id']} specimen_id={a['specimen_id']} fish_box={a['fish_box']}")

    if args.fix:
        if not args.out:
            raise SystemExit("--fix requires --out (never overwrites the input file).")
        by_ann_id = {ann.get("id"): ann for ann in coco["annotations"]}
        for hit in oob:
            ann = by_ann_id[hit["ann_id"]]
            i = hit["landmark_index"]
            ann["keypoints"][3 * i : 3 * i + 3] = [0, 0, int(Visibility.OUT_OF_FRAME)]
        Path(args.out).write_text(json.dumps(coco, indent=2))
        print(f"\nMasked {len(oob)} out-of-bounds point(s) (set to v=-1) and wrote {args.out}")
        print("Duplicate images were NOT touched -- review those by hand in Roboflow if needed.")


if __name__ == "__main__":
    main()