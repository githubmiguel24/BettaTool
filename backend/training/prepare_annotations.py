"""Normalizes a raw Roboflow COCO-Keypoints export into this repo's expected
annotation schema (Build Prompt v2 Sec5.1 / Sec5.2), so
training.validate_annotations / training.splitting / training.dataset can
read it without every sample failing validation on day one.

A raw Roboflow export does NOT close FOUR schema gaps this codebase requires:

  1. `fish_box` -- this codebase's per-sample bounding box field. Roboflow's
     COCO export uses `bbox` instead; this script copies it over.

  2. `specimen_id` -- required by training/splitting.py's grouped split (so
     multiple photos of the same physical fish never leak across
     train/val/test -- see that module's docstring). Roboflow has no such
     concept. Without real grouping info, this script defaults every image
     to its own specimen (specimen_id = image_id). This is SAFE (no
     leakage), it just means grouping has no protective effect until real
     groups are supplied. If several of your photos really are the same
     physical fish, pass --specimen-map, a CSV of `file_name,specimen_id`
     rows, and those override the default.

  3. `source` / `color_morph` / `has_occlusion` -- the three fields
     training/configs/base.yaml's `dataset.stratify_by` names.
     `has_occlusion` is derived automatically per sample (True if any
     landmark's visibility != CLEAR, i.e. any keypoint is occluded,
     ambiguous, or out-of-frame). `source` and `color_morph` have no
     automatic answer -- Roboflow doesn't track either -- so they default to
     constant placeholders (`--source`, default "roboflow") unless you pass
     `--color-morph-map`, a CSV of `file_name,color_morph` rows. A constant
     stratify field just means that field contributes nothing to balancing
     the split (every sample lands in the same stratum for it); it does NOT
     break the split, and note it in the thesis methodology as a limitation
     if you don't fill it in.

  4. KEYPOINT ORDER. Roboflow's `categories[].keypoints` list is NOT in the
     same order this codebase uses (see app/perception/keypoints.py /
     training/configs/keypoints.yaml) -- indices 4 and 5 are swapped
     (Roboflow: caudal_peduncle_top, dorsal_fin_tip; system: dorsal_tip,
     peduncle_top). Every annotation's `keypoints` array is permuted into
     system order via ROBOFLOW_TO_SYSTEM_NAME below. This is the single
     most dangerous gap to skip: get it wrong and the model trains on two
     landmarks silently swapped, with no error anywhere -- the shapes and
     types all still check out, only the anatomy is wrong. The permutation
     is verified against the export's OWN `categories[].keypoints` list
     every run (not just trusted from memory), and raises loudly if the
     export's keypoint names/order/count ever change.

Note on visibility flags: Roboflow's "Visible" click -> COCO v=2, "Occluded"
click (position still recorded) -> COCO v=1, and a landmark you never
clicked at all (out-of-frame) -> COCO v=0 with x=0, y=0 (COCO's "not
labeled" default). That maps directly onto this repo's
Visibility.CLEAR(2)/AMBIGUOUS(1)/OCCLUDED(0) -- no fix needed. The repo also
recognizes v=-1 (OUT_OF_FRAME) as a distinct flag, but functionally OCCLUDED
(0) and OUT_OF_FRAME (-1) are masked identically in the loss
(training/losses/gaussian_nll.py's visibility_mask), so leaving true
out-of-frame points as the COCO-default v=0 (rather than -1) changes
nothing about training -- it's a labeling nicety, not a bug.

Usage:
    python -m training.prepare_annotations \\
        --in data/annotations/raw_roboflow_export.json \\
        --out data/annotations/annotations.json \\
        [--specimen-map data/annotations/specimen_map.csv] \\
        [--color-morph-map data/annotations/color_morph_map.csv] \\
        [--source roboflow]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.perception.keypoints import KEYPOINT_SHORT_CODES, NUM_KEYPOINTS, Visibility

# Roboflow's keypoint names -> this codebase's short codes (see
# app/perception/keypoints.py's KEYPOINT_SHORT_CODES / keypoints.yaml).
# Order here doesn't matter -- it's a name->name lookup, not a positional
# list -- but every one of Roboflow's 13 names must appear as a key.
ROBOFLOW_TO_SYSTEM_NAME: dict[str, str] = {
    "snout_tip": "snout_tip",
    "eye_center": "eye_center",
    "dorsal_fin_base_anterior": "dorsal_base_ant",
    "dorsal_fin_base_posterior": "dorsal_base_post",
    "caudal_peduncle_top": "peduncle_top",
    "dorsal_fin_tip": "dorsal_tip",
    "caudal_peduncle_bottom": "peduncle_bottom",
    "caudal_fin_tip_upper": "caudal_tip_upper",
    "caudal_fin_tip_lower": "caudal_tip_lower",
    "caudal_fin_center": "caudal_center",
    "anal_fin_base_anterior": "anal_base_ant",
    "anal_fin_base_posterior": "anal_base_post",
    "anal_fin_tip": "anal_tip",
}


def get_roboflow_keypoint_names(raw: dict[str, Any]) -> list[str]:
    """Finds the category entry that declares a `keypoints` list and returns it."""
    for cat in raw.get("categories", []):
        if "keypoints" in cat:
            return cat["keypoints"]
    raise ValueError("No category with a 'keypoints' list found in the export's categories[].")


def build_permutation(roboflow_names: list[str]) -> list[int]:
    """Returns `perm` such that `system_kpts[i] = roboflow_kpts[perm[i]]`.

    Raises ValueError loudly (never silently) if the export's declared
    keypoint names/count don't exactly match what ROBOFLOW_TO_SYSTEM_NAME
    expects -- mis-permuting 13 landmarks per image with no error is exactly
    the failure mode this function exists to prevent.
    """
    if len(roboflow_names) != NUM_KEYPOINTS:
        raise ValueError(
            f"Export declares {len(roboflow_names)} keypoints, expected {NUM_KEYPOINTS}. "
            f"Export's list: {roboflow_names}"
        )
    try:
        system_names_in_roboflow_order = [ROBOFLOW_TO_SYSTEM_NAME[name] for name in roboflow_names]
    except KeyError as e:
        raise ValueError(
            f"Roboflow keypoint name {e} is not in ROBOFLOW_TO_SYSTEM_NAME (training/prepare_annotations.py) "
            "-- update that map before proceeding. Do NOT skip this: it will silently corrupt every "
            "landmark's identity, with no error anywhere downstream."
        ) from None

    perm = [system_names_in_roboflow_order.index(short_code) for short_code in KEYPOINT_SHORT_CODES]

    remapped = [
        f"system[{i}]={KEYPOINT_SHORT_CODES[i]} <- roboflow[{p}]={roboflow_names[p]}"
        for i, p in enumerate(perm)
        if i != p
    ]
    if remapped:
        print("Keypoint order differs from Roboflow's export -- remapping:")
        for line in remapped:
            print(f"  {line}")
    else:
        print("Keypoint order already matches the system's order -- no remapping needed.")

    return perm


def permute_keypoints(flat_keypoints: list[float], perm: list[int]) -> list[float]:
    """Reorders a flat [x,y,v]*13 list from Roboflow order into system order."""
    out: list[float] = [0.0] * (NUM_KEYPOINTS * 3)
    for system_i, roboflow_i in enumerate(perm):
        out[3 * system_i : 3 * system_i + 3] = flat_keypoints[3 * roboflow_i : 3 * roboflow_i + 3]
    return out


def prepare(
    raw: dict[str, Any],
    specimen_map: dict[str, str] | None,
    color_morph_map: dict[str, str] | None,
    source: str,
) -> dict[str, Any]:
    """Mutates and returns `raw`: permutes keypoints into system order and
    fills in `fish_box`, `specimen_id`, `source`, `color_morph`, and
    `has_occlusion` on every annotation."""
    images_by_id = {img["id"]: img for img in raw["images"]}
    perm = build_permutation(get_roboflow_keypoint_names(raw))
    n_defaulted_specimen = 0
    n_defaulted_morph = 0

    for ann in raw["annotations"]:
        if "keypoints" in ann:
            if len(ann["keypoints"]) != NUM_KEYPOINTS * 3:
                raise ValueError(
                    f"annotation {ann.get('id', '?')} (image_id={ann.get('image_id', '?')}) has "
                    f"{len(ann['keypoints'])} keypoint values, expected {NUM_KEYPOINTS * 3}."
                )
            ann["keypoints"] = permute_keypoints(ann["keypoints"], perm)
            ann["has_occlusion"] = any(
                ann["keypoints"][3 * i + 2] != Visibility.CLEAR for i in range(NUM_KEYPOINTS)
            )
        else:
            ann["has_occlusion"] = False

        if "fish_box" not in ann:
            if "bbox" not in ann:
                raise ValueError(
                    f"annotation {ann.get('id', '?')} (image_id={ann.get('image_id', '?')}) has "
                    "neither 'fish_box' nor 'bbox' -- cannot proceed."
                )
            ann["fish_box"] = ann["bbox"]

        image_meta = images_by_id.get(ann["image_id"])
        file_name = image_meta.get("file_name") if image_meta else None

        if "specimen_id" not in ann:
            mapped = specimen_map.get(file_name) if (specimen_map and file_name) else None
            if mapped is not None:
                ann["specimen_id"] = mapped
            else:
                # Default to the ANNOTATION's own id, not the image's id.
                # An image with multiple fish in it (a catalog/comparison
                # photo) has multiple annotations sharing one image_id --
                # keying off image_id would silently force every fish in
                # that photo into the same "specimen", which isn't correct
                # and needlessly couples unrelated fish in the split.
                ann["specimen_id"] = f"img{ann['image_id']}_ann{ann.get('id', 'NA')}"
                n_defaulted_specimen += 1

        if "color_morph" not in ann:
            mapped = color_morph_map.get(file_name) if (color_morph_map and file_name) else None
            if mapped is not None:
                ann["color_morph"] = mapped
            else:
                ann["color_morph"] = "unspecified"
                n_defaulted_morph += 1

        if "source" not in ann:
            ann["source"] = source

    if n_defaulted_specimen:
        print(
            f"NOTE: {n_defaulted_specimen} annotation(s) had no specimen_id and no --specimen-map entry, "
            "so each was defaulted to its own specimen_id (unique per annotation, so multiple fish "
            "in one photo don't get merged). This is SAFE (no "
            "train/val/test leakage) but means grouped splitting has no effect for them -- if "
            "several of your photos are the SAME physical fish, supply --specimen-map or "
            "leakage protection won't apply to those images."
        )
    if n_defaulted_morph:
        print(
            f"NOTE: {n_defaulted_morph} annotation(s) had no color_morph and no --color-morph-map entry, "
            "so each was defaulted to 'unspecified'. Stratification by color_morph is a no-op until "
            "you supply real values -- fine for now, but note it as a limitation in the thesis text."
        )

    return raw


def load_csv_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        key, value = reader.fieldnames[0], reader.fieldnames[1]
        return {row[key]: row[value] for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", required=True, help="Raw Roboflow COCO Keypoints export.")
    parser.add_argument("--out", dest="out_path", required=True, help="Where to write the normalized annotations.json.")
    parser.add_argument(
        "--specimen-map", default=None,
        help="Optional CSV with columns file_name,specimen_id, for images that are the same physical fish.",
    )
    parser.add_argument(
        "--color-morph-map", default=None,
        help="Optional CSV with columns file_name,color_morph.",
    )
    parser.add_argument("--source", default="roboflow", help="Constant value for the 'source' stratify field.")
    args = parser.parse_args()

    raw = json.loads(Path(args.in_path).read_text())
    specimen_map = load_csv_map(args.specimen_map)
    color_morph_map = load_csv_map(args.color_morph_map)
    prepared = prepare(raw, specimen_map, color_morph_map, args.source)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(prepared, indent=2))
    print(f"Wrote {len(prepared['annotations'])} normalized annotations to {args.out_path}")


if __name__ == "__main__":
    main()