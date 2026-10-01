# converts raw roboflow coco exports into our internal annotation format
# fixes bbox naming, missing metadata fields, and keypoint ordering issues

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.perception.keypoints import KEYPOINT_SHORT_CODES, NUM_KEYPOINTS, Visibility

# maps roboflow keypoint names to our internal short codes (indices 4 and 5 are swapped in roboflow)
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


# grabs the keypoint names list from the first matching category in the Export
def get_roboflow_keypoint_names(raw: dict[str, Any]) -> list[str]:
    for cat in raw.get("categories", []):
        if "keypoints" in cat:
            return cat["keypoints"]
    raise ValueError("No category with a 'keypoints' list found in the export's categories[].")


# builds index mapping from roboflow keypoint order to our system order
def build_permutation(roboflow_names: list[str]) -> list[int]:
    # make sure the export actually has the expected number of landmarks
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

    # find where each system keypoint sits in the roboflow list
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


# reorders flat [x, y, v] keyponts array using the permutation indices
def permute_keypoints(flat_keypoints: list[float], perm: list[int]) -> list[float]:
    out: list[float] = [0.0] * (NUM_KEYPOINTS * 3)
    for system_i, roboflow_i in enumerate(perm):
        # copy each x, y, v triplet into the right slot
        out[3 * system_i : 3 * system_i + 3] = flat_keypoints[3 * roboflow_i : 3 * roboflow_i + 3]
    return out


# updates annotations in place with reordered keypoints and required metadata fields
def prepare(
    raw: dict[str, Any],
    specimen_map: dict[str, str] | None,
    color_morph_map: dict[str, str] | None,
    source: str,
) -> dict[str, Any]:
    images_by_id = {img["id"]: img for img in raw["images"]}
    perm = build_permutation(get_roboflow_keypoint_names(raw))
    n_defaulted_specimen = 0
    n_defaulted_morph = 0

    for ann in raw["annotations"]:
        # reorder keypoints and check if any landmark is not clearly visible
        if "keypoints" in ann:
            if len(ann["keypoints"]) != NUM_KEYPOINTS * 3:
                raise ValueError(
                    f"annotation {ann.get('id', '?')} (image_id={ann.get('image_id', '?')}) has "
                    f"{len(ann['keypoints'])} keypoint values, expected {NUM_KEYPOINTS * 3}."
                )
            ann["keypoints"] = permute_keypoints(ann["keypoints"], perm)
            # coco v=2 is clear, v=1 is occluded, v=0 is unlabeled or out of frame
            ann["has_occlusion"] = any(
                ann["keypoints"][3 * i + 2] != Visibility.CLEAR for i in range(NUM_KEYPOINTS)
            )
        else:
            ann["has_occlusion"] = False

        # copy coco bbox to fish_box
        if "fish_box" not in ann:
            if "bbox" not in ann:
                raise ValueError(
                    f"annotation {ann.get('id', '?')} (image_id={ann.get('image_id', '?')}) has "
                    "neither 'fish_box' nor 'bbox' -- cannot proceed."
                )
            ann["fish_box"] = ann["bbox"]

        image_meta = images_by_id.get(ann["image_id"])
        file_name = image_meta.get("file_name") if image_meta else None

        # assign specimen_id from csv map or fallback to unique id per annotation
        if "specimen_id" not in ann:
            mapped = specimen_map.get(file_name) if (specimen_map and file_name) else None
            if mapped is not None:
                ann["specimen_id"] = mapped
            else:
                # include annotation id so multiple fish in one photo dont get grouped as one specimen
                ann["specimen_id"] = f"img{ann['image_id']}_ann{ann.get('id', 'NA')}"
                n_defaulted_specimen += 1

        # set color_morph from map or defualt to unspecified
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


# loads a two column csv into a simple lookup dict
def load_csv_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        key, value = reader.fieldnames[0], reader.fieldnames[1]
        return {row[key]: row[value] for row in reader}


# parses cli args and runs the annotation normalization
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

    # read raw export and optional mapping files
    raw = json.loads(Path(args.in_path).read_text())
    specimen_map = load_csv_map(args.specimen_map)
    color_morph_map = load_csv_map(args.color_morph_map)
    prepared = prepare(raw, specimen_map, color_morph_map, args.source)

    # save normalized annotaions to disk
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(prepared, indent=2))
    print(f"Wrote {len(prepared['annotations'])} normalized annotations to {args.out_path}")


if __name__ == "__main__":
    main()