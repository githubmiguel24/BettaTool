"""Visual review for specimen_id grouping.

Crops each annotation's fish_box out of its source image and tiles crops
side-by-side, grouped by specimen_id, so you can eyeball whether the
grouping is right: crops sharing a specimen_id sheet should all be the SAME
physical fish; crops on different sheets should be genuinely different fish.
This is the direct way to check --specimen-map (or the auto default) got it
right before you freeze a split on top of it (training/splitting.py trusts
specimen_id completely -- a wrong grouping here silently weakens or
defeats its leakage protection).

By default only renders groups with more than one annotation -- a
single-annotation specimen has nothing to compare against, so there's
nothing to check. Pass --all to render every group instead (useful once you
have a real --specimen-map and want to confirm ALL of it, not just the
multi-photo cases).

Usage:
    python -m training.review_specimens --ann data/annotations/annotations.json \\
        --images data/raw --out training/runs/_specimen_review
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_review(ann_path: Path, images_dir: Path, out_dir: Path, only_grouped: bool = True) -> tuple[int, int]:
    coco = json.loads(ann_path.read_text())
    images_by_id = {img["id"]: img for img in coco["images"]}

    by_specimen: dict[str, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_specimen[str(ann.get("specimen_id"))].append(ann)

    out_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(16)

    n_written = 0
    n_skipped_missing_image = 0
    for specimen_id, anns in sorted(by_specimen.items(), key=lambda kv: -len(kv[1])):
        if only_grouped and len(anns) < 2:
            continue

        crops: list[Image.Image] = []
        labels: list[str] = []
        for ann in anns:
            img_meta = images_by_id.get(ann["image_id"])
            if img_meta is None:
                continue
            img_path = images_dir / img_meta["file_name"]
            if not img_path.exists():
                n_skipped_missing_image += 1
                continue
            box = ann.get("fish_box") or ann.get("bbox")
            if not box:
                continue
            x, y, w, h = box
            pad = 0.08
            x0, y0 = max(0, x - w * pad), max(0, y - h * pad)
            x1, y1 = x + w * (1 + pad), y + h * (1 + pad)
            with Image.open(img_path) as im:
                crop = im.convert("RGB").crop((int(x0), int(y0), int(x1), int(y1)))
            crops.append(crop)
            labels.append(f"{img_meta['file_name']}\nann_id={ann.get('id')}")

        if not crops:
            continue

        target_h = 260
        resized = []
        for c in crops:
            scale = target_h / max(1, c.height)
            resized.append(c.resize((max(1, int(c.width * scale)), target_h)))

        label_h = 40
        gap = 10
        total_w = sum(r.width for r in resized) + gap * (len(resized) - 1)
        sheet = Image.new("RGB", (total_w, target_h + label_h), "white")
        draw = ImageDraw.Draw(sheet)
        x_cursor = 0
        for r, label in zip(resized, labels):
            sheet.paste(r, (x_cursor, label_h))
            draw.text((x_cursor + 4, 4), label, fill="black", font=font)
            x_cursor += r.width + gap

        safe_id = specimen_id.replace("/", "_").replace("\\", "_")
        sheet.save(out_dir / f"specimen_{safe_id}.png")
        n_written += 1

    return n_written, n_skipped_missing_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann", required=True, help="Path to annotations.json (post prepare_annotations.py).")
    parser.add_argument("--images", required=True, help="Directory containing the actual image files.")
    parser.add_argument("--out", required=True, help="Directory to write review sheets into (one PNG per specimen_id).")
    parser.add_argument(
        "--all", dest="all_groups", action="store_true",
        help="Render every specimen group, not just ones with more than one annotation.",
    )
    args = parser.parse_args()

    n_written, n_skipped = build_review(
        Path(args.ann), Path(args.images), Path(args.out), only_grouped=not args.all_groups
    )
    print(f"Wrote {n_written} specimen review sheet(s) to {args.out}")
    if n_skipped:
        print(f"NOTE: skipped {n_skipped} annotation(s) whose image file wasn't found under {args.images}.")
    if n_written == 0 and not args.all_groups:
        print(
            "No sheets written because no specimen_id currently has more than one annotation -- "
            "expected if you haven't supplied --specimen-map yet (every fish defaults to its own "
            "specimen). Re-run with --all to see every crop individually instead."
        )


if __name__ == "__main__":
    main()