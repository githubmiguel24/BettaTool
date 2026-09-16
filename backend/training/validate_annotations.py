"""Annotation sanity-check CLI (Build Prompt v2 §5.4).

    python -m training.validate_annotations --ann data/annotations/annotations.json \\
        --images data/raw --plots-out training/runs/_validation_plots

Reports: images referenced but missing from disk, annotations with keypoint
count != 13, coordinates outside image bounds, duplicate image IDs, missing
fish_box, a per-keypoint visibility histogram (counts of flags 2/1/0/-1),
and per-keypoint coordinate distribution plots. Run this the moment labeling
finishes, and watch the visibility histogram: if a landmark is masked in a
large fraction of images, masking stops being viable and that landmark's
observability needs rethinking at the data level (Build Prompt v2 §5.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.perception.keypoints import KEYPOINT_SHORT_CODES, NUM_KEYPOINTS
from training.dataset import build_validation_report


def _print_report(report: dict, ann_path: Path) -> int:
    """Prints the human-readable report; returns a process exit code (0 clean, 1 problems found)."""
    print(f"Annotation file: {ann_path}")
    print(f"  images:      {report['n_images']}")
    print(f"  annotations: {report['n_annotations']}")
    print()

    print("Visibility histogram (all keypoints, all images):")
    total = sum(report["visibility_histogram"].values()) or 1
    for flag, label in [(2, "clear"), (1, "ambiguous"), (0, "occluded"), (-1, "out_of_frame")]:
        count = report["visibility_histogram"].get(flag, 0)
        print(f"    {flag:>2} ({label:<12}): {count:>6}  ({100 * count / total:5.1f}%)")
    print()

    print("Per-keypoint visibility (rows sum to n_annotations; watch for a landmark that is")
    print("mostly occluded/out-of-frame — masking stops being viable at that point, Build Prompt v2 §5.4):")
    header = f"  {'landmark':<20} {'clear':>7} {'ambig':>7} {'occl':>7} {'oof':>7}  masked%"
    print(header)
    n_ann = report["n_annotations"] or 1
    for i in range(NUM_KEYPOINTS):
        counts = report["per_keypoint_visibility"][i]
        masked_pct = 100 * (counts.get(0, 0) + counts.get(-1, 0)) / n_ann
        flag_str = f"{counts.get(2, 0):>7} {counts.get(1, 0):>7} {counts.get(0, 0):>7} {counts.get(-1, 0):>7}"
        flag_alert = "  <-- >20% masked" if masked_pct > 20 else ""
        print(f"  {KEYPOINT_SHORT_CODES[i]:<20} {flag_str}  {masked_pct:5.1f}%{flag_alert}")
    print()

    n_problems = len(report["problems_by_image"])
    if n_problems:
        print(f"PROBLEMS FOUND in {n_problems} annotation(s):")
        for image_id, problems in list(report["problems_by_image"].items())[:50]:
            print(f"  image_id={image_id}:")
            for p in problems:
                print(f"    - {p}")
        if n_problems > 50:
            print(f"  ... and {n_problems - 50} more")
    else:
        print("No structural problems found.")

    if report["duplicate_image_ids"]:
        print(f"\nDUPLICATE image_ids: {report['duplicate_image_ids']}")

    if report["missing_fish_box_count"]:
        print(f"\n{report['missing_fish_box_count']} annotation(s) missing fish_box.")

    return 1 if (n_problems or report["duplicate_image_ids"] or report["missing_fish_box_count"]) else 0


def _plot_coordinate_distributions(report: dict, ann_path: Path, out_dir: Path) -> None:
    """Per-keypoint (x, y) scatter, one subplot per landmark, saved as one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coco = json.loads(ann_path.read_text())
    coords_by_kp: dict[int, list[tuple[float, float]]] = {i: [] for i in range(NUM_KEYPOINTS)}
    for sample in coco.get("annotations", []):
        keypoints = sample.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) != NUM_KEYPOINTS * 3:
            continue
        for i in range(NUM_KEYPOINTS):
            x, y, v = keypoints[3 * i : 3 * i + 3]
            if v in (2, 1):
                coords_by_kp[i].append((x, y))

    out_dir.mkdir(parents=True, exist_ok=True)
    n_cols = 4
    n_rows = (NUM_KEYPOINTS + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i in range(NUM_KEYPOINTS):
        ax = axes.flat[i]
        pts = coords_by_kp[i]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=4, alpha=0.5)
            ax.invert_yaxis()
        ax.set_title(KEYPOINT_SHORT_CODES[i], fontsize=9)
    for ax in axes.flat[NUM_KEYPOINTS:]:
        ax.axis("off")
    fig.tight_layout()
    out_path = out_dir / "keypoint_coordinate_distributions.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved per-keypoint coordinate distribution plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=str, required=True, help="Path to the COCO Keypoints JSON file.")
    parser.add_argument("--images", type=str, required=True, help="Directory containing the referenced image files.")
    parser.add_argument("--plots-out", type=str, default=None, help="If set, saves coordinate-distribution plots here.")
    args = parser.parse_args()

    ann_path = Path(args.ann)
    report = build_validation_report(ann_path, args.images)
    exit_code = _print_report(report, ann_path)

    if args.plots_out:
        _plot_coordinate_distributions(report, ann_path, Path(args.plots_out))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
