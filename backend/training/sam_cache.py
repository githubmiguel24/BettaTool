"""Run SAM (facebook/sam-vit-base via transformers) on the VAL images and cache the masks (feasibility study only).

    python -m training.sam_cache --pred training/runs/v2_run/eval_val/predictions.npz --out training/runs/v2_run/sam_cache

Prompts per image (all in original-image pixels):
  A  the fish_box                                  (SAM first, no keypoints)
  B  the 13 model-predicted keypoints as points    (keypoints first, no box)
  B2 the 13 model-predicted keypoints + fish_box
Each prompt returns SAM's 3 candidate masks, stored at <=512 px on the long side (bool, plus the IoU scores).
Nothing is trained or fitted here, and the test split is not touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from training.utils.config import load_config

STORE = 512


@torch.no_grad()
def run_prompt(model, emb, scale, box=None, points=None, device="cuda"):
    kw = {}
    if box is not None:
        kw["input_boxes"] = torch.tensor([[list(np.asarray(box, dtype=np.float32) * scale)]], device=device)
    if points is not None:
        pts = np.asarray(points, dtype=np.float32) * scale
        kw["input_points"] = torch.tensor(pts[None, None], device=device)           # (1, 1, n, 2)
        kw["input_labels"] = torch.ones((1, 1, len(pts)), dtype=torch.long, device=device)
    out = model(image_embeddings=emb, multimask_output=True, **kw)
    low = out.pred_masks[0, 0]                                                      # (3, 256, 256) logits
    return low, out.iou_scores[0, 0].float().cpu().numpy()


def to_store_res(low_logits, w, h, scale_store):
    # upsample the 256x256 low-res logits (defined on the 1024 padded frame) to a <=512 px mask of the original image
    side = 1024
    s1024 = side / max(w, h)
    up = torch.nn.functional.interpolate(low_logits[None].float(), size=(side, side), mode="bilinear", align_corners=False)[0]
    up = up[:, : int(round(h * s1024)), : int(round(w * s1024))]
    sw, sh = max(1, int(round(w * scale_store))), max(1, int(round(h * scale_store)))
    up = torch.nn.functional.interpolate(up[None], size=(sh, sw), mode="bilinear", align_corners=False)[0]
    return (up > 0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default="training/configs/hrnet_w32_v2.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()

    coco = json.loads((Path(cfg["paths"]["annotations_dir"]) / "annotations.json").read_text())
    meta = {str(i["id"]): i for i in coco["images"]}
    box = {}
    for a in coco["annotations"]:
        box.setdefault(str(a["image_id"]), a["fish_box"])
    d = np.load(args.pred)
    ids = [str(i) for i in d["image_ids"]]
    if args.limit:
        ids = ids[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for n, iid in enumerate(ids):
        im = Image.open(Path(cfg["paths"]["images_dir"]) / meta[iid]["file_name"]).convert("RGB")
        w, h = im.size
        bx, by, bw, bh = box[iid]
        xyxy = [bx, by, bx + bw, by + bh]
        enc = processor(im, return_tensors="pt").to(dev)
        emb = model.get_image_embeddings(enc["pixel_values"])
        scale1024 = 1024 / max(w, h)
        scale_store = min(1.0, STORE / max(w, h))
        pts = d["pred_mu"][n]
        masks, scores = {}, {}
        for name, kw in {"A": {"box": xyxy}, "B": {"points": pts}, "B2": {"box": xyxy, "points": pts}}.items():
            low, sc = run_prompt(model, emb, scale1024, device=dev, **kw)
            masks[name] = np.packbits(to_store_res(low, w, h, scale_store), axis=-1)
            scores[name] = sc
        np.savez_compressed(out / f"{iid}.npz", A=masks["A"], B=masks["B"], B2=masks["B2"], sA=scores["A"], sB=scores["B"], sB2=scores["B2"],
                            scale=scale_store, size=np.array([w, h]), shape=np.array([max(1, int(round(h * scale_store))), max(1, int(round(w * scale_store)))]))
        if n % 25 == 0:
            print(f"{n + 1}/{len(ids)} image {iid} done", flush=True)
    print("cached", len(ids), "images to", out)


if __name__ == "__main__":
    main()
