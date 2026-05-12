#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prediction COCO json against GT COCO json on GT-covered images "
            "and export per-image error details."
        )
    )
    parser.add_argument("--pred", default="Test/pred.json", help="Path to prediction COCO json")
    parser.add_argument("--gt", default="Test/GT.json", help="Path to GT COCO json")
    parser.add_argument(
        "--output",
        default="Test/pred_gt_error_cases.json",
        help="Path to output error details json",
    )
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--image-key",
        default="image_relpath",
        help="Image identity key in images[] for cross-file matching",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = b
    return [float(x), float(y), float(x + w), float(y + h)]


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_image_map(images: List[Dict[str, Any]], image_key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for im in images:
        key = im.get(image_key) or im.get("file_name") or str(im.get("id"))
        out[key] = im
    return out


def build_ann_index(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    ann_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_map[ann["image_id"]].append(ann)
    return ann_map


def preprocess_ann(
    ann: Dict[str, Any],
    cat_id_to_name: Dict[int, str],
    ann_type: str,
) -> Dict[str, Any]:
    bbox_xywh = ann["bbox"]
    bbox_xyxy = ann.get("bbox_xyxy") or xywh_to_xyxy(bbox_xywh)
    obj = {
        "ann_id": ann.get("id"),
        "category_id": ann.get("category_id"),
        "category_name": cat_id_to_name.get(ann.get("category_id"), str(ann.get("category_id"))),
        "bbox_xywh": [float(v) for v in bbox_xywh],
        "bbox_xyxy": [float(v) for v in bbox_xyxy],
    }
    if ann_type == "pred":
        obj["score"] = ann.get("score")
    return obj


def best_iou(target_box: List[float], candidates: List[Dict[str, Any]]) -> Tuple[float, int]:
    best = 0.0
    best_idx = -1
    for i, cand in enumerate(candidates):
        cur = iou_xyxy(target_box, cand["bbox_xyxy"])
        if cur > best:
            best = cur
            best_idx = i
    return best, best_idx


def evaluate_image(
    image_key: str,
    gt_image: Dict[str, Any],
    pred_image: Dict[str, Any],
    gt_objs: List[Dict[str, Any]],
    pred_objs: List[Dict[str, Any]],
    iou_thr: float,
) -> Dict[str, Any]:
    pairs = []
    for gi, g in enumerate(gt_objs):
        for pi, p in enumerate(pred_objs):
            pairs.append((iou_xyxy(g["bbox_xyxy"], p["bbox_xyxy"]), gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_g = set()
    matched_p = set()
    matched_pairs = []
    for iou, gi, pi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        matched_pairs.append((gi, pi, iou))

    errors = []

    # Matched pairs: classify low IoU and class mismatch
    for gi, pi, iou in matched_pairs:
        g = gt_objs[gi]
        p = pred_objs[pi]

        if iou < iou_thr:
            errors.append(
                {
                    "type": "iou_below_threshold",
                    "iou": round(float(iou), 6),
                    "threshold": iou_thr,
                    "gt_index": gi,
                    "pred_index": pi,
                    "gt": g,
                    "pred": p,
                }
            )

        if g["category_id"] != p["category_id"]:
            errors.append(
                {
                    "type": "class_mismatch",
                    "iou": round(float(iou), 6),
                    "gt_index": gi,
                    "pred_index": pi,
                    "gt": g,
                    "pred": p,
                }
            )

    # Unmatched GT: missed detection
    for gi, g in enumerate(gt_objs):
        if gi in matched_g:
            continue
        best, best_pi = best_iou(g["bbox_xyxy"], pred_objs)
        payload = {
            "type": "missed_detection",
            "gt_index": gi,
            "gt": g,
            "best_iou_with_any_pred": round(float(best), 6),
        }
        if best_pi >= 0:
            payload["best_pred_index"] = best_pi
            payload["best_pred"] = pred_objs[best_pi]
        errors.append(payload)

    # Unmatched pred: false positive
    for pi, p in enumerate(pred_objs):
        if pi in matched_p:
            continue
        best, best_gi = best_iou(p["bbox_xyxy"], gt_objs)
        payload = {
            "type": "false_positive",
            "pred_index": pi,
            "pred": p,
            "best_iou_with_any_gt": round(float(best), 6),
        }
        if best_gi >= 0:
            payload["best_gt_index"] = best_gi
            payload["best_gt"] = gt_objs[best_gi]
        errors.append(payload)

    error_types = sorted({e["type"] for e in errors})

    image_info = {
        "image_key": image_key,
        "file_name": gt_image.get("file_name"),
        "image_relpath": gt_image.get("image_relpath"),
        "image_abspath": gt_image.get("image_abspath"),
        "case_relpath": gt_image.get("case_relpath"),
        "width": gt_image.get("width"),
        "height": gt_image.get("height"),
        "gt_image_id": gt_image.get("id"),
        "pred_image_id": pred_image.get("id") if pred_image else None,
        "gt_count": len(gt_objs),
        "pred_count": len(pred_objs),
        "error_types": error_types,
        "errors": errors,
    }
    return image_info


def main() -> None:
    args = parse_args()

    pred = load_json(args.pred)
    gt = load_json(args.gt)

    pred_cat = {c["id"]: c["name"] for c in pred.get("categories", [])}
    gt_cat = {c["id"]: c["name"] for c in gt.get("categories", [])}

    pred_img_by_key = build_image_map(pred.get("images", []), args.image_key)
    gt_img_by_key = build_image_map(gt.get("images", []), args.image_key)

    pred_ann_map = build_ann_index(pred.get("annotations", []))
    gt_ann_map = build_ann_index(gt.get("annotations", []))

    error_images = []
    type_counter = Counter()

    for key, gt_image in gt_img_by_key.items():
        pred_image = pred_img_by_key.get(key)

        gt_anns = gt_ann_map.get(gt_image["id"], [])
        pred_anns = pred_ann_map.get(pred_image["id"], []) if pred_image else []

        gt_objs = [preprocess_ann(a, gt_cat, ann_type="gt") for a in gt_anns]
        pred_objs = [preprocess_ann(a, pred_cat, ann_type="pred") for a in pred_anns]

        image_result = evaluate_image(
            image_key=key,
            gt_image=gt_image,
            pred_image=pred_image,
            gt_objs=gt_objs,
            pred_objs=pred_objs,
            iou_thr=args.iou_thr,
        )

        if image_result["errors"]:
            error_images.append(image_result)
            for e in image_result["errors"]:
                type_counter[e["type"]] += 1

    summary = {
        "pred_json": args.pred,
        "gt_json": args.gt,
        "output_json": args.output,
        "image_key": args.image_key,
        "iou_threshold": args.iou_thr,
        "gt_image_count": len(gt_img_by_key),
        "pred_image_count": len(pred_img_by_key),
        "evaluated_gt_images": len(gt_img_by_key),
        "error_image_count": len(error_images),
        "error_instance_count": int(sum(type_counter.values())),
        "error_type_counts": dict(type_counter),
    }

    out = {
        "summary": summary,
        "error_images": error_images,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", args.output)
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
