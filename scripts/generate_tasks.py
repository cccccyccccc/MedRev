#!/usr/bin/env python3
"""Generate review task JSONL files from legacy list JSON or COCO JSON."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def evaluate_gt_prediction(gt_annos, preds, iou_thresh: float):
    """Return whether pred fully matches GT plus mismatch reasons for page 1.

    A GT image is considered correct only when predictions and GT annotations
    can be matched one-to-one with the same category and IoU greater than or
    equal to the configured threshold. Unmatched GT means missed detection;
    unmatched prediction means false positive. This intentionally does not
    filter predictions by confidence, matching the reviewer-provided script.
    """
    qualified_preds = list(preds)

    candidate_pairs = []
    for gt_index, gt in enumerate(gt_annos):
        for pred_index, pred in enumerate(qualified_preds):
            pair_iou = iou(gt.get('bbox', [0, 0, 0, 0]), pred.get('bbox', [0, 0, 0, 0]))
            if gt.get('category') == pred.get('category') and pair_iou >= iou_thresh:
                candidate_pairs.append((pair_iou, gt_index, pred_index))

    matched_gt = set()
    matched_pred = set()
    for _, gt_index, pred_index in sorted(candidate_pairs, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

    reasons = []
    for gt_index, gt in enumerate(gt_annos):
        if gt_index in matched_gt:
            continue
        best_pred = None
        best_iou = 0.0
        for pred_index, pred in enumerate(qualified_preds):
            if pred_index in matched_pred:
                continue
            pair_iou = iou(gt.get('bbox', [0, 0, 0, 0]), pred.get('bbox', [0, 0, 0, 0]))
            if pair_iou > best_iou:
                best_iou = pair_iou
                best_pred = pred
        if best_pred and best_iou >= iou_thresh and best_pred.get('category') != gt.get('category'):
            reasons.append({
                'type': 'category_error',
                'gt_category': gt.get('category'),
                'pred_category': best_pred.get('category'),
                'iou': best_iou,
            })
        elif best_pred and best_pred.get('category') == gt.get('category') and best_iou < iou_thresh:
            reasons.append({
                'type': 'box_iou_error',
                'gt_category': gt.get('category'),
                'pred_category': best_pred.get('category'),
                'iou': best_iou,
            })
        else:
            reasons.append({
                'type': 'false_negative',
                'gt_category': gt.get('category'),
                'iou': best_iou,
            })

    for pred_index, pred in enumerate(qualified_preds):
        if pred_index in matched_pred:
            continue
        best_gt = None
        best_iou = 0.0
        for gt in gt_annos:
            pair_iou = iou(gt.get('bbox', [0, 0, 0, 0]), pred.get('bbox', [0, 0, 0, 0]))
            if pair_iou > best_iou:
                best_iou = pair_iou
                best_gt = gt
        if best_gt and best_iou >= iou_thresh and best_gt.get('category') != pred.get('category'):
            reasons.append({
                'type': 'category_error',
                'gt_category': best_gt.get('category'),
                'pred_category': pred.get('category'),
                'iou': best_iou,
            })
        else:
            reasons.append({
                'type': 'false_positive',
                'pred_category': pred.get('category'),
                'confidence': pred.get('confidence', 0.0),
                'iou': best_iou,
            })

    return len(reasons) == 0, reasons, qualified_preds


def load_json(path: Path):
    if not path.exists():
        return {} if path.suffix.lower() == ".json" else []
    return json.loads(path.read_text(encoding="utf-8"))


def as_xyxy(bbox) -> List[float]:
    """Convert likely COCO xywh or legacy xyxy boxes to xyxy for app rendering."""
    if not bbox or len(bbox) < 4:
        return [0, 0, 0, 0]
    x0, y0, a, b = [float(v) for v in bbox[:4]]
    # COCO boxes in this project are xywh; legacy demo boxes are already xyxy.
    if a >= x0 and b >= y0:
        return [x0, y0, a, b]
    return [x0, y0, x0 + max(0.0, a), y0 + max(0.0, b)]


def xywh_to_xyxy(bbox) -> List[float]:
    if not bbox or len(bbox) < 4:
        return [0, 0, 0, 0]
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def find_image_path(root: Path, image_name: str, image_relpath: str = "", image_abspath: str = "") -> str:
    if image_abspath and Path(image_abspath).exists():
        return Path(image_abspath).as_posix()
    if image_relpath:
        candidates = [
            root / image_relpath,
            root / "raw_subset" / image_relpath,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve().as_posix()
    for p in root.rglob(image_name):
        return p.resolve().as_posix()
    return ""


def build_category_map(coco: Dict[str, Any]) -> Dict[Any, str]:
    return {c.get("id"): c.get("name", str(c.get("id"))) for c in coco.get("categories", [])}


def group_annotations(coco: Dict[str, Any]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for anno in coco.get("annotations", []):
        grouped.setdefault(anno.get("image_id"), []).append(anno)
    return grouped


def load_case_metadata(organ_dir: Path) -> Dict[str, Any]:
    path = organ_dir / "case_metadata.json"
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, dict):
        return data.get("cases", data)
    return {}


def case_info_for(image: Dict[str, Any], case_metadata: Dict[str, Any]) -> Dict[str, Any]:
    case_relpath = image.get("case_relpath") or str(Path(image.get("image_relpath", "")).parent).replace("\\", "/")
    return case_metadata.get(case_relpath, {})


def infer_source_root(image_path: str, image_relpath: str) -> Path | None:
    if not image_path or not image_relpath:
        return None
    image = Path(image_path).resolve()
    rel_parts = Path(image_relpath).parts
    if len(image.parts) < len(rel_parts):
        return None
    return Path(*image.parts[:len(image.parts) - len(rel_parts)])


def localize_relpath(data_root: Path, image_relpath: str, fallback_abspath: str = "", source_root: Path | None = None) -> str:
    if source_root:
        candidate = source_root / image_relpath
        if candidate.exists():
            return candidate.resolve().as_posix()
    return find_image_path(data_root, Path(image_relpath).name, image_relpath, fallback_abspath)


def related_images_for(image_path: str, current_name: str, data_root: Path, image: Dict[str, Any], case_metadata: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    info = case_info_for(image, case_metadata)
    source_root = infer_source_root(image_path, image.get("image_relpath", ""))
    report_paths = [
        localize_relpath(data_root, rel, source_root=source_root)
        for rel in info.get("report_images", [])
    ]
    report_paths = [p for p in report_paths if p]

    related = []
    for rel in info.get("sequence_images", []):
        path = localize_relpath(data_root, rel, source_root=source_root)
        if path and Path(path).name != current_name:
            related.append(path)

    if not related and image_path:
        folder = Path(image_path).parent
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"} and p.name != current_name:
                    related.append(p.resolve().as_posix())

    report_image = report_paths[0] if report_paths else ""
    if not report_image and image_path:
        legacy_report = Path(image_path).parent / "病例报告.jpg"
        if legacy_report.exists():
            report_image = legacy_report.resolve().as_posix()
            report_paths = [report_image]
    return report_image, report_paths, related


def stable_task_suffix(task_type: str, image: Dict[str, Any], image_name: str) -> str:
    rel = image.get("image_relpath") or image_name
    return f"{task_type}_{rel}".replace("\\", "/").replace("/", "_")


def coco_to_maps(coco: Dict[str, Any], data_root: Path, is_prediction: bool) -> Dict[str, Dict[str, Any]]:
    categories = build_category_map(coco)
    annotations_by_image = group_annotations(coco)
    out: Dict[str, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_name = image.get("file_name") or image.get("image_name") or ""
        key = image.get("image_relpath") or image_name
        anns = annotations_by_image.get(image.get("id"), [])
        boxes = []
        for anno in anns:
            category = categories.get(anno.get("category_id"), str(anno.get("category_id")))
            if is_prediction:
                boxes.append({
                    "category": category,
                    "bbox": xywh_to_xyxy(anno.get("bbox")),
                    "confidence": anno.get("score", 0.0),
                    "category_id": anno.get("category_id"),
                    "model_class_id": anno.get("model_class_id"),
                })
            else:
                boxes.append({
                    "category": category,
                    "bbox": xywh_to_xyxy(anno.get("bbox")),
                    "confidence": 1.0,
                    "category_id": anno.get("category_id"),
                    "source_note": anno.get("source_note", ""),
                    "source_category_raw": anno.get("source_category_raw", category),
                })
        out[key] = {
            "image_name": image_name,
            "image_relpath": image.get("image_relpath", ""),
            "image_path": find_image_path(data_root, image_name, image.get("image_relpath", ""), image.get("image_abspath", "")),
            "image_width": image.get("width", 0),
            "image_height": image.get("height", 0),
            "case_relpath": image.get("case_relpath", ""),
            "organ": image.get("organ", ""),
            "image": image,
            "annotations" if not is_prediction else "predictions": boxes,
            "model_version": coco.get("info", {}).get("model_version") or coco.get("info", {}).get("version"),
        }
    return out


def organ_coco_to_map(coco: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(coco, dict):
        return {}
    out = {}
    for image in coco.get("images", []):
        key = image.get("image_relpath") or image.get("file_name") or image.get("image_name")
        if not key:
            continue
        out[key] = {
            "organ_present": bool(image.get("has_organ", image.get("organ_present", False))),
            "confidence": image.get("best_score", image.get("confidence", 0.0)),
            "raw": image,
        }
    return out


def legacy_list_to_map(rows: List[Dict[str, Any]], data_root: Path, key_name: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        image_name = row.get("image_name") or row.get("file_name") or ""
        row = dict(row)
        row["image_name"] = image_name
        if key_name == "annotations":
            for box in row.get("annotations", []):
                box["bbox"] = as_xyxy(box.get("bbox"))
        if key_name == "predictions":
            for box in row.get("predictions", []):
                box["bbox"] = as_xyxy(box.get("bbox"))
        row.setdefault("image_path", find_image_path(data_root, image_name))
        out[image_name] = row
    return out


def load_gt_pred_organ(organ_dir: Path, data_root: Path):
    gt_raw = load_json(organ_dir / "GT.json")
    pred_raw = load_json(organ_dir / "pred.json")
    organ_raw = load_json(organ_dir / "organ.json")

    if isinstance(gt_raw, dict):
        gt_map = {k: v for k, v in coco_to_maps(gt_raw, data_root, is_prediction=False).items() if v.get("annotations")}
    else:
        gt_map = legacy_list_to_map(gt_raw, data_root, "annotations")

    if isinstance(pred_raw, dict):
        pred_map = coco_to_maps(pred_raw, data_root, is_prediction=True)
    else:
        pred_map = legacy_list_to_map(pred_raw, data_root, "predictions")

    if isinstance(organ_raw, dict):
        organ_map = organ_coco_to_map(organ_raw)
        organ_has_filter = bool(organ_raw.get("images"))
    else:
        organ_map = {e.get("image_name"): e for e in organ_raw}
        organ_has_filter = bool(organ_raw)

    return gt_map, pred_map, organ_map, organ_has_filter


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', dest='run_id', default='test_run_001')
    parser.add_argument('--organ', dest='organ', default='肾脏')
    parser.add_argument('--data-root', dest='data_root', default='test_data')
    parser.add_argument('--conf-threshold', dest='conf_threshold', type=float, default=0.85)
    parser.add_argument('--iou-threshold', dest='iou_threshold', type=float, default=0.5)
    parser.add_argument(
        '--hard-sample-mode',
        dest='hard_sample_mode',
        choices=['strict', 'all_pred'],
        default='strict',
        help='strict keeps only high-confidence GT/pred mismatches; all_pred also includes GT images with predictions for UI testing.',
    )
    args = parser.parse_args()

    base = Path.cwd()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    test_data = data_root
    organ = args.organ
    organ_dir = test_data / "整理后标注目录" / organ
    if not organ_dir.exists():
        raise SystemExit(f"organ data path not found: {organ_dir}")
    gt_map, pred_map, organ_map, organ_has_filter = load_gt_pred_organ(organ_dir, test_data)
    case_metadata = load_case_metadata(organ_dir)

    run_id = args.run_id
    tasks_dir = base / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    hard_path = tasks_dir / "hard_sample_gt_review.jsonl"
    pseudo_path = tasks_dir / "pseudo_label_review.jsonl"

    conf_thresh = args.conf_threshold
    iou_thresh = args.iou_threshold

    hard_tasks: List[Dict] = []
    pseudo_tasks: List[Dict] = []

    # PAGE 1: iterate GT images
    for image_key, gt_entry in gt_map.items():
        pred_entry = pred_map.get(image_key)
        image_name = gt_entry.get('image_name') or Path(image_key).name
        image_path = gt_entry.get('image_path') or find_image_path(test_data, image_name, gt_entry.get("image_relpath", ""))
        image_meta = gt_entry.get("image", {})
        case_info = case_info_for(image_meta, case_metadata)
        case_id = case_info.get("case_id") or (Path(image_path).parent.name if image_path else "")
        report_image, report_images, related = related_images_for(image_path, image_name, test_data, image_meta, case_metadata)

        if not pred_entry:
            pred_entry = {
                "image_name": image_name,
                "predictions": [],
                "model_version": None,
            }

        gt_annos = gt_entry.get('annotations', [])
        preds = pred_entry.get('predictions', [])
        max_conf = max((p.get('confidence', 0.0) for p in preds), default=0.0)
        is_fully_correct, hard_reasons, qualified_preds = evaluate_gt_prediction(
            gt_annos,
            preds,
            iou_thresh,
        )

        include_for_ui_test = (
            args.hard_sample_mode == 'all_pred'
            and preds
            and max_conf >= conf_thresh
        )

        if (not is_fully_correct) or include_for_ui_test:
            model_version = pred_entry.get('model_version')
            task = {
                "run_id": run_id,
                "task_id": f"{run_id}_{stable_task_suffix('hard', image_meta, image_name)}",
                "case_id": case_id,
                "organ": organ,
                "diagnosis": case_info.get("diagnosis", ""),
                "age_group": case_info.get("age_group", ""),
                "gender": case_info.get("gender", ""),
                "image_name": image_name,
                "image_path": image_path,
                "report_image_path": str((Path(image_path).parent / '病例报告.jpg').as_posix()) if image_path else "",
                "report_image_paths": report_images,
                "related_images": related,
                "gt_annotation": gt_annos,
                "model_prediction": preds,
                "hard_sample_reasons": hard_reasons,
                "qualified_prediction_count": len(qualified_preds),
                "model_version": model_version,
                "model_confidence": max_conf,
                "image_width": gt_entry.get("image_width", 0),
                "image_height": gt_entry.get("image_height", 0),
                "image_relpath": gt_entry.get("image_relpath", image_key),
                "case_relpath": gt_entry.get("case_relpath", ""),
            }
            task["report_image_path"] = report_image
            hard_tasks.append(task)

    # PAGE 2: unlabeled images with preds
    # collect GT image names
    gt_names = set(gt_map.keys())
    for image_key, pred_entry in pred_map.items():
        if image_key in gt_names:
            continue
        preds = pred_entry.get('predictions', [])
        if not preds:
            continue
        # organ filter: if organ_map empty => skip filter; else require organ_present true
        organ_info = organ_map.get(image_key)
        if organ_has_filter and (organ_info is None or not organ_info.get('organ_present', False)):
            continue

        image_name = pred_entry.get('image_name') or Path(image_key).name
        image_meta = pred_entry.get("image", {})
        image_path = pred_entry.get('image_path') or find_image_path(test_data, image_name, pred_entry.get("image_relpath", ""))
        case_info = case_info_for(image_meta, case_metadata)
        case_id = case_info.get("case_id") or (Path(image_path).parent.name if image_path else "")
        report_image, report_images, related = related_images_for(image_path, image_name, test_data, image_meta, case_metadata)

        task = {
            "run_id": run_id,
            "task_id": f"{run_id}_{stable_task_suffix('pseudo', image_meta, image_name)}",
            "case_id": case_id,
            "organ": organ,
            "diagnosis": case_info.get("diagnosis", ""),
            "age_group": case_info.get("age_group", ""),
            "gender": case_info.get("gender", ""),
            "image_name": image_name,
            "image_path": image_path,
            "report_image_path": report_image,
            "report_image_paths": report_images,
            "related_images": related,
            "pseudo_label_prediction": preds,
            "model_version": pred_entry.get('model_version'),
            "model_confidence": max((p.get('confidence', 0.0) for p in preds), default=0.0),
            "image_width": pred_entry.get("image_width", 0),
            "image_height": pred_entry.get("image_height", 0),
            "image_relpath": pred_entry.get("image_relpath", image_key),
            "case_relpath": pred_entry.get("case_relpath", ""),
        }
        pseudo_tasks.append(task)

    # write JSONL files (overwrite)
    with hard_path.open('w', encoding='utf-8') as f:
        for t in hard_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with pseudo_path.open('w', encoding='utf-8') as f:
        for t in pseudo_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Wrote {len(hard_tasks)} hard tasks to {hard_path}")
    print(f"Wrote {len(pseudo_tasks)} pseudo tasks to {pseudo_path}")


if __name__ == '__main__':
    main()
