#!/usr/bin/env python3
"""Adapt Test production-style data into the app's configured test_data layout.

The production sample already provides COCO-style GT/pred/organ JSON files at
Test/. This script keeps that schema, fixes absolute paths for the local
workspace, and writes the files under test_data/整理后标注目录/<organ>/ so the
existing app can bootstrap from the configured data_root.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


SEQUENCE_IMAGE_RE = re.compile(r".+_p\d+_\d+_\d+\.jpe?g$", re.IGNORECASE)
JSON_NAMES = ("GT.json", "pred.json", "organ.json")
REPORT_KEYWORDS_BY_ORGAN = {
    "肾脏": ("泌尿", "肾", "尿"),
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def normalize_coco_paths(coco: Dict[str, Any], source_root: Path, organ: str) -> Dict[str, Any]:
    """Return a copy of a COCO object with image_abspath corrected locally."""
    out = dict(coco)
    images = []
    source_root = source_root.resolve()

    for image in coco.get("images", []):
        item = dict(image)
        relpath = item.get("image_relpath")
        if relpath:
            local_path = (source_root / relpath).resolve()
            item["image_abspath"] = local_path.as_posix()
            item.setdefault("case_relpath", str(Path(relpath).parent).replace("\\", "/"))
        item.setdefault("file_name", item.get("image_name", ""))
        if item.get("file_name"):
            item.setdefault("image_name", item["file_name"])
        item.setdefault("organ", organ)
        images.append(item)

    out["images"] = images
    return out


def iter_case_dirs(source_root: Path) -> Iterable[Path]:
    for path in source_root.rglob("isat.yaml"):
        if path.is_file():
            yield path.parent


def is_relevant_report_image(file_name: str, organ: str) -> bool:
    keywords = REPORT_KEYWORDS_BY_ORGAN.get(organ)
    if not keywords:
        return True
    return any(keyword in file_name for keyword in keywords)


def infer_case_metadata(source_root: Path, organ: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_root": source_root.resolve().as_posix(),
        "organ": organ,
        "cases": {},
    }

    for case_dir in sorted(iter_case_dirs(source_root), key=lambda p: p.as_posix()):
        case_relpath = case_dir.relative_to(source_root).as_posix()
        parts = case_relpath.split("/")
        diagnosis = parts[0] if len(parts) > 0 else ""
        age_group = parts[1] if len(parts) > 1 else ""
        gender = parts[2] if len(parts) > 2 else ""
        case_id = case_dir.name

        jpgs = sorted(p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        report_candidates: List[str] = []
        report_images: List[str] = []
        sequence_images: List[str] = []
        for jpg in jpgs:
            rel = jpg.relative_to(source_root).as_posix()
            if SEQUENCE_IMAGE_RE.match(jpg.name):
                sequence_images.append(rel)
            else:
                report_candidates.append(rel)
                if is_relevant_report_image(jpg.name, organ):
                    report_images.append(rel)

        history = case_dir / "病史.txt"
        metadata["cases"][case_relpath] = {
            "case_id": case_id,
            "case_relpath": case_relpath,
            "case_abspath": case_dir.resolve().as_posix(),
            "diagnosis": diagnosis,
            "age_group": age_group,
            "gender": gender,
            "organ": organ,
            "report_images": report_images,
            "all_context_images": report_candidates,
            "sequence_images": sequence_images,
            "medical_history_path": history.resolve().as_posix() if history.exists() else "",
        }

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Test production data for MedRev")
    parser.add_argument("--source-json-dir", default="Test", help="Directory containing GT.json, pred.json, organ.json")
    parser.add_argument("--source-root", default="Test/raw_subset", help="Raw production image root")
    parser.add_argument("--output-root", default="test_data", help="App data_root to write into")
    parser.add_argument("--organ", default="肾脏")
    args = parser.parse_args()

    base = Path.cwd()
    source_json_dir = (base / args.source_json_dir).resolve()
    source_root = (base / args.source_root).resolve()
    output_dir = (base / args.output_root).resolve() / "整理后标注目录" / args.organ

    if not source_root.exists():
        raise SystemExit(f"source root not found: {source_root}")

    for name in JSON_NAMES:
        src = source_json_dir / name
        if not src.exists():
            raise SystemExit(f"source JSON not found: {src}")
        data = normalize_coco_paths(load_json(src), source_root, args.organ)
        write_json(output_dir / name, data)
        print(f"Wrote {output_dir / name} ({len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations)")

    metadata = infer_case_metadata(source_root, args.organ)
    write_json(output_dir / "case_metadata.json", metadata)
    print(f"Wrote {output_dir / 'case_metadata.json'} ({len(metadata['cases'])} cases)")


if __name__ == "__main__":
    main()
