import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.db as db

CONFIG_FILE = ROOT / 'config.json'
DEFAULT_CONFIG = {
    'run_id': 'test_run_001',
}


def load_config():
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with CONFIG_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()
    merged = DEFAULT_CONFIG.copy()
    merged.update(data if isinstance(data, dict) else {})
    return merged


CONFIG = load_config()
RUN_ID = CONFIG['run_id']
TASKS_DIR = ROOT / 'tasks'
REVIEW_DIR = ROOT / 'review_outputs' / RUN_ID


TASK_FILES = [TASKS_DIR / 'hard_sample_gt_review.jsonl', TASKS_DIR / 'pseudo_label_review.jsonl']


def read_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def build_coco_from_accepted(accepted_rows, task_map):
    categories = {}
    images = {}
    annotations = []
    next_image_id = 1
    next_annotation_id = 1

    def get_category_id(name):
        if name not in categories:
            categories[name] = len(categories) + 1
        return categories[name]

    for row in accepted_rows:
        task = task_map.get(row['task_id'], {})
        image_key = task.get('image_path') or row.get('image_path') or row['task_id']
        if image_key not in images:
            images[image_key] = {
                'id': next_image_id,
                'file_name': task.get('image_name', row.get('image_name', '')),
                'width': task.get('image_width', 0),
                'height': task.get('image_height', 0),
            }
            next_image_id += 1
        image_id = images[image_key]['id']
        preds = task.get('pseudo_label_prediction', []) or []
        for pred in preds:
            bbox = pred.get('bbox') or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            annotations.append({
                'id': next_annotation_id,
                'image_id': image_id,
                'category_id': get_category_id(pred.get('category', 'unknown')),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'area': max(0, (x2 - x1)) * max(0, (y2 - y1)),
                'iscrowd': 0,
            })
            next_annotation_id += 1

    coco = {
        'images': list(images.values()),
        'annotations': annotations,
        'categories': [{'id': cid, 'name': name} for name, cid in sorted(categories.items(), key=lambda item: item[1])],
    }
    return coco


def export_results(run_id: str):
    task_rows = []
    for task_file in TASK_FILES:
        task_rows.extend(read_jsonl(task_file))
    task_map = {row['task_id']: row for row in task_rows if row.get('task_id')}

    raw_reviews = db.get_reviews_for_run(run_id)
    raw_reviews = [dict(r) if hasattr(r, 'keys') else r for r in raw_reviews]

    # Filter out already-exported tasks for incremental export
    exported = db.get_exported_tasks_for_run(run_id)
    raw_reviews = [r for r in raw_reviews if r.get('task_id') not in exported]

    if not raw_reviews:
        return {'run_id': run_id, 'total_reviewed': 0, 'message': '没有新增的审核结果需要导出'}

    # Create timestamped batch directory for this export
    batch_id = datetime.now().strftime('export_%Y%m%d_%H%M%S')
    export_dir = REVIEW_DIR / 'exports' / batch_id

    gt_issue_rows = []
    accepted_rows = []
    pseudo_error_rows = []
    error_counter = Counter()
    exported_task_ids = []

    for row in raw_reviews:
        task = task_map.get(row.get('task_id'), {})
        review_result = row.get('review_result')
        task_id = row.get('task_id')
        if task_id:
            exported_task_ids.append(task_id)
        if review_result == 'gt_error':
            gt_issue_rows.append({
                'run_id': row.get('run_id'),
                'task_id': row.get('task_id'),
                'organ': row.get('organ'),
                'image_name': row.get('image_name'),
                'image_path': row.get('image_path'),
            })
        elif review_result == 'pseudo_label_correct':
            accepted_rows.append({
                'run_id': row.get('run_id'),
                'task_id': row.get('task_id'),
                'organ': row.get('organ'),
                'image_name': row.get('image_name'),
                'image_path': row.get('image_path'),
                'pseudo_label_prediction': task.get('pseudo_label_prediction', []),
            })
        elif review_result == 'pseudo_label_error':
            error_type = row.get('error_type')
            pseudo_error_rows.append({
                'run_id': row.get('run_id'),
                'task_id': row.get('task_id'),
                'organ': row.get('organ'),
                'image_name': row.get('image_name'),
                'image_path': row.get('image_path'),
                'error_type': error_type,
            })
            if error_type:
                error_counter[error_type] += 1

    summary = {
        'run_id': run_id,
        'export_batch_id': batch_id,
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'total_reviewed': len(raw_reviews),
        'gt_error_count': len(gt_issue_rows),
        'accepted_pseudo_label_count': len(accepted_rows),
        'pseudo_label_error_count': len(pseudo_error_rows),
        'error_type_count': {
            'category_error': error_counter.get('category_error', 0),
            'box_size_error': error_counter.get('box_size_error', 0),
            'false_positive': error_counter.get('false_positive', 0),
            'false_negative': error_counter.get('false_negative', 0),
        },
    }

    # Write to batch-specific directory
    write_jsonl(export_dir / 'raw_reviews.jsonl', raw_reviews)
    write_jsonl(export_dir / 'gt_issue_list.jsonl', gt_issue_rows)
    write_jsonl(export_dir / 'accepted_pseudo_labels.jsonl', accepted_rows)
    write_jsonl(export_dir / 'pseudo_label_error_list.jsonl', pseudo_error_rows)
    (export_dir / 'accepted_pseudo_labels.coco.json').write_text(
        json.dumps(build_coco_from_accepted(accepted_rows, task_map), ensure_ascii=False, indent=2),
        encoding='utf-8')
    (export_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Mark tasks as exported in the database
    if exported_task_ids:
        db.mark_tasks_exported(run_id, exported_task_ids, batch_id)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Export MedRev review outputs (incremental)')
    parser.add_argument('--run-id', default=CONFIG['run_id'])
    args = parser.parse_args()
    summary = export_results(args.run_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
