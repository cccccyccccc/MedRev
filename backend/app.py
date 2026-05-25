from flask import Flask, jsonify, request, abort, send_file, make_response, session, redirect, url_for
from pathlib import Path
import json
import subprocess
import shutil
import threading
import sys
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from flask import render_template
from datetime import datetime
from collections import Counter
import string

# Add parent directory to path so we can import backend module
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import backend.db as db

CONFIG_FILE = ROOT / 'config.json'
DEFAULT_CONFIG = {
    'run_id': 'test_run_001',
    'reset_outputs_on_start': True,
    'bootstrap_run_id': 'test_run_001',
    'bootstrap_organ': '肾脏',
    'data_root': 'Test',
    'prepared_data_root': 'test_data',
    'conf_threshold': 0.85,
    'iou_threshold': 0.5,
    'hard_sample_mode': 'strict',
    # Export is intentionally manual: admin export marks reviewed tasks as finalized.
    'auto_export': False,
    'echobox_project_id': None,
    'echobox_data_dir': None,
    'echobox_app_url': 'http://127.0.0.1:8000',
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


def save_config(config: dict):
    """Persist config to disk, keeping only keys that differ from defaults."""
    with CONFIG_FILE.open('w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


CONFIG = load_config()
RUN_ID = CONFIG['run_id']

app = Flask(__name__, template_folder=str(HERE / 'templates'), static_folder=str(HERE / 'static'))
app.secret_key = 'dev-local-secret'

# Initialize SQLite database
db.init_db()

# users file (simple local JSON for prototype)
USERS_FILE = HERE / 'users.json'


def load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def authenticate(username: str, password: str):
    users = load_users()
    info = users.get(username)
    if not info:
        return None
    # plaintext check for prototype
    if info.get('password') == password:
        return info.get('role')
    return None
BASE = ROOT
TASKS_DIR = BASE / "tasks"
REVIEW_DIR = BASE / "review_outputs" / RUN_ID
RAW_REVIEWS = REVIEW_DIR / "raw_reviews.jsonl"
GT_ISSUE_LIST = REVIEW_DIR / "gt_issue_list.jsonl"
ACCEPTED_PSEUDO_LABELS = REVIEW_DIR / "accepted_pseudo_labels.jsonl"
ACCEPTED_PSEUDO_LABELS_COCO = REVIEW_DIR / "accepted_pseudo_labels.coco.json"
PSEUDO_LABEL_ERROR_LIST = REVIEW_DIR / "pseudo_label_error_list.jsonl"
SUMMARY_FILE = REVIEW_DIR / "summary.json"
ASSIGNMENTS_FILE = REVIEW_DIR / "assignments.json"
lock = threading.Lock()
export_lock = threading.Lock()


def initialize_review_outputs():
    if CONFIG.get('reset_outputs_on_start', True) and REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for path in [RAW_REVIEWS, GT_ISSUE_LIST, ACCEPTED_PSEUDO_LABELS, PSEUDO_LABEL_ERROR_LIST]:
        path.write_text('', encoding='utf-8')
    ACCEPTED_PSEUDO_LABELS_COCO.write_text(json.dumps({'images': [], 'annotations': [], 'categories': []}, ensure_ascii=False, indent=2), encoding='utf-8')
    SUMMARY_FILE.write_text(json.dumps({
        'run_id': RUN_ID,
        'total_reviewed': 0,
        'gt_error_count': 0,
        'accepted_pseudo_label_count': 0,
        'pseudo_label_error_count': 0,
        'error_type_count': {
            'category_error': 0,
            'box_size_error': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    # assignments persist which task is assigned to which reviewer
    ASSIGNMENTS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding='utf-8')


def resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def prepare_data_root_for_generation(data_root: str, organ: str) -> str:
    """Return the data_root consumed by generate_tasks.py.

    The admin UI should point to the real data source, such as Test. If that
    source is production-style data, adapt it into the prepared data directory
    before generating tasks.
    """
    selected = resolve_workspace_path(data_root)
    organized_dir = selected / "\u6574\u7406\u540e\u6807\u6ce8\u76ee\u5f55" / organ
    if organized_dir.exists():
        return str(selected)

    has_production_json = all((selected / name).exists() for name in ("GT.json", "pred.json", "organ.json"))
    raw_subset = selected / "raw_subset"
    if has_production_json and raw_subset.exists():
        prepared = resolve_workspace_path(CONFIG.get('prepared_data_root') or 'test_data')
        adapter = ROOT / 'scripts' / 'adapt_production_test_data.py'
        if not adapter.exists():
            raise RuntimeError('production data adapter script not found')

        result = subprocess.run(
            [
                sys.executable,
                str(adapter),
                '--source-json-dir',
                str(selected),
                '--source-root',
                str(raw_subset),
                '--output-root',
                str(prepared),
                '--organ',
                organ,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or 'production data adaptation failed')
        return str(prepared)

    return str(selected)


def bootstrap_test_stage_data():
    """Reset persisted review state and regenerate the initial test task set on startup."""
    if not CONFIG.get('reset_outputs_on_start', True):
        return

    bootstrap_run_id = CONFIG.get('bootstrap_run_id') or CONFIG.get('run_id') or 'test_run_001'
    bootstrap_organ = CONFIG.get('bootstrap_organ') or '肾脏'
    bootstrap_data_root = CONFIG.get('data_root') or 'Test'

    try:
        conn = db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM raw_reviews')
            cursor.execute('DELETE FROM assignments')
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass

    script = ROOT / 'scripts' / 'generate_tasks.py'
    if script.exists():
        try:
            generation_data_root = prepare_data_root_for_generation(bootstrap_data_root, bootstrap_organ)
        except Exception:
            generation_data_root = bootstrap_data_root
        subprocess.run(
            [
                sys.executable,
                str(script),
                '--run-id',
                bootstrap_run_id,
                '--organ',
                bootstrap_organ,
                '--data-root',
                str(generation_data_root),
                '--conf-threshold',
                str(CONFIG.get('conf_threshold', 0.85)),
                '--iou-threshold',
                str(CONFIG.get('iou_threshold', 0.5)),
                '--hard-sample-mode',
                str(CONFIG.get('hard_sample_mode', 'strict')),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    initialize_review_outputs_for_run(bootstrap_run_id, persist_config=True)


def update_paths_for_run(new_run_id: str):
    """Update global paths to point to a different run_id at runtime."""
    global RUN_ID, REVIEW_DIR, RAW_REVIEWS, GT_ISSUE_LIST, ACCEPTED_PSEUDO_LABELS, ACCEPTED_PSEUDO_LABELS_COCO, PSEUDO_LABEL_ERROR_LIST, SUMMARY_FILE, ASSIGNMENTS_FILE, CONFIG
    RUN_ID = new_run_id
    CONFIG['run_id'] = new_run_id
    REVIEW_DIR = BASE / "review_outputs" / RUN_ID
    RAW_REVIEWS = REVIEW_DIR / "raw_reviews.jsonl"
    GT_ISSUE_LIST = REVIEW_DIR / "gt_issue_list.jsonl"
    ACCEPTED_PSEUDO_LABELS = REVIEW_DIR / "accepted_pseudo_labels.jsonl"
    ACCEPTED_PSEUDO_LABELS_COCO = REVIEW_DIR / "accepted_pseudo_labels.coco.json"
    PSEUDO_LABEL_ERROR_LIST = REVIEW_DIR / "pseudo_label_error_list.jsonl"
    SUMMARY_FILE = REVIEW_DIR / "summary.json"
    ASSIGNMENTS_FILE = REVIEW_DIR / "assignments.json"


def initialize_review_outputs_for_run(run_id: str, persist_config: bool = True):
    """Create review_outputs/<run_id> and initialize files. Optionally persist to config.json."""
    update_paths_for_run(run_id)
    # persist run_id to config.json so it survives restart
    if persist_config:
        try:
            with CONFIG_FILE.open('w', encoding='utf-8') as f:
                json.dump(CONFIG, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for path in [RAW_REVIEWS, GT_ISSUE_LIST, ACCEPTED_PSEUDO_LABELS, PSEUDO_LABEL_ERROR_LIST]:
        path.write_text('', encoding='utf-8')
    ACCEPTED_PSEUDO_LABELS_COCO.write_text(json.dumps({'images': [], 'annotations': [], 'categories': []}, ensure_ascii=False, indent=2), encoding='utf-8')
    SUMMARY_FILE.write_text(json.dumps({
        'run_id': RUN_ID,
        'total_reviewed': 0,
        'gt_error_count': 0,
        'accepted_pseudo_label_count': 0,
        'pseudo_label_error_count': 0,
        'error_type_count': {
            'category_error': 0,
            'box_size_error': 0,
            'false_positive': 0,
            'false_negative': 0,
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if not ASSIGNMENTS_FILE.exists():
        ASSIGNMENTS_FILE.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding='utf-8')


initialize_review_outputs()
bootstrap_test_stage_data()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_assignments():
    """Load assignments from database for current run."""
    return db.get_assignments_for_run(RUN_ID)


def save_assignments(data: dict):
    """Save assignments to database for current run."""
    db.save_assignments_dict(RUN_ID, data)


def read_tasks(filename: str):
    p = TASKS_DIR / filename
    return read_jsonl(p)


def get_task_type_meta(task_list_type: str):
    if task_list_type == 'hard':
        return 'hard_sample_gt_review.jsonl', 'hard_sample_gt_review'
    if task_list_type == 'pseudo':
        return 'pseudo_label_review.jsonl', 'pseudo_label_review'
    return None, None


def get_reviewed_task_ids(task_type: str = None):
    """Get set of reviewed task IDs from database."""
    reviewed = set()
    for row in db.get_latest_reviews_for_run(RUN_ID).values():
        if task_type and row.get('task_type') != task_type:
            continue
        task_id = row.get('task_id')
        if task_id:
            reviewed.add(task_id)
    return reviewed


def get_reviewed_task_reviewers(task_type: str = None):
    """Get task_id -> reviewer_id for reviewed tasks in the current run."""
    reviewed = {}
    for row in db.get_latest_reviews_for_run(RUN_ID).values():
        if task_type and row.get('task_type') != task_type:
            continue
        task_id = row.get('task_id')
        reviewer_id = row.get('reviewer_id')
        if task_id and reviewer_id:
            reviewed[task_id] = reviewer_id
    return reviewed


def get_latest_task_reviews(task_type: str = None):
    reviews = db.get_latest_reviews_for_run(RUN_ID)
    if not task_type:
        return reviews
    return {task_id: row for task_id, row in reviews.items() if row.get('task_type') == task_type}


def get_exported_task_ids():
    return set(db.get_exported_tasks_for_run(RUN_ID).keys())


def get_assigned_task_ids_for_user(user: str, task_type: str = None):
    if not user:
        return set()
    assignments = load_assignments()
    assigned = set()
    marker = None
    if task_type == 'hard_sample_gt_review':
        marker = '_hard_'
    elif task_type == 'pseudo_label_review':
        marker = '_pseudo_'

    for task_id, assignee in assignments.items():
        if assignee != user:
            continue
        if marker and marker not in task_id:
            continue
        assigned.add(task_id)
    return assigned


def get_task_kind(task_id: str):
    if '_hard_' in task_id:
        return 'hard'
    if '_pseudo_' in task_id:
        return 'pseudo'
    return 'unknown'


def summarize_reviews():
    raw_reviews = list(db.get_latest_reviews_for_run(RUN_ID).values())
    gt_issue_count = len(read_jsonl(GT_ISSUE_LIST))
    accepted_count = len(read_jsonl(ACCEPTED_PSEUDO_LABELS))
    pseudo_error_count = len(read_jsonl(PSEUDO_LABEL_ERROR_LIST))
    error_types = {
        'category_error': 0,
        'box_size_error': 0,
        'false_positive': 0,
        'false_negative': 0,
    }
    doctor_counts = {}
    task_counts = {'hard': 0, 'pseudo': 0}
    review_result_counts = {
        'gt_correct': 0,
        'gt_error': 0,
        'pseudo_label_correct': 0,
        'pseudo_label_error': 0,
    }
    model_versions = {}
    recent_reviews = []

    for row in raw_reviews:
        reviewer = row.get('reviewer_id') or 'unknown'
        doctor_counts[reviewer] = doctor_counts.get(reviewer, 0) + 1
        task_kind = get_task_kind(row.get('task_id', ''))
        if task_kind in task_counts:
            task_counts[task_kind] += 1
        result = row.get('review_result')
        if result in review_result_counts:
            review_result_counts[result] += 1
        error_type = row.get('error_type')
        if error_type in error_types:
            error_types[error_type] += 1
        recent_reviews.append({
            'task_id': row.get('task_id'),
            'task_type': row.get('task_type'),
            'review_result': row.get('review_result'),
            'error_type': row.get('error_type'),
            'reviewer_id': reviewer,
            'submitted_at': row.get('submitted_at'),
        })

    for task in read_tasks('hard_sample_gt_review.jsonl') + read_tasks('pseudo_label_review.jsonl'):
        version = task.get('model_version') or 'unknown'
        model_versions[version] = model_versions.get(version, 0) + 1

    recent_reviews = list(reversed(recent_reviews))[:12]

    assignments = load_assignments()
    doctors = load_users()
    doctor_summaries = []
    
    # Convert raw_reviews to list of dicts for easier processing
    reviews_list = [dict(r) if hasattr(r, 'keys') else r for r in raw_reviews]
    
    for username, info in doctors.items():
        if info.get('role') != 'doctor':
            continue
        hard_assigned = get_assigned_task_ids_for_user(username, task_type='hard_sample_gt_review')
        pseudo_assigned = get_assigned_task_ids_for_user(username, task_type='pseudo_label_review')
        hard_total = len(hard_assigned)
        pseudo_total = len(pseudo_assigned)
        hard_done = len([r for r in reviews_list if r.get('reviewer_id') == username and get_task_kind(r.get('task_id', '')) == 'hard'])
        pseudo_done = len([r for r in reviews_list if r.get('reviewer_id') == username and get_task_kind(r.get('task_id', '')) == 'pseudo'])
        doctor_summaries.append({
            'user_id': username,
            'assigned_total': hard_total + pseudo_total,
            'assigned_hard': hard_total,
            'assigned_pseudo': pseudo_total,
            'reviewed_total': hard_done + pseudo_done,
            'remaining_total': (hard_total + pseudo_total) - (hard_done + pseudo_done),
        })

    hard_tasks = read_tasks('hard_sample_gt_review.jsonl')
    pseudo_tasks = read_tasks('pseudo_label_review.jsonl')
    hard_reviewed = get_reviewed_task_ids(task_type='hard_sample_gt_review')
    pseudo_reviewed = get_reviewed_task_ids(task_type='pseudo_label_review')
    exported = get_exported_task_ids()
    hard_exported = {task_id for task_id in exported if task_id and '_hard_' in task_id}
    pseudo_exported = {task_id for task_id in exported if task_id and '_pseudo_' in task_id}
    hard_remaining = len([t for t in hard_tasks if t.get('task_id') not in exported])
    pseudo_remaining = len([t for t in pseudo_tasks if t.get('task_id') not in exported])

    return {
        'run_id': RUN_ID,
        'config': {
            'conf_threshold': CONFIG.get('conf_threshold'),
            'iou_threshold': CONFIG.get('iou_threshold'),
            'hard_sample_mode': CONFIG.get('hard_sample_mode', 'strict'),
            'reset_outputs_on_start': CONFIG.get('reset_outputs_on_start', True),
            'data_root': CONFIG.get('data_root', 'Test'),
            'prepared_data_root': CONFIG.get('prepared_data_root', 'test_data'),
        },
        'tasks': {
            'hard': {'total': len(hard_tasks), 'reviewed': len(hard_reviewed), 'exported': len(hard_exported), 'remaining': hard_remaining},
            'pseudo': {'total': len(pseudo_tasks), 'reviewed': len(pseudo_reviewed), 'exported': len(pseudo_exported), 'remaining': pseudo_remaining},
            'total': len(hard_tasks) + len(pseudo_tasks),
            'reviewed_total': len(raw_reviews),
            'exported_total': len(exported),
            'remaining_total': hard_remaining + pseudo_remaining,
            'hard_reviewed_ids': sorted(list(hard_reviewed)),
            'pseudo_reviewed_ids': sorted(list(pseudo_reviewed)),
        },
        'reviews': {
            'raw_total': len(raw_reviews),
            'gt_issue_count': gt_issue_count,
            'accepted_pseudo_label_count': accepted_count,
            'pseudo_error_count': pseudo_error_count,
            'result_counts': review_result_counts,
            'error_type_counts': error_types,
        },
        'doctors': sorted(doctor_summaries, key=lambda item: item['user_id']),
        'doctor_review_counts': doctor_counts,
        'model_versions': model_versions,
        'recent_reviews': recent_reviews,
        'assignments': assignments,
    }


def find_task(task_id: str):
    for t in read_tasks('hard_sample_gt_review.jsonl') + read_tasks('pseudo_label_review.jsonl'):
        if t.get('task_id') == task_id:
            return t
    return None


def send_local_image(image_path: str):
    if not image_path:
        return jsonify({'error': 'image not found'}), 404
    p = Path(image_path)
    if not p.exists():
        return jsonify({'error': 'image not found'}), 404
    return send_file(p)


def render_boxes_on_image(image_path: str, gt_boxes=None, pred_boxes=None):
    gt_boxes = gt_boxes or []
    pred_boxes = pred_boxes or []
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    img = Image.open(p).convert('RGB')
    draw = ImageDraw.Draw(img)
    # prefer a Chinese-capable TrueType font; fall back to Latin fonts, then default
    font = None
    for font_name in (
        "msyh.ttc",     # 微软雅黑
        "simhei.ttf",   # 黑体
        "Deng.ttf",     # 等线
        "simsun.ttc",   # 宋体
        "arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(font_name, size=18)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # draw GT in green
    for g in gt_boxes:
        bbox = g.get('bbox')
        label = g.get('category', '')
        if bbox and len(bbox) >= 4:
            x0, y0, x1, y1 = map(int, bbox[:4])
            draw.rectangle([x0, y0, x1, y1], outline='green', width=3)
            text = f"GT:{label}"
            # background box for text
            try:
                tw, th = font.getsize(text)
            except Exception:
                # fallback
                tw, th = draw.textbbox((0,0), text, font=font)[2:4]
            bx0, by0 = x0 + 3, y0 + 3
            bx1, by1 = bx0 + tw + 6, by0 + th + 4
            draw.rectangle([bx0, by0, bx1, by1], fill='black')
            draw.text((bx0 + 3, by0 + 2), text, fill='white', font=font)

    # draw preds in red
    for pbox in pred_boxes:
        bbox = pbox.get('bbox')
        label = pbox.get('category', '')
        conf = pbox.get('confidence')
        if bbox and len(bbox) >= 4:
            x0, y0, x1, y1 = map(int, bbox[:4])
            draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
            text = f"P:{label} {conf:.2f}" if conf is not None else f"P:{label}"
            try:
                tw, th = font.getsize(text)
            except Exception:
                tw, th = draw.textbbox((0,0), text, font=font)[2:4]
            bx0, by0 = x0 + 3, y1 - th - 7
            if by0 < 0:
                by0 = y0 + 3
            bx1, by1 = bx0 + tw + 6, by0 + th + 4
            draw.rectangle([bx0, by0, bx1, by1], fill='black')
            draw.text((bx0 + 3, by0 + 2), text, fill='white', font=font)

    bio = BytesIO()
    img.save(bio, format='JPEG', quality=90)
    bio.seek(0)
    return bio


@app.route('/tasks/hard', methods=['GET'])
def get_hard_tasks():
    return jsonify(read_tasks('hard_sample_gt_review.jsonl'))


@app.route('/tasks/pseudo', methods=['GET'])
def get_pseudo_tasks():
    return jsonify(read_tasks('pseudo_label_review.jsonl'))


@app.route('/tasks/next', methods=['GET'])
def get_next_task():
    task_list_type = request.args.get('type', 'hard')
    filename, task_type = get_task_type_meta(task_list_type)
    if not filename:
        return jsonify({'error': 'invalid task type'}), 400

    exported = get_exported_task_ids()
    # if logged-in doctor, only return tasks assigned to them
    user = session.get('user')
    role = session.get('role')
    assigned_to_user = set()
    if user and role == 'doctor':
        assigned_to_user = get_assigned_task_ids_for_user(user, task_type=task_type)
    for task in read_tasks(filename):
        task_id = task.get('task_id')
        if not task_id or task_id in exported:
            continue
        # if doctor, only return assigned tasks
        if user and role == 'doctor' and task_id not in assigned_to_user:
            continue
        return jsonify(task)
    return jsonify({'error': 'no available task'}), 404


@app.route('/tasks/my-next', methods=['GET'])
def get_my_next_task():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'doctor':
        return jsonify({'error': 'doctor login required'}), 403

    task_list_type = request.args.get('type', 'hard')
    filename, task_type = get_task_type_meta(task_list_type)
    if not filename:
        return jsonify({'error': 'invalid task type'}), 400

    skip_task_id = request.args.get('skip')
    exported = get_exported_task_ids()
    assigned_to_user = get_assigned_task_ids_for_user(user, task_type=task_type)
    for task in read_tasks(filename):
        task_id = task.get('task_id')
        if not task_id or task_id in exported:
            continue
        if skip_task_id and task_id == skip_task_id:
            continue
        if task_id not in assigned_to_user:
            continue
        return jsonify(task)
    return jsonify({'error': 'no available task'}), 404


@app.route('/tasks/my-list', methods=['GET'])
def get_my_task_list():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'doctor':
        return jsonify({'error': 'doctor login required'}), 403

    task_list_type = request.args.get('type', 'hard')
    filename, task_type = get_task_type_meta(task_list_type)
    if not filename:
        return jsonify({'error': 'invalid task type'}), 400

    reviewed = get_reviewed_task_ids(task_type=task_type)
    latest_reviews = get_latest_task_reviews(task_type=task_type)
    exported = get_exported_task_ids()
    assigned_to_user = get_assigned_task_ids_for_user(user, task_type=task_type)
    items = []
    for task in read_tasks(filename):
        task_id = task.get('task_id')
        if not task_id or task_id not in assigned_to_user or task_id in exported:
            continue
        review = latest_reviews.get(task_id) or {}
        items.append({
            'task_id': task_id,
            'case_id': task.get('case_id'),
            'image_name': task.get('image_name'),
            'organ': task.get('organ'),
            'model_version': task.get('model_version'),
            'status': 'reviewed' if task_id in reviewed else 'assigned',
            'review_result': review.get('review_result'),
            'error_type': review.get('error_type'),
        })

    return jsonify({'items': items})


@app.route('/tasks/remaining', methods=['GET'])
def get_remaining_stats():
    hard_tasks = read_tasks('hard_sample_gt_review.jsonl')
    pseudo_tasks = read_tasks('pseudo_label_review.jsonl')

    hard_reviewed = get_reviewed_task_ids(task_type='hard_sample_gt_review')
    pseudo_reviewed = get_reviewed_task_ids(task_type='pseudo_label_review')

    user = session.get('user')
    role = session.get('role')
    # For doctors, show only their assigned workload (not global pool counts).
    if user and role == 'doctor':
        hard_assigned = get_assigned_task_ids_for_user(user, task_type='hard_sample_gt_review')
        pseudo_assigned = get_assigned_task_ids_for_user(user, task_type='pseudo_label_review')
        hard_total = len([t for t in hard_tasks if t.get('task_id') in hard_assigned])
        pseudo_total = len([t for t in pseudo_tasks if t.get('task_id') in pseudo_assigned])
        hard_remaining = len([
            t for t in hard_tasks
            if t.get('task_id') in hard_assigned and t.get('task_id') not in hard_reviewed and t.get('task_id') not in exported
        ])
        pseudo_remaining = len([
            t for t in pseudo_tasks
            if t.get('task_id') in pseudo_assigned and t.get('task_id') not in pseudo_reviewed and t.get('task_id') not in exported
        ])
    else:
        hard_total = len(hard_tasks)
        pseudo_total = len(pseudo_tasks)
        hard_remaining = len([t for t in hard_tasks if t.get('task_id') not in hard_reviewed and t.get('task_id') not in exported])
        pseudo_remaining = len([t for t in pseudo_tasks if t.get('task_id') not in pseudo_reviewed and t.get('task_id') not in exported])

    return jsonify({
        'hard': {
            'total': hard_total,
            'reviewed': len([t for t in hard_tasks if t.get('task_id') in hard_assigned and t.get('task_id') in hard_reviewed]) if user and role == 'doctor' else len(hard_reviewed),
            'remaining': hard_remaining,
        },
        'pseudo': {
            'total': pseudo_total,
            'reviewed': len([t for t in pseudo_tasks if t.get('task_id') in pseudo_assigned and t.get('task_id') in pseudo_reviewed]) if user and role == 'doctor' else len(pseudo_reviewed),
            'remaining': pseudo_remaining,
        },
        'total_remaining': hard_remaining + pseudo_remaining,
    })


@app.route('/tasks/my-remaining', methods=['GET'])
def get_my_remaining_stats():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'doctor':
        return jsonify({'error': 'doctor login required'}), 403

    hard_tasks = read_tasks('hard_sample_gt_review.jsonl')
    pseudo_tasks = read_tasks('pseudo_label_review.jsonl')
    hard_reviewed = get_reviewed_task_ids(task_type='hard_sample_gt_review')
    pseudo_reviewed = get_reviewed_task_ids(task_type='pseudo_label_review')
    exported = get_exported_task_ids()

    hard_assigned = get_assigned_task_ids_for_user(user, task_type='hard_sample_gt_review')
    pseudo_assigned = get_assigned_task_ids_for_user(user, task_type='pseudo_label_review')

    hard_total = len([t for t in hard_tasks if t.get('task_id') in hard_assigned])
    pseudo_total = len([t for t in pseudo_tasks if t.get('task_id') in pseudo_assigned])
    hard_remaining = len([
        t for t in hard_tasks
        if t.get('task_id') in hard_assigned and t.get('task_id') not in hard_reviewed and t.get('task_id') not in exported
    ])
    pseudo_remaining = len([
        t for t in pseudo_tasks
        if t.get('task_id') in pseudo_assigned and t.get('task_id') not in pseudo_reviewed and t.get('task_id') not in exported
    ])

    return jsonify({
        'hard': {
            'total': hard_total,
            'reviewed': len([t for t in hard_tasks if t.get('task_id') in hard_assigned and t.get('task_id') in hard_reviewed]),
            'remaining': hard_remaining,
        },
        'pseudo': {
            'total': pseudo_total,
            'reviewed': len([t for t in pseudo_tasks if t.get('task_id') in pseudo_assigned and t.get('task_id') in pseudo_reviewed]),
            'remaining': pseudo_remaining,
        },
        'total_remaining': hard_remaining + pseudo_remaining,
    })


@app.route('/task/<task_id>', methods=['GET'])
def get_task(task_id):
    # search both lists
    task = find_task(task_id)
    if task:
        exported = db.get_exported_tasks_for_run(RUN_ID).get(task_id)
        if exported:
            if session.get('role') == 'doctor':
                abort(404)
            task['status'] = 'exported'
            task['export_batch_id'] = exported.get('export_batch_id')
        else:
            review = db.get_latest_reviews_for_run(RUN_ID).get(task_id)
            if review:
                task['status'] = 'reviewed'
                task['last_review'] = {
                    'review_result': review.get('review_result'),
                    'error_type': review.get('error_type'),
                    'reviewer_id': review.get('reviewer_id'),
                    'submitted_at': review.get('submitted_at'),
                }
            else:
                task['status'] = 'assigned' if task_id in load_assignments() else 'unassigned'
        return jsonify(task)
    abort(404)


@app.route('/task/<task_id>/asset/<kind>', methods=['GET'])
def get_task_asset(task_id, kind):
    task = find_task(task_id)
    if not task:
        abort(404)
    if session.get('role') == 'doctor' and task_id in get_exported_task_ids():
        abort(404)

    if kind == 'current':
        return send_local_image(task.get('image_path'))
    if kind == 'report':
        index = int(request.args.get('index', '0'))
        report_images = task.get('report_image_paths') or []
        if report_images:
            if index < 0 or index >= len(report_images):
                abort(404)
            return send_local_image(report_images[index])
        if index > 0:
            abort(404)
        return send_local_image(task.get('report_image_path'))
    if kind == 'related':
        index = int(request.args.get('index', '0'))
        related_images = task.get('related_images') or []
        if index < 0 or index >= len(related_images):
            abort(404)
        return send_local_image(related_images[index])
    abort(404)


# ── echobox integration ────────────────────────────────────────────


def _echobox_api(method: str, path: str, body: dict | None = None) -> tuple:
    """Call echobox REST API. Returns (status_code, response_json_dict)."""
    import urllib.request

    url = f"{CONFIG.get('echobox_app_url', 'http://127.0.0.1:8000')}{path}"
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode('utf-8'))
        except Exception:
            detail = {'error': exc.reason}
        return exc.code, detail
    except Exception as exc:
        return 0, {'error': str(exc)}


_echobox_image_cache: dict[str, int] = {}


def _discover_labels() -> list[str]:
    """Extract label names from MedRev data files."""
    data_root = CONFIG.get('data_root', 'Test')
    data_dir = ROOT / data_root

    # Try GT.json first, then pred.json
    for fname in ('GT.json', 'pred.json'):
        fp = data_dir / fname
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
            cats = data.get('categories', [])
            if cats:
                return [c['name'] for c in cats]
        except Exception:
            pass
    return []


def _echobox_setup_project() -> tuple:
    """Create an echobox project for the current MedRev run. Idempotent."""
    data_root = CONFIG.get('data_root', 'Test')
    abs_root = str((ROOT / data_root).resolve())
    labels = _discover_labels()

    # 1. create project
    code, proj = _echobox_api('POST', '/api/projects', {
        'source_folder': abs_root,
        'name': f'MedRev-{RUN_ID}',
        'initial_labels': labels,
        'train_val_test': [1.0, 0.0, 0.0],
        'export_format': 'coco',
    })
    if code not in (200, 201) or not proj.get('id'):
        return code, {'error': 'failed to create echobox project', 'detail': proj}

    project_id = proj['id']

    # 2. finalize (scans images)
    code, _ = _echobox_api('POST', f'/api/projects/{project_id}/finalize')
    if code not in (200, 201, 204):
        return code, {'error': 'echobox finalize failed'}

    # 3. persist
    CONFIG['echobox_project_id'] = project_id
    CONFIG['echobox_data_dir'] = abs_root
    save_config(CONFIG)
    _echobox_image_cache.clear()

    return 200, {'project_id': project_id, 'data_dir': abs_root}


@app.route('/task/<task_id>/echobox-mapping', methods=['GET'])
def echobox_mapping(task_id):
    """Return {project_id, image_id} so the frontend can load the echobox iframe."""
    task = find_task(task_id)
    if not task:
        abort(404)

    project_id = CONFIG.get('echobox_project_id')
    if not project_id:
        # auto-setup on first use
        code, result = _echobox_setup_project()
        if code not in (200, 201):
            return jsonify({'error': 'echobox project setup failed', 'detail': result}), 503
        project_id = result['project_id']

    image_path = task.get('image_path', '')
    if not image_path:
        return jsonify({'error': 'task has no image_path'}), 400

    cache_key = f"{project_id}:{image_path}"
    if cache_key in _echobox_image_cache:
        return jsonify({'project_id': project_id, 'image_id': _echobox_image_cache[cache_key]})

    norm = image_path.replace('\\', '/')
    import urllib.parse
    encoded = urllib.parse.quote(norm, safe='')
    code, payload = _echobox_api('GET', f'/api/projects/{project_id}/lookup-by-path?abs_path={encoded}')
    if code == 200 and payload.get('image_id'):
        _echobox_image_cache[cache_key] = payload['image_id']
        return jsonify({'project_id': project_id, 'image_id': payload['image_id']})

    return jsonify({'project_id': project_id, 'image_id': None, 'error': 'image not in echobox index'})


@app.route('/admin/setup-echobox', methods=['POST'])
def setup_echobox():
    """Create an echobox project for the current MedRev run (admin only)."""
    role = session.get('role')
    if role != 'admin':
        return jsonify({'error': 'admin required'}), 403
    code, result = _echobox_setup_project()
    if code not in (200, 201):
        return jsonify(result), 500
    return jsonify({'status': 'ok', **result})


@app.route('/admin/echobox-status', methods=['GET'])
def echobox_status():
    """Check whether echobox is running and a project exists for this run."""
    project_id = CONFIG.get('echobox_project_id')
    healthy = False
    if project_id:
        code, _ = _echobox_api('GET', f'/api/projects/{project_id}')
        healthy = code == 200
    return jsonify({
        'echobox_project_id': project_id,
        'echobox_healthy': healthy,
        'echobox_app_url': CONFIG.get('echobox_app_url', 'http://127.0.0.1:8000'),
    })


@app.route('/review', methods=['POST'])
def post_review():
    data = request.get_json(force=True)
    required = ['run_id', 'task_id', 'task_type', 'organ', 'image_name', 'image_path', 'review_result', 'reviewer_id', 'submitted_at']
    if not all(k in data for k in required):
        return jsonify({'error': 'missing fields'}), 400
    # enforce logged-in reviewer
    user = session.get('user')
    if not user:
        return jsonify({'error': 'login required'}), 403
    # always bind review to current runtime run_id
    data['run_id'] = RUN_ID
    data['reviewer_id'] = user
    if data['task_id'] in get_exported_task_ids():
        return jsonify({'error': 'task already exported'}), 409

    # serialize annotations from echobox if present
    annotations = data.pop('annotations', None)
    if annotations and isinstance(annotations, list):
        data['annotation_json'] = json.dumps(annotations, ensure_ascii=False)

    # Save to database first, then mirror to JSONL so a failure can be reported cleanly.
    if not db.append_review(data):
        return jsonify({'error': 'failed to save review'}), 500

    line = json.dumps(data, ensure_ascii=False)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with lock:
            with RAW_REVIEWS.open('a', encoding='utf-8') as f:
                f.write(line + "\n")
    except Exception as exc:
        # Best effort rollback for the database row so retry semantics stay clean.
        try:
            db.delete_review(data['run_id'], data['task_id'], data['reviewer_id'], data['submitted_at'])
        except Exception:
            pass
        return jsonify({'error': f'failed to save review jsonl: {exc}'}), 500
    return jsonify({'status': 'ok'})


def run_export_background(run_id: str):
    """Run the export script in background, skip if another export is running."""
    if not export_lock.acquire(blocking=False):
        # another export already running; skip this run
        return
    try:
        script = ROOT / 'scripts' / 'export_review_results.py'
        if not script.exists():
            return
        result = subprocess.run([sys.executable, str(script), '--run-id', run_id], cwd=str(ROOT), capture_output=True, text=True)
        # write last export result for debugging
        try:
            REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            last = REVIEW_DIR / 'last_export.json'
            payload = {
                'run_id': run_id,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
            last.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass
    finally:
        export_lock.release()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def build_incremental_export(run_id: str):
    task_rows = read_tasks('hard_sample_gt_review.jsonl') + read_tasks('pseudo_label_review.jsonl')
    task_map = {row.get('task_id'): row for row in task_rows if row.get('task_id')}
    latest_reviews = db.get_latest_reviews_for_run(run_id)
    exported = set(db.get_exported_tasks_for_run(run_id).keys())
    rows = [row for task_id, row in latest_reviews.items() if task_id not in exported]

    batch_id = datetime.now().strftime('export_%Y%m%d_%H%M%S')
    export_dir = REVIEW_DIR / 'exports' / batch_id
    raw_rows = []
    gt_issue_rows = []
    accepted_rows = []
    pseudo_error_rows = []
    error_counter = Counter()

    for row in rows:
        task_id = row.get('task_id')
        task = task_map.get(task_id, {})
        raw_rows.append(row)
        review_result = row.get('review_result')
        if review_result == 'gt_error':
            gt_issue_rows.append({
                'run_id': row.get('run_id'),
                'task_id': task_id,
                'organ': row.get('organ'),
                'image_name': row.get('image_name'),
                'image_path': row.get('image_path'),
            })
        elif review_result == 'pseudo_label_correct':
            accepted_rows.append({
                'run_id': row.get('run_id'),
                'task_id': task_id,
                'organ': row.get('organ'),
                'image_name': row.get('image_name'),
                'image_path': row.get('image_path'),
                'pseudo_label_prediction': task.get('pseudo_label_prediction', []),
            })
        elif review_result == 'pseudo_label_error':
            error_type = row.get('error_type')
            pseudo_error_rows.append({
                'run_id': row.get('run_id'),
                'task_id': task_id,
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
        'exported_task_count': len(raw_rows),
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

    write_jsonl(export_dir / 'raw_reviews.jsonl', raw_rows)
    write_jsonl(export_dir / 'gt_issue_list.jsonl', gt_issue_rows)
    write_jsonl(export_dir / 'accepted_pseudo_labels.jsonl', accepted_rows)
    write_jsonl(export_dir / 'pseudo_label_error_list.jsonl', pseudo_error_rows)
    (export_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return batch_id, export_dir, summary, [row.get('task_id') for row in raw_rows if row.get('task_id')]


@app.route('/render', methods=['GET'])
def render_task_image():
    # params: task_id or image_path; show: 'both'|'gt'|'pred'
    task_id = request.args.get('task_id')
    image_path = request.args.get('image_path')
    show = request.args.get('show', 'both')

    task = None
    if task_id:
        for t in read_tasks('hard_sample_gt_review.jsonl') + read_tasks('pseudo_label_review.jsonl'):
            if t.get('task_id') == task_id:
                task = t
                break

    if task:
        image_path = image_path or task.get('image_path')
        gt_boxes = task.get('gt_annotation') if show in ('both', 'gt') else []
        pred_key = 'model_prediction' if 'model_prediction' in task else 'pseudo_label_prediction'
        pred_boxes = task.get(pred_key, []) if show in ('both', 'pred') else []
    else:
        gt_boxes = []
        pred_boxes = []

    bio = render_boxes_on_image(image_path, gt_boxes=gt_boxes, pred_boxes=pred_boxes)
    if bio is None:
        return jsonify({'error': 'image not found'}), 404
    return send_file(bio, mimetype='image/jpeg')


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/doctor')
def doctor_page():
    user = session.get('user')
    role = session.get('role')
    if not user:
        return redirect(url_for('login'))
    if role != 'doctor':
        if role == 'admin':
            return redirect(url_for('admin_page'))
        return redirect(url_for('login'))
    return render_template('doctor.html', reviewer_id=user)


@app.route('/admin')
def admin_page():
    user = session.get('user')
    role = session.get('role')
    if not user:
        return redirect(url_for('login', next='/admin', msg='admin_required'))
    if role != 'admin':
        return redirect(url_for('login', next='/admin', msg='admin_required'))
    return render_template('admin.html')


@app.route('/admin/settings')
def admin_settings_page():
    user = session.get('user')
    role = session.get('role')
    if not user:
        return redirect(url_for('login', next='/admin/settings', msg='admin_required'))
    if role != 'admin':
        return redirect(url_for('login', next='/admin/settings', msg='admin_required'))
    return render_template('admin_settings.html')


@app.route('/admin/results')
def admin_results_page():
    user = session.get('user')
    role = session.get('role')
    if not user:
        return redirect(url_for('login', next='/admin/results', msg='admin_required'))
    if role != 'admin':
        return redirect(url_for('login', next='/admin/results', msg='admin_required'))
    return render_template('admin_results.html')


@app.route('/_admin/users')
def admin_users():
    # return list of doctors for assignment
    users = load_users()
    doctors = [u for u, info in users.items() if info.get('role') == 'doctor']
    return jsonify({'doctors': doctors})


@app.route('/admin/fs/roots', methods=['GET'])
def admin_fs_roots():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403

    roots = []
    if os.name == 'nt':
        for drive in string.ascii_uppercase:
            drive_path = Path(f'{drive}:/')
            if drive_path.exists():
                roots.append(str(drive_path))
    else:
        roots.append('/')

    return jsonify({'roots': roots})


@app.route('/admin/fs/list', methods=['GET'])
def admin_fs_list():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403

    raw_path = (request.args.get('path') or '').strip()
    if not raw_path:
        return jsonify({'path': '', 'parent': None, 'dirs': []})

    try:
        current = Path(raw_path).expanduser().resolve()
    except Exception:
        return jsonify({'error': 'invalid path'}), 400

    if not current.exists() or not current.is_dir():
        return jsonify({'error': 'directory not found'}), 404

    parent = str(current.parent) if current.parent != current else None
    dirs = []
    try:
        for child in sorted(current.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                dirs.append({'name': child.name, 'path': str(child)})
    except PermissionError:
        return jsonify({'error': 'permission denied'}), 403

    return jsonify({'path': str(current), 'parent': parent, 'dirs': dirs})


@app.route('/admin/tasks', methods=['GET'])
def admin_tasks():
    task_list_type = request.args.get('type', 'hard')
    filename, task_type = get_task_type_meta(task_list_type)
    if not filename:
        return jsonify({'error': 'invalid task type'}), 400
    assignments = load_assignments()
    reviewed = get_reviewed_task_ids(task_type=task_type)
    reviewed_by = get_reviewed_task_reviewers(task_type=task_type)
    exported = get_exported_task_ids()
    out = []
    for t in read_tasks(filename):
        tid = t.get('task_id')
        t['assigned_to'] = assignments.get(tid)
        t['reviewed_by'] = reviewed_by.get(tid)
        t['status'] = 'exported' if tid in exported else ('reviewed' if tid in reviewed else ('assigned' if tid in assignments else 'unassigned'))
        out.append(t)
    return jsonify(out)


@app.route('/admin/summary', methods=['GET'])
def admin_summary():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403
    return jsonify(summarize_reviews())


@app.route('/admin/export/history', methods=['GET'])
def admin_export_history():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403
    exports_dir = REVIEW_DIR / 'exports'
    batches = []
    if exports_dir.exists():
        for batch_dir in sorted(exports_dir.iterdir(), reverse=True):
            if batch_dir.is_dir():
                summary_file = batch_dir / 'summary.json'
                summary = {}
                if summary_file.exists():
                    try:
                        summary = json.loads(summary_file.read_text(encoding='utf-8'))
                    except Exception:
                        pass
                batches.append({
                    'batch_id': batch_dir.name,
                    'exported_at': summary.get('exported_at', ''),
                    'exported_task_count': summary.get('total_reviewed', 0),
                    'gt_error_count': summary.get('gt_error_count', 0),
                    'accepted_pseudo_label_count': summary.get('accepted_pseudo_label_count', 0),
                    'pseudo_label_error_count': summary.get('pseudo_label_error_count', 0),
                })
    return jsonify({'batches': batches})


@app.route('/admin/export', methods=['POST'])
def admin_export():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403
    if not export_lock.acquire(blocking=False):
        return jsonify({'error': 'export already running'}), 409
    try:
        batch_id, export_dir, summary, task_ids = build_incremental_export(RUN_ID)
        if not task_ids:
            return jsonify({'status': 'ok', 'exported_task_count': 0, 'message': 'no reviewed tasks to export'})
        if not db.mark_tasks_exported(RUN_ID, task_ids, batch_id):
            return jsonify({'error': 'failed to mark tasks exported'}), 500
        return jsonify({
            'status': 'ok',
            'export_batch_id': batch_id,
            'export_dir': str(export_dir),
            **summary,
        })
    finally:
        export_lock.release()


@app.route('/admin/assign', methods=['POST'])
def admin_assign():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403
    payload = request.get_json(force=True) or {}
    assignee = payload.get('assignee')
    task_ids = payload.get('task_ids') or []
    if not assignee or not task_ids:
        return jsonify({'error': 'missing fields'}), 400
    assignments = load_assignments()
    reviewed = get_reviewed_task_ids()
    exported = get_exported_task_ids()
    blocked = []
    for tid in task_ids:
        if tid in exported:
            blocked.append({'task_id': tid, 'reason': 'exported', 'assigned_to': assignments.get(tid)})
        elif tid in reviewed:
            blocked.append({'task_id': tid, 'reason': 'reviewed', 'assigned_to': assignments.get(tid)})
    if blocked:
        return jsonify({'error': '已审核或已导出的任务不能重新分配', 'blocked': blocked}), 409
    for tid in task_ids:
        assignments[tid] = assignee
    save_assignments(assignments)
    return jsonify({'status': 'ok'})


@app.route('/admin/generate', methods=['POST'])
def admin_generate():
    # generate tasks for a new run_id based on admin inputs
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403

    payload = request.get_json(force=True) or {}
    # Accept orgs (list) or organ (single string) for backward compatibility
    organ_list = payload.get('organs')
    if not organ_list:
        single_organ = payload.get('organ') or '肾脏'
        organ_list = [single_organ]
    if not isinstance(organ_list, list) or not organ_list:
        return jsonify({'error': 'organs must be a non-empty list'}), 400
    organs_str = ','.join(organ_list)

    model_version = payload.get('model_version') or ''
    data_version = payload.get('data_version') or ''
    data_root = payload.get('data_root') or CONFIG.get('data_root') or 'Test'
    provided_run_id = payload.get('run_id')

    # build run_id if not provided
    if provided_run_id:
        run_id = provided_run_id
    else:
        from datetime import datetime
        date = datetime.now().strftime('%Y%m%d')
        parts = [organ_list[0].replace(' ', '_')]
        if len(organ_list) > 1:
            parts.append(f'等多器官')
        if model_version:
            parts.append(str(model_version))
        parts.append(date)
        if data_version:
            parts.append(str(data_version))
        run_id = '_'.join([p for p in parts if p])

    # call task generator script with run_id and organs
    script = ROOT / 'scripts' / 'generate_tasks.py'
    if not script.exists():
        return jsonify({'error': 'generate script not found'}), 404

    # Prepare data for first organ (primary); the script will handle each organ independently
    try:
        generation_data_root = prepare_data_root_for_generation(data_root, organ_list[0])
    except Exception as exc:
        return jsonify({'error': f'failed to prepare data_root: {exc}'}), 500

    result = subprocess.run([
        sys.executable,
        str(script),
        '--run-id',
        run_id,
        '--organs',
        organs_str,
        '--data-root',
        str(generation_data_root),
        '--conf-threshold',
        str(CONFIG.get('conf_threshold', 0.85)),
        '--iou-threshold',
        str(CONFIG.get('iou_threshold', 0.5)),
        '--hard-sample-mode',
        str(CONFIG.get('hard_sample_mode', 'strict')),
    ], cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}), 500

    # initialize review outputs for this run and persist in config
    try:
        CONFIG['data_root'] = str(data_root)
        CONFIG['prepared_data_root'] = CONFIG.get('prepared_data_root') or 'test_data'
        initialize_review_outputs_for_run(run_id, persist_config=True)
    except Exception as e:
        return jsonify({'error': 'failed to initialize review outputs', 'exc': str(e)}), 500

    return jsonify({'status': 'ok', 'run_id': run_id, 'stdout': result.stdout, 'stderr': result.stderr})


@app.route('/admin/unassign', methods=['POST'])
def admin_unassign():
    user = session.get('user')
    role = session.get('role')
    if not user or role != 'admin':
        return jsonify({'error': 'forbidden'}), 403
    payload = request.get_json(force=True) or {}
    task_ids = payload.get('task_ids') or []
    if not task_ids:
        return jsonify({'error': 'missing fields'}), 400
    reviewed = get_reviewed_task_ids()
    exported = get_exported_task_ids()
    blocked = []
    for tid in task_ids:
        if tid in exported:
            blocked.append({'task_id': tid, 'reason': 'exported'})
        elif tid in reviewed:
            blocked.append({'task_id': tid, 'reason': 'reviewed'})
    if blocked:
        return jsonify({'error': '已审核或已导出的任务不能取消分配', 'blocked': blocked}), 409
    assignments = load_assignments()
    for tid in task_ids:
        if tid in assignments:
            assignments.pop(tid, None)
    save_assignments(assignments)
    return jsonify({'status': 'ok'})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template(
            'login.html',
            next_url=request.args.get('next', ''),
            error='请使用管理员账号登录' if request.args.get('msg') == 'admin_required' else None,
        )
    # POST
    payload = request.get_json(silent=True) or {}
    username = request.form.get('username') or payload.get('username')
    password = request.form.get('password') or payload.get('password')
    next_url = request.form.get('next') or payload.get('next') or ''
    if not username or not password:
        return render_template('login.html', error='missing credentials', next_url=next_url), 400
    role = authenticate(username, password)
    if role:
        session['user'] = username
        session['role'] = role
        if next_url == '/admin' and role == 'admin':
            return redirect(url_for('admin_page'))
        return redirect(url_for('doctor_page'))
    return render_template('login.html', error='invalid credentials', next_url=next_url), 401


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    import atexit
    import signal as _signal

    ECHOBOX = ROOT / "echobox"
    procs: list[subprocess.Popen] = []

    def _stop_echobox():
        for p in procs:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        capture_output=True,
                    )
                else:
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass

    def _on_sig(signum, frame):
        print("\nShutting down all services...")
        _stop_echobox()
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _on_sig)
    _signal.signal(_signal.SIGTERM, _on_sig)
    atexit.register(_stop_echobox)

    env = os.environ.copy()

    # 1. echobox FastAPI (port 8000)
    print("[echobox] Starting API (port 8000)...")
    uv_cmd = "uv.exe" if sys.platform == "win32" else "uv"
    api_proc = subprocess.Popen(
        [uv_cmd, "run", "--package", "echobox-app",
         "uvicorn", "echobox_app.main:create_app",
         "--factory", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ECHOBOX), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(api_proc)

    # 2. echobox Vite dev server (port 5173)
    print("[echobox] Starting frontend (port 5173)...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    web_proc = subprocess.Popen(
        [npm, "--prefix", str(ECHOBOX / "frontend"), "run", "dev"],
        cwd=str(ECHOBOX), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(web_proc)

    import time
    time.sleep(2)

    print("\nServices:")
    print("  MedRev:         http://127.0.0.1:8080")
    print("  echobox API:    http://127.0.0.1:8000")
    print("  echobox UI:     http://127.0.0.1:5173")
    print()

    try:
        app.run(port=8080)
    finally:
        _stop_echobox()



