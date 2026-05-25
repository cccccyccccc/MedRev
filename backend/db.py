"""
SQLite database module for MedRev review system.
Handles persistent storage of assignments and reviews with proper concurrency control.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import threading

DB_PATH = Path(__file__).parent.parent / 'medrev.db'
_DB_LOCK = threading.Lock()

def get_connection():
    """Get a connection with proper timeout and isolation level."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0, check_same_thread=False)
    conn.isolation_level = 'DEFERRED'
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database schema if not exists."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Assignments table: which doctor is assigned which task in which run
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                assigned_doctor TEXT,
                assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, task_id)
            )
        ''')
        
        # Raw reviews table: streaming records of each doctor's submission
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_type TEXT,
                organ TEXT,
                image_name TEXT,
                image_path TEXT,
                review_result TEXT,
                error_type TEXT,
                reviewer_id TEXT,
                submitted_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                annotation_json TEXT
            )
        ''')

        # migration: add annotation_json if missing from existing table
        try:
            cursor.execute('ALTER TABLE raw_reviews ADD COLUMN annotation_json TEXT')
        except Exception:
            pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exported_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                export_batch_id TEXT NOT NULL,
                exported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, task_id)
            )
        ''')
        
        # Indexes for query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_run_task ON assignments(run_id, task_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_doctor ON assignments(assigned_doctor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_reviews_run_id ON raw_reviews(run_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_reviews_task_id ON raw_reviews(task_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exported_tasks_run_task ON exported_tasks(run_id, task_id)')
        
        conn.commit()
    finally:
        conn.close()

# ============ Assignments API ============

def assign_task(run_id: str, task_id: str, doctor: str) -> bool:
    """Assign a task to a doctor. Returns True if successful."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO assignments (run_id, task_id, assigned_doctor) VALUES (?, ?, ?)',
            (run_id, task_id, doctor)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error assigning task: {e}")
        return False
    finally:
        conn.close()

def unassign_task(run_id: str, task_id: str) -> bool:
    """Remove assignment for a task."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM assignments WHERE run_id = ? AND task_id = ?', (run_id, task_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error unassigning task: {e}")
        return False
    finally:
        conn.close()

def get_assignment(run_id: str, task_id: str) -> Optional[str]:
    """Get the doctor assigned to a task, or None."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT assigned_doctor FROM assignments WHERE run_id = ? AND task_id = ?', 
                      (run_id, task_id))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()

def get_assignments_for_run(run_id: str) -> Dict[str, str]:
    """Get all task->doctor assignments for a run."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT task_id, assigned_doctor FROM assignments WHERE run_id = ?', (run_id,))
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()

def get_tasks_for_doctor(run_id: str, doctor: str) -> List[str]:
    """Get all task_ids assigned to a doctor in a run."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT task_id FROM assignments WHERE run_id = ? AND assigned_doctor = ?',
                      (run_id, doctor))
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

def load_assignments_dict(run_id: str) -> Dict[str, str]:
    """Load all assignments for a run as dict (for compatibility)."""
    return get_assignments_for_run(run_id)

def save_assignments_dict(run_id: str, assignments: Dict[str, str]):
    """Save a full assignments dict for a run (overwrites existing)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM assignments WHERE run_id = ?', (run_id,))
        for task_id, doctor in assignments.items():
            cursor.execute('INSERT INTO assignments (run_id, task_id, assigned_doctor) VALUES (?, ?, ?)',
                          (run_id, task_id, doctor))
        conn.commit()
    finally:
        conn.close()

# ============ Raw Reviews API ============

def append_review(review_record: Dict[str, Any]) -> bool:
    """Append a review record. Returns True if successful."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO raw_reviews
            (run_id, task_id, task_type, organ, image_name, image_path,
             review_result, error_type, reviewer_id, submitted_at, annotation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            review_record.get('run_id'),
            review_record.get('task_id'),
            review_record.get('task_type'),
            review_record.get('organ'),
            review_record.get('image_name'),
            review_record.get('image_path'),
            review_record.get('review_result'),
            review_record.get('error_type'),
            review_record.get('reviewer_id'),
            review_record.get('submitted_at'),
            review_record.get('annotation_json'),
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error appending review: {e}")
        return False
    finally:
        conn.close()


def delete_review(run_id: str, task_id: str, reviewer_id: str, submitted_at: str) -> bool:
    """Delete a review record by its identifying fields."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM raw_reviews WHERE run_id = ? AND task_id = ? AND reviewer_id = ? AND submitted_at = ?',
            (run_id, task_id, reviewer_id, submitted_at),
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting review: {e}")
        return False
    finally:
        conn.close()

def get_reviews_for_run(run_id: str) -> List[Dict[str, Any]]:
    """Get all reviews for a run."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM raw_reviews WHERE run_id = ? ORDER BY created_at', (run_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_reviews_for_run(run_id: str) -> Dict[str, Dict[str, Any]]:
    """Get the latest review per task for a run."""
    latest = {}
    for row in get_reviews_for_run(run_id):
        record = dict(row) if hasattr(row, 'keys') else row
        task_id = record.get('task_id')
        if task_id:
            latest[task_id] = record
    return latest


def get_exported_tasks_for_run(run_id: str) -> Dict[str, Dict[str, Any]]:
    """Get exported task metadata keyed by task_id."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM exported_tasks WHERE run_id = ?', (run_id,))
        return {row['task_id']: dict(row) for row in cursor.fetchall()}
    finally:
        conn.close()


def mark_tasks_exported(run_id: str, task_ids: List[str], export_batch_id: str) -> bool:
    """Mark tasks as exported in a batch. Existing exported marks are kept."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        for task_id in task_ids:
            cursor.execute(
                'INSERT OR IGNORE INTO exported_tasks (run_id, task_id, export_batch_id) VALUES (?, ?, ?)',
                (run_id, task_id, export_batch_id),
            )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error marking tasks exported: {e}")
        return False
    finally:
        conn.close()

def get_reviews_summary(run_id: str) -> Dict[str, Any]:
    """Get summary statistics for reviews in a run."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_reviewed,
                SUM(CASE WHEN review_result = 'gt_error' THEN 1 ELSE 0 END) as gt_error_count,
                SUM(CASE WHEN review_result = 'pseudo_label_correct' THEN 1 ELSE 0 END) as accepted_pseudo_count,
                SUM(CASE WHEN review_result = 'pseudo_label_error' THEN 1 ELSE 0 END) as pseudo_label_error_count
            FROM raw_reviews
            WHERE run_id = ?
        ''', (run_id,))
        row = cursor.fetchone()
        if row:
            return {
                'total_reviewed': row['total_reviewed'] or 0,
                'gt_error_count': row['gt_error_count'] or 0,
                'accepted_pseudo_count': row['accepted_pseudo_count'] or 0,
                'pseudo_label_error_count': row['pseudo_label_error_count'] or 0,
            }
        return {'total_reviewed': 0, 'gt_error_count': 0, 'accepted_pseudo_count': 0, 'pseudo_label_error_count': 0}
    finally:
        conn.close()

def export_reviews_as_jsonl(run_id: str, output_file: Path) -> bool:
    """Export all reviews for a run as JSONL."""
    try:
        reviews = get_reviews_for_run(run_id)
        with output_file.open('w', encoding='utf-8') as f:
            for review in reviews:
                # Convert datetime objects to ISO strings for JSON serialization
                for key in ['submitted_at', 'created_at']:
                    if key in review and isinstance(review[key], str):
                        # Already a string
                        pass
                import json
                f.write(json.dumps(review, ensure_ascii=False, default=str) + '\n')
        return True
    except Exception as e:
        print(f"Error exporting reviews: {e}")
        return False

# ============ Migration Helpers ============

def import_assignments_from_json(json_file: Path, run_id: str):
    """Import assignments from a JSON file into database."""
    try:
        import json
        if json_file.exists():
            with json_file.open('r', encoding='utf-8') as f:
                assignments = json.load(f)
            if isinstance(assignments, dict):
                save_assignments_dict(run_id, assignments)
                print(f"Imported {len(assignments)} assignments from {json_file}")
    except Exception as e:
        print(f"Error importing assignments: {e}")

def import_reviews_from_jsonl(jsonl_file: Path, run_id: str):
    """Import reviews from JSONL file into database."""
    try:
        import json
        if jsonl_file.exists():
            count = 0
            with jsonl_file.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # Ensure run_id matches
                        record['run_id'] = run_id
                        append_review(record)
                        count += 1
            print(f"Imported {count} reviews from {jsonl_file}")
    except Exception as e:
        print(f"Error importing reviews: {e}")
