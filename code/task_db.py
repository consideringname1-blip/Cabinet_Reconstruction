import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ARUCO_ANCHOR_MARKER_ID,
    ARUCO_REFERENCE_ROOT,
    ARUCO_SYNC_MARKER_REGISTRY_ON_START,
    ARUCO_TEMPLATE_PATH,
    DATABASE_PATH,
    UPLOAD_FOLDER,
)
from task_json import normalize_path_for_storage


TABLE_NAME = "tasks"
STAGE_RUN_TABLE = "task_stage_runs"
ARUCO_REFERENCE_TABLE = "aruco_references"
ARUCO_MARKER_TABLE = "aruco_markers"
ARUCO_MARKER_RELATION_TABLE = "aruco_marker_relations"
MODEL_BOUNDS_TABLE = "model_bounds"
MODEL_BOUNDS_STATUSES = (
    "pending",
    "ready",
    "failed",
    "pending_reference",
)
ALLOWED_STATUSES = (
    "pending",
    "hololens2depth",
    "aruco_detect",
    "sam3mask",
    "instantmesh",
    "depthpointcloud",
    "modelscale",
    "icpalignment",
    "pose",
    "aruco_sync",
    "runtime_mesh",
    "blender",
    "model_bounds",
    "completed",
    "aruco_completed",
    "failed",
)
TERMINAL_STATUSES = ("completed", "aruco_completed", "failed")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


def _status_list_sql() -> str:
    return ", ".join(f"'{status}'" for status in ALLOWED_STATUSES)


def _create_task_table_sql() -> str:
    return f"""
        CREATE TABLE {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ({_status_list_sql()})),
            json_path TEXT NOT NULL,
            startup_session_id TEXT,
            aruco_coordinate_synced INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        )
    """


def _create_aruco_reference_table_sql() -> str:
    return f"""
        CREATE TABLE {ARUCO_REFERENCE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            startup_session_id TEXT NOT NULL,
            task_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            marker_pose_json TEXT NOT NULL,
            raw_record_path TEXT NOT NULL,
            config_snapshot_json TEXT NOT NULL
        )
    """


def _create_stage_run_table_sql() -> str:
    return f"""
        CREATE TABLE {STAGE_RUN_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'completed', 'failed')),
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            duration_ms INTEGER,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_id, stage_name)
        )
    """


def _create_aruco_marker_table_sql() -> str:
    return f"""
        CREATE TABLE {ARUCO_MARKER_TABLE} (
            marker_id INTEGER PRIMARY KEY,
            dictionary TEXT NOT NULL,
            marker_size_mm REAL NOT NULL,
            reference_image_name TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            source_path TEXT,
            config_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_aruco_marker_relation_table_sql() -> str:
    return f"""
        CREATE TABLE {ARUCO_MARKER_RELATION_TABLE} (
            anchor_marker_id INTEGER NOT NULL,
            marker_id INTEGER NOT NULL,
            relation_pose_json TEXT NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 0,
            mean_error REAL,
            last_observed_task_id TEXT,
            raw_record_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (anchor_marker_id, marker_id)
        )
    """


def _model_bounds_status_list_sql() -> str:
    return ", ".join(f"'{status}'" for status in MODEL_BOUNDS_STATUSES)


def _create_model_bounds_table_sql() -> str:
    return f"""
        CREATE TABLE {MODEL_BOUNDS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ({_model_bounds_status_list_sql()})),
            uploaded_at TEXT,
            model_name TEXT,
            fbx_name TEXT,
            coordinate_space TEXT NOT NULL DEFAULT 'aruco',
            aruco_reference_task_id TEXT,
            object_aruco_json TEXT,
            aabb_min_aruco_json TEXT,
            aabb_max_aruco_json TEXT,
            corners_aruco_json TEXT,
            source_model_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _table_sql(conn: sqlite3.Connection, table_name: str) -> Optional[str]:
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row["sql"] if row else None


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _task_table_needs_migration(conn: sqlite3.Connection) -> bool:
    sql = _table_sql(conn, TABLE_NAME)
    if not sql:
        return False

    required_columns = {"startup_session_id", "aruco_coordinate_synced"}
    existing_columns = _get_table_columns(conn, TABLE_NAME)
    return (
        any(f"'{status}'" not in sql for status in ALLOWED_STATUSES)
        or not required_columns.issubset(existing_columns)
    )


def _migrate_task_table(conn: sqlite3.Connection) -> None:
    legacy_table = f"{TABLE_NAME}_legacy"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE {TABLE_NAME} RENAME TO {legacy_table}")
    conn.execute(_create_task_table_sql())

    legacy_columns = _get_table_columns(conn, legacy_table)
    has_startup_session_id = "startup_session_id" in legacy_columns
    has_aruco_coordinate_synced = "aruco_coordinate_synced" in legacy_columns

    startup_select = "startup_session_id" if has_startup_session_id else "NULL"
    synced_select = "aruco_coordinate_synced" if has_aruco_coordinate_synced else "0"

    conn.execute(
        f"""
        INSERT INTO {TABLE_NAME} (
            id,
            task_id,
            status,
            json_path,
            startup_session_id,
            aruco_coordinate_synced,
            created_at,
            started_at,
            completed_at,
            updated_at,
            error_message
        )
        SELECT
            id,
            task_id,
            status,
            json_path,
            {startup_select},
            {synced_select},
            created_at,
            started_at,
            completed_at,
            updated_at,
            error_message
        FROM {legacy_table}
        """
    )
    conn.execute(f"DROP TABLE {legacy_table}")


def _normalize_stored_paths(conn: sqlite3.Connection) -> None:
    task_rows = conn.execute(f"SELECT id, json_path FROM {TABLE_NAME}").fetchall()
    for row in task_rows:
        current = str(row["json_path"])
        normalized = normalize_path_for_storage(current)
        if normalized == current:
            continue
        conn.execute(
            f"UPDATE {TABLE_NAME} SET json_path = ? WHERE id = ?",
            (normalized, int(row["id"])),
        )

    if _table_sql(conn, ARUCO_REFERENCE_TABLE) is None:
        return

    ref_rows = conn.execute(
        f"SELECT id, raw_record_path FROM {ARUCO_REFERENCE_TABLE}"
    ).fetchall()
    for row in ref_rows:
        current = str(row["raw_record_path"])
        normalized = normalize_path_for_storage(current)
        if normalized == current:
            continue
        conn.execute(
            f"UPDATE {ARUCO_REFERENCE_TABLE} SET raw_record_path = ? WHERE id = ?",
            (normalized, int(row["id"])),
        )

    if _table_sql(conn, ARUCO_MARKER_TABLE) is not None:
        marker_rows = conn.execute(
            f"SELECT marker_id, source_path FROM {ARUCO_MARKER_TABLE} WHERE source_path IS NOT NULL"
        ).fetchall()
        for row in marker_rows:
            current = str(row["source_path"] or "")
            if not current:
                continue
            normalized = normalize_path_for_storage(current)
            if normalized == current:
                continue
            conn.execute(
                f"UPDATE {ARUCO_MARKER_TABLE} SET source_path = ? WHERE marker_id = ?",
                (normalized, int(row["marker_id"])),
            )

    if _table_sql(conn, ARUCO_MARKER_RELATION_TABLE) is not None:
        relation_rows = conn.execute(
            f"SELECT anchor_marker_id, marker_id, raw_record_path FROM {ARUCO_MARKER_RELATION_TABLE} WHERE raw_record_path IS NOT NULL"
        ).fetchall()
        for row in relation_rows:
            current = str(row["raw_record_path"] or "")
            if not current:
                continue
            normalized = normalize_path_for_storage(current)
            if normalized == current:
                continue
            conn.execute(
                f"""
                UPDATE {ARUCO_MARKER_RELATION_TABLE}
                SET raw_record_path = ?
                WHERE anchor_marker_id = ? AND marker_id = ?
                """,
                (normalized, int(row["anchor_marker_id"]), int(row["marker_id"])),
            )


def _load_aruco_template_config() -> Dict[str, Any]:
    if not ARUCO_TEMPLATE_PATH.is_file():
        return {}
    try:
        with ARUCO_TEMPLATE_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _infer_marker_id_from_path(path: Path) -> Optional[int]:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _marker_overrides_by_id(template: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    overrides: Dict[int, Dict[str, Any]] = {}
    markers = template.get("markers")
    if isinstance(markers, list):
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            marker_id = marker.get("marker_id", marker.get("id"))
            try:
                marker_id_int = int(marker_id)
            except Exception:
                continue
            overrides[marker_id_int] = marker
    return overrides


def _sync_marker_registry_from_reference_folder(conn: sqlite3.Connection) -> int:
    template = _load_aruco_template_config()
    overrides = _marker_overrides_by_id(template)
    default_dictionary = str(template.get("dictionary") or "DICT_7X7_1000")
    default_marker_size_mm = float(template.get("marker_size_mm") or 200.0)

    seen_marker_ids: set[int] = set()
    synced_count = 0
    for image_path in sorted(ARUCO_REFERENCE_ROOT.glob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        marker_id = _infer_marker_id_from_path(image_path)
        if marker_id is None:
            continue

        marker_config = dict(overrides.get(marker_id) or {})
        dictionary = str(marker_config.get("dictionary") or default_dictionary)
        marker_size_mm = float(marker_config.get("marker_size_mm") or default_marker_size_mm)
        enabled = bool(marker_config.get("enabled", True))
        reference_image_name = str(marker_config.get("reference_image_name") or image_path.name)
        source_path = normalize_path_for_storage(image_path)

        conn.execute(
            f"""
            INSERT INTO {ARUCO_MARKER_TABLE} (
                marker_id,
                dictionary,
                marker_size_mm,
                reference_image_name,
                enabled,
                source_path,
                config_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(marker_id) DO UPDATE SET
                dictionary = excluded.dictionary,
                marker_size_mm = excluded.marker_size_mm,
                reference_image_name = excluded.reference_image_name,
                enabled = excluded.enabled,
                source_path = excluded.source_path,
                config_json = excluded.config_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                marker_id,
                dictionary,
                marker_size_mm,
                reference_image_name,
                1 if enabled else 0,
                source_path,
                json.dumps(marker_config, ensure_ascii=False),
            ),
        )
        seen_marker_ids.add(marker_id)
        synced_count += 1

    if seen_marker_ids:
        placeholders = ", ".join("?" for _ in seen_marker_ids)
        conn.execute(
            f"""
            UPDATE {ARUCO_MARKER_TABLE}
            SET enabled = 0, updated_at = CURRENT_TIMESTAMP
            WHERE marker_id NOT IN ({placeholders})
            """,
            tuple(sorted(seen_marker_ids)),
        )
    else:
        conn.execute(
            f"UPDATE {ARUCO_MARKER_TABLE} SET enabled = 0, updated_at = CURRENT_TIMESTAMP"
        )

    return synced_count


def initialize_task_table() -> None:
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")

        if _table_sql(conn, TABLE_NAME) is None:
            conn.execute(_create_task_table_sql())
        elif _task_table_needs_migration(conn):
            _migrate_task_table(conn)

        if _table_sql(conn, ARUCO_REFERENCE_TABLE) is None:
            conn.execute(_create_aruco_reference_table_sql())
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{ARUCO_REFERENCE_TABLE}_startup_session
                ON {ARUCO_REFERENCE_TABLE} (startup_session_id)
                """
            )

        if _table_sql(conn, STAGE_RUN_TABLE) is None:
            conn.execute(_create_stage_run_table_sql())
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{STAGE_RUN_TABLE}_task
                ON {STAGE_RUN_TABLE} (task_id)
                """
            )

        if _table_sql(conn, ARUCO_MARKER_TABLE) is None:
            conn.execute(_create_aruco_marker_table_sql())
        if _table_sql(conn, ARUCO_MARKER_RELATION_TABLE) is None:
            conn.execute(_create_aruco_marker_relation_table_sql())
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{ARUCO_MARKER_RELATION_TABLE}_marker
                ON {ARUCO_MARKER_RELATION_TABLE} (marker_id)
                """
            )

        if _table_sql(conn, MODEL_BOUNDS_TABLE) is None:
            conn.execute(_create_model_bounds_table_sql())
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{MODEL_BOUNDS_TABLE}_status_uploaded
                ON {MODEL_BOUNDS_TABLE} (status, uploaded_at)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{MODEL_BOUNDS_TABLE}_task
                ON {MODEL_BOUNDS_TABLE} (task_id)
                """
            )

        if ARUCO_SYNC_MARKER_REGISTRY_ON_START:
            _sync_marker_registry_from_reference_folder(conn)

        _normalize_stored_paths(conn)
        conn.commit()


def get_latest_10_records() -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT 10").fetchall()
    return [dict(row) for row in rows]


def sync_marker_registry_from_reference_folder() -> int:
    initialize_task_table()
    with _get_connection() as conn:
        synced_count = _sync_marker_registry_from_reference_folder(conn)
        conn.commit()
    return synced_count


def get_enabled_aruco_markers() -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {ARUCO_MARKER_TABLE}
            WHERE enabled = 1
            ORDER BY marker_id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_aruco_marker_relation(
    marker_id: int,
    *,
    anchor_marker_id: int = ARUCO_ANCHOR_MARKER_ID,
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {ARUCO_MARKER_RELATION_TABLE}
            WHERE anchor_marker_id = ? AND marker_id = ?
            """,
            (int(anchor_marker_id), int(marker_id)),
        ).fetchone()
    return _row_to_dict(row)


def upsert_aruco_marker_relation(
    *,
    marker_id: int,
    relation_pose_json: Any,
    sample_error: float | None = None,
    task_id: str | None = None,
    raw_record_path: str | None = None,
    anchor_marker_id: int = ARUCO_ANCHOR_MARKER_ID,
) -> Dict[str, Any]:
    initialize_task_table()
    marker_id = int(marker_id)
    anchor_marker_id = int(anchor_marker_id)
    if marker_id == anchor_marker_id:
        sample_count = 1
        mean_error = float(sample_error) if sample_error is not None else 0.0
    else:
        existing = get_aruco_marker_relation(marker_id, anchor_marker_id=anchor_marker_id)
        if existing:
            old_count = int(existing.get("sample_count") or 0)
            old_mean = existing.get("mean_error")
            sample_count = old_count + 1
            if sample_error is None:
                mean_error = float(old_mean) if old_mean is not None else None
            elif old_mean is None:
                mean_error = float(sample_error)
            else:
                mean_error = ((float(old_mean) * old_count) + float(sample_error)) / max(sample_count, 1)
        else:
            sample_count = 1
            mean_error = float(sample_error) if sample_error is not None else None

    raw_path = normalize_path_for_storage(raw_record_path) if raw_record_path else None
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {ARUCO_MARKER_RELATION_TABLE} (
                anchor_marker_id,
                marker_id,
                relation_pose_json,
                sample_count,
                mean_error,
                last_observed_task_id,
                raw_record_path,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(anchor_marker_id, marker_id) DO UPDATE SET
                relation_pose_json = excluded.relation_pose_json,
                sample_count = excluded.sample_count,
                mean_error = excluded.mean_error,
                last_observed_task_id = excluded.last_observed_task_id,
                raw_record_path = excluded.raw_record_path,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                anchor_marker_id,
                marker_id,
                json.dumps(relation_pose_json, ensure_ascii=False),
                sample_count,
                mean_error,
                task_id,
                raw_path,
            ),
        )
        conn.commit()
        row = conn.execute(
            f"""
            SELECT *
            FROM {ARUCO_MARKER_RELATION_TABLE}
            WHERE anchor_marker_id = ? AND marker_id = ?
            """,
            (anchor_marker_id, marker_id),
        ).fetchone()
    return dict(row)


def mark_task_stage_started(task_id: str, stage_name: str) -> None:
    initialize_task_table()
    task_id = str(task_id)
    stage_name = str(stage_name)
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {STAGE_RUN_TABLE} (
                task_id,
                stage_name,
                status,
                started_at,
                completed_at,
                duration_ms,
                error_message,
                updated_at
            )
            VALUES (?, ?, 'running', ?, NULL, NULL, NULL, ?)
            ON CONFLICT(task_id, stage_name) DO UPDATE SET
                status = 'running',
                started_at = excluded.started_at,
                completed_at = NULL,
                duration_ms = NULL,
                error_message = NULL,
                updated_at = excluded.updated_at
            """,
            (task_id, stage_name, now, now),
        )
        conn.commit()


def mark_task_stage_completed(task_id: str, stage_name: str) -> None:
    initialize_task_table()
    task_id = str(task_id)
    stage_name = str(stage_name)
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute(
            f"""
            UPDATE {STAGE_RUN_TABLE}
            SET
                status = 'completed',
                completed_at = ?,
                duration_ms = CAST((julianday(?) - julianday(started_at)) * 86400000 AS INTEGER),
                error_message = NULL,
                updated_at = ?
            WHERE task_id = ? AND stage_name = ?
            """,
            (now, now, now, task_id, stage_name),
        )
        conn.commit()


def mark_task_stage_failed(task_id: str, stage_name: str, error_message: str | None = None) -> None:
    initialize_task_table()
    task_id = str(task_id)
    stage_name = str(stage_name)
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute(
            f"""
            UPDATE {STAGE_RUN_TABLE}
            SET
                status = 'failed',
                completed_at = ?,
                duration_ms = CAST((julianday(?) - julianday(started_at)) * 86400000 AS INTEGER),
                error_message = ?,
                updated_at = ?
            WHERE task_id = ? AND stage_name = ?
            """,
            (now, now, error_message, now, task_id, stage_name),
        )
        conn.commit()


def get_task_stage_runs(task_id: str) -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {STAGE_RUN_TABLE}
            WHERE task_id = ?
            ORDER BY started_at ASC, id ASC
            """,
            (str(task_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_status_by_task_id(task_id: str) -> Optional[str]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT status FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return row["status"] if row else None


def get_json_path_by_task_id(task_id: str) -> Optional[str]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT json_path FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return row["json_path"] if row else None


def create_task(
    task_id: str,
    json_path: Path | str,
    *,
    startup_session_id: str | None = None,
) -> Dict[str, Any]:
    initialize_task_table()
    json_path_str = normalize_path_for_storage(json_path, default_base=UPLOAD_FOLDER)
    startup_session_id = str(startup_session_id or "").strip() or None
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                task_id,
                status,
                json_path,
                startup_session_id,
                aruco_coordinate_synced
            )
            VALUES (?, 'pending', ?, ?, 0)
            """,
            (task_id, json_path_str, startup_session_id),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row)


def get_latest_unfinished_task() -> Optional[Dict[str, Any]]:
    initialize_task_table()
    terminal_list = ", ".join(f"'{status}'" for status in TERMINAL_STATUSES)
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT task_id, json_path, status
            FROM {TABLE_NAME}
            WHERE status NOT IN ({terminal_list})
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    return _row_to_dict(row)


def get_latest_completed_task(
    startup_session_id: str | None = None,
    require_aruco_coordinate_synced: bool = False,
    history_offset: int = 0,
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    history_offset = max(0, int(history_offset or 0))
    where_clauses = ["status = 'completed'"]
    params: List[Any] = []
    if startup_session_id:
        where_clauses.append("startup_session_id = ?")
        params.append(startup_session_id)
    if require_aruco_coordinate_synced:
        where_clauses.append("aruco_coordinate_synced = 1")

    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY id DESC
            LIMIT 1 OFFSET ?
            """,
            tuple(params + [history_offset]),
        ).fetchone()
    return _row_to_dict(row)


def get_latest_completed_tasks(
    startup_session_id: str | None = None,
    require_aruco_coordinate_synced: bool = False,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    limit = max(1, min(int(limit or 5), 50))
    where_clauses = ["status = 'completed'"]
    params: List[Any] = []
    if startup_session_id:
        where_clauses.append("startup_session_id = ?")
        params.append(startup_session_id)
    if require_aruco_coordinate_synced:
        where_clauses.append("aruco_coordinate_synced = 1")

    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params + [limit]),
        ).fetchall()
    return [dict(row) for row in rows]


def get_completed_tasks_for_startup(
    startup_session_id: str,
    *,
    require_unsynced: bool = False,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    if not startup_session_id:
        return []

    where_clauses = ["status = 'completed'", "startup_session_id = ?"]
    params: List[Any] = [startup_session_id]
    if require_unsynced:
        where_clauses.append("aruco_coordinate_synced = 0")

    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY id DESC
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def get_unsynced_completed_tasks() -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE status = 'completed' AND aruco_coordinate_synced = 0
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_unfinished_tasks() -> List[Dict[str, Any]]:
    initialize_task_table()
    terminal_list = ", ".join(f"'{status}'" for status in TERMINAL_STATUSES)
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE status NOT IN ({terminal_list})
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def update_task_status(
    task_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    initialize_task_table()

    set_parts = [
        "status = ?",
        "updated_at = CURRENT_TIMESTAMP",
        "error_message = ?",
    ]
    params: List[Any] = [status, error_message]

    if status != "pending":
        set_parts.append(
            "started_at = CASE WHEN started_at IS NULL THEN CURRENT_TIMESTAMP ELSE started_at END"
        )

    if status in {"completed", "aruco_completed"}:
        set_parts.append("completed_at = CURRENT_TIMESTAMP")
    else:
        set_parts.append("completed_at = NULL")

    params.append(task_id)

    with _get_connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET {", ".join(set_parts)}
            WHERE task_id = ?
            """,
            tuple(params),
        )
        conn.commit()
    return cursor.rowcount > 0


def update_task_aruco_coordinate_synced(task_id: str, synced: bool) -> bool:
    initialize_task_table()
    with _get_connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET aruco_coordinate_synced = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """,
            (1 if synced else 0, task_id),
        )
        conn.commit()
    return cursor.rowcount > 0


def get_task_by_task_id(task_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return _row_to_dict(row)


def create_aruco_reference(
    *,
    startup_session_id: str,
    task_id: str | None,
    marker_pose_json: Any,
    raw_record_path: str,
    config_snapshot_json: Any,
) -> Dict[str, Any]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    if not startup_session_id:
        raise ValueError("startup_session_id is required to store an ArUco reference")

    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {ARUCO_REFERENCE_TABLE} (
                startup_session_id,
                task_id,
                marker_pose_json,
                raw_record_path,
                config_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                startup_session_id,
                task_id,
                json.dumps(marker_pose_json, ensure_ascii=False),
                normalize_path_for_storage(raw_record_path),
                json.dumps(config_snapshot_json, ensure_ascii=False),
            ),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {ARUCO_REFERENCE_TABLE} WHERE id = last_insert_rowid()"
        ).fetchone()
    return dict(row)


def get_latest_aruco_reference(startup_session_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    if not startup_session_id:
        return None
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {ARUCO_REFERENCE_TABLE}
            WHERE startup_session_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (startup_session_id,),
        ).fetchone()
    return _row_to_dict(row)


def _dump_optional_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def upsert_model_bounds(
    *,
    task_id: str,
    status: str,
    uploaded_at: str | None = None,
    model_name: str | None = None,
    fbx_name: str | None = None,
    coordinate_space: str = "aruco",
    aruco_reference_task_id: str | None = None,
    object_aruco_json: Any = None,
    aabb_min_aruco_json: Any = None,
    aabb_max_aruco_json: Any = None,
    corners_aruco_json: Any = None,
    source_model_path: str | None = None,
    error_message: str | None = None,
) -> Dict[str, Any]:
    if status not in MODEL_BOUNDS_STATUSES:
        raise ValueError(f"Invalid model bounds status: {status}")

    initialize_task_table()
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {MODEL_BOUNDS_TABLE} (
                task_id,
                status,
                uploaded_at,
                model_name,
                fbx_name,
                coordinate_space,
                aruco_reference_task_id,
                object_aruco_json,
                aabb_min_aruco_json,
                aabb_max_aruco_json,
                corners_aruco_json,
                source_model_path,
                error_message,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status = excluded.status,
                uploaded_at = excluded.uploaded_at,
                model_name = excluded.model_name,
                fbx_name = excluded.fbx_name,
                coordinate_space = excluded.coordinate_space,
                aruco_reference_task_id = excluded.aruco_reference_task_id,
                object_aruco_json = excluded.object_aruco_json,
                aabb_min_aruco_json = excluded.aabb_min_aruco_json,
                aabb_max_aruco_json = excluded.aabb_max_aruco_json,
                corners_aruco_json = excluded.corners_aruco_json,
                source_model_path = excluded.source_model_path,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                str(task_id),
                status,
                uploaded_at,
                model_name,
                fbx_name,
                coordinate_space,
                aruco_reference_task_id,
                _dump_optional_json(object_aruco_json),
                _dump_optional_json(aabb_min_aruco_json),
                _dump_optional_json(aabb_max_aruco_json),
                _dump_optional_json(corners_aruco_json),
                source_model_path,
                error_message,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {MODEL_BOUNDS_TABLE} WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
    return dict(row)


def get_model_bounds_by_task_id(task_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {MODEL_BOUNDS_TABLE} WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
    return _row_to_dict(row)


def get_latest_ready_model_bounds(limit: int = 5) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit or 5), 50))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {MODEL_BOUNDS_TABLE}
            WHERE status = 'ready'
            ORDER BY uploaded_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_ready_model_bounds_in_range(
    start: str,
    end: str,
    *,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    start = str(start or "").strip()
    end = str(end or "").strip()
    if not start or not end:
        raise ValueError("start and end are required")
    limit = max(1, min(int(limit or 50), 200))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {MODEL_BOUNDS_TABLE}
            WHERE status = 'ready'
                AND uploaded_at >= ?
                AND uploaded_at <= ?
            ORDER BY uploaded_at DESC, id DESC
            LIMIT ?
            """,
            (start, end, limit),
        ).fetchall()
    return [dict(row) for row in rows]
