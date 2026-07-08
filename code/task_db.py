import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import (
    ARUCO_ANCHOR_MARKER_ID,
    ARUCO_SYNC_MARKER_REGISTRY_ON_START,
)
from artifact_layout import (
    ARUCO_REFERENCE_ROOT,
    ARUCO_TEMPLATE_PATH,
    DATABASE_PATH,
    ensure_database_root,
)
from task_json import normalize_path_for_storage


TABLE_NAME = "tasks"
STAGE_RUN_TABLE = "task_stage_runs"
ARUCO_REFERENCE_TABLE = "aruco_references"
ARUCO_MARKER_TABLE = "aruco_markers"
ARUCO_MARKER_RELATION_TABLE = "aruco_marker_relations"
MODEL_BOUNDS_TABLE = "model_bounds"
TASK_TIMING_EVENT_TABLE = "task_timing_events"
AI_MODEL_TIMING_TABLE = "ai_model_timings"
DISPLAY_OBJECT_TABLE = "display_objects"
CAPTURE_INSTANCE_TABLE = "capture_instances"
CAPTURE_BINDING_LOG_TABLE = "capture_binding_logs"
HISTORY_PLACEMENT_REQUEST_TABLE = "history_placement_requests"
MODEL_BOUNDS_STATUSES = (
    "pending",
    "ready",
    "failed",
    "pending_reference",
)
CAPTURE_BINDING_STATUSES = (
    "bound",
    "unbound",
    "rejected",
)
ALLOWED_STATUSES = (
    "uploading",
    "upload_failed",
    "pending",
    "hololens2depth",
    "aruco_detect",
    "sam3mask",
    "historical_model_match",
    "instantmesh",
    "depthpointcloud",
    "modelscale",
    "object_alignment",
    "pose",
    "aruco_sync",
    "runtime_mesh",
    "model_bounds",
    "display_identity",
    "history_placement_restoration",
    "taken_object_detection",
    "sam3d_body_mesh",
    "completed",
    "aruco_completed",
    "failed",
)
TERMINAL_STATUSES = ("completed", "aruco_completed", "failed", "upload_failed")
_SCHEMA_INITIALIZED = False


def _get_connection() -> sqlite3.Connection:
    ensure_database_root()
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


def _utc_text_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


def _ensure_schema_initialized() -> None:
    if not _SCHEMA_INITIALIZED:
        initialize_task_table()


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
            task_timestamp TEXT,
            startup_session_id TEXT,
            aruco_coordinate_synced INTEGER NOT NULL DEFAULT 0,
            artifact_schema_version INTEGER NOT NULL DEFAULT 1,
            debug_enabled INTEGER NOT NULL DEFAULT 1,
            logs_enabled INTEGER NOT NULL DEFAULT 1,
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


def _create_task_timing_event_table_sql() -> str:
    return f"""
        CREATE TABLE {TASK_TIMING_EVENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            event_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed'
                CHECK (status IN ('completed', 'failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_ai_model_timing_table_sql() -> str:
    return f"""
        CREATE TABLE {AI_MODEL_TIMING_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT NOT NULL,
            timing_kind TEXT NOT NULL
                CHECK (timing_kind IN ('initialization', 'task')),
            stage_name TEXT,
            task_id TEXT,
            status TEXT NOT NULL DEFAULT 'completed'
                CHECK (status IN ('completed', 'failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def _capture_binding_status_list_sql() -> str:
    return ", ".join(f"'{status}'" for status in CAPTURE_BINDING_STATUSES)


def _create_display_object_table_sql() -> str:
    return f"""
        CREATE TABLE {DISPLAY_OBJECT_TABLE} (
            display_object_id TEXT PRIMARY KEY,
            canonical_capture_instance_id TEXT,
            capture_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_capture_instance_table_sql() -> str:
    return f"""
        CREATE TABLE {CAPTURE_INSTANCE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_instance_id TEXT NOT NULL UNIQUE,
            display_object_id TEXT,
            task_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL DEFAULT 'hololens',
            timestamp TEXT,
            yolo_object_id TEXT,
            binding_status TEXT NOT NULL DEFAULT 'unbound'
                CHECK (binding_status IN ({_capture_binding_status_list_sql()})),
            binding_reason TEXT,
            identity_distance REAL,
            candidate_scores_json TEXT NOT NULL DEFAULT '[]',
            feature_json TEXT NOT NULL DEFAULT '{{}}',
            evidence_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_capture_binding_log_table_sql() -> str:
    return f"""
        CREATE TABLE {CAPTURE_BINDING_LOG_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_instance_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            display_object_id TEXT,
            decision TEXT NOT NULL,
            binding_status TEXT NOT NULL
                CHECK (binding_status IN ({_capture_binding_status_list_sql()})),
            reason TEXT,
            candidate_scores_json TEXT NOT NULL DEFAULT '[]',
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_history_placement_request_table_sql() -> str:
    return f"""
        CREATE TABLE {HISTORY_PLACEMENT_REQUEST_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL UNIQUE,
            request_timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'completed', 'cancelled', 'failed', 'cleanup_pending')),
            startup_session_id TEXT,
            target_time TEXT,
            model_limit INTEGER,
            selected_task_count INTEGER NOT NULL DEFAULT 0,
            result_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
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

    required_columns = {
        "startup_session_id",
        "aruco_coordinate_synced",
        "task_timestamp",
        "artifact_schema_version",
        "debug_enabled",
        "logs_enabled",
    }
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
    has_task_timestamp = "task_timestamp" in legacy_columns
    has_artifact_schema_version = "artifact_schema_version" in legacy_columns
    has_debug_enabled = "debug_enabled" in legacy_columns
    has_logs_enabled = "logs_enabled" in legacy_columns

    startup_select = "startup_session_id" if has_startup_session_id else "NULL"
    synced_select = "aruco_coordinate_synced" if has_aruco_coordinate_synced else "0"
    timestamp_select = "task_timestamp" if has_task_timestamp else "NULL"
    schema_version_select = "artifact_schema_version" if has_artifact_schema_version else "1"
    debug_enabled_select = "debug_enabled" if has_debug_enabled else "1"
    logs_enabled_select = "logs_enabled" if has_logs_enabled else "1"

    conn.execute(
        f"""
        INSERT INTO {TABLE_NAME} (
            id,
            task_id,
            status,
            json_path,
            task_timestamp,
            startup_session_id,
            aruco_coordinate_synced,
            artifact_schema_version,
            debug_enabled,
            logs_enabled,
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
            {timestamp_select},
            {startup_select},
            {synced_select},
            {schema_version_select},
            {debug_enabled_select},
            {logs_enabled_select},
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
    global _SCHEMA_INITIALIZED
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

        if _table_sql(conn, TASK_TIMING_EVENT_TABLE) is None:
            conn.execute(_create_task_timing_event_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{TASK_TIMING_EVENT_TABLE}_task_stage
            ON {TASK_TIMING_EVENT_TABLE} (task_id, stage_name, event_name)
            """
        )

        if _table_sql(conn, AI_MODEL_TIMING_TABLE) is None:
            conn.execute(_create_ai_model_timing_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{AI_MODEL_TIMING_TABLE}_task
            ON {AI_MODEL_TIMING_TABLE} (task_id, service_name, timing_kind)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{AI_MODEL_TIMING_TABLE}_service
            ON {AI_MODEL_TIMING_TABLE} (service_name, timing_kind, started_at)
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

        if _table_sql(conn, DISPLAY_OBJECT_TABLE) is None:
            conn.execute(_create_display_object_table_sql())
        if _table_sql(conn, CAPTURE_INSTANCE_TABLE) is None:
            conn.execute(_create_capture_instance_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{CAPTURE_INSTANCE_TABLE}_display_object
            ON {CAPTURE_INSTANCE_TABLE} (display_object_id, binding_status, updated_at)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{CAPTURE_INSTANCE_TABLE}_task
            ON {CAPTURE_INSTANCE_TABLE} (task_id)
            """
        )
        if _table_sql(conn, CAPTURE_BINDING_LOG_TABLE) is None:
            conn.execute(_create_capture_binding_log_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{CAPTURE_BINDING_LOG_TABLE}_capture
            ON {CAPTURE_BINDING_LOG_TABLE} (capture_instance_id, created_at)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{CAPTURE_BINDING_LOG_TABLE}_task
            ON {CAPTURE_BINDING_LOG_TABLE} (task_id, created_at)
            """
        )

        if _table_sql(conn, HISTORY_PLACEMENT_REQUEST_TABLE) is None:
            conn.execute(_create_history_placement_request_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{HISTORY_PLACEMENT_REQUEST_TABLE}_status_created
            ON {HISTORY_PLACEMENT_REQUEST_TABLE} (status, created_at)
            """
        )

        if ARUCO_SYNC_MARKER_REGISTRY_ON_START:
            _sync_marker_registry_from_reference_folder(conn)

        _normalize_stored_paths(conn)
        conn.commit()
        _SCHEMA_INITIALIZED = True


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


def record_task_timing_event(
    *,
    task_id: str,
    stage_name: str,
    event_name: str,
    duration_ms: int | float,
    started_at: str | None = None,
    completed_at: str | None = None,
    started_at_unix: float | None = None,
    completed_at_unix: float | None = None,
    status: str = "completed",
    detail: Any | None = None,
    error_message: str | None = None,
) -> None:
    _ensure_schema_initialized()
    if status not in {"completed", "failed"}:
        status = "failed" if error_message else "completed"
    if completed_at is None:
        completed_at = _utc_text_from_timestamp(completed_at_unix) if completed_at_unix is not None else _utc_now_text()
    if started_at is None:
        started_at = _utc_text_from_timestamp(started_at_unix) if started_at_unix is not None else completed_at
    detail_json = json.dumps(detail if detail is not None else {}, ensure_ascii=False)
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {TASK_TIMING_EVENT_TABLE} (
                task_id, stage_name, event_name, status, started_at, completed_at,
                duration_ms, detail_json, error_message, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(task_id),
                str(stage_name),
                str(event_name),
                status,
                started_at,
                completed_at,
                int(round(float(duration_ms))),
                detail_json,
                error_message,
            ),
        )
        conn.commit()


def get_task_timing_events(task_id: str) -> List[Dict[str, Any]]:
    _ensure_schema_initialized()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TASK_TIMING_EVENT_TABLE}
            WHERE task_id = ?
            ORDER BY started_at ASC, id ASC
            """,
            (str(task_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def record_ai_model_timing(
    *,
    service_name: str,
    timing_kind: str,
    duration_ms: int | float,
    stage_name: str | None = None,
    task_id: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    started_at_unix: float | None = None,
    completed_at_unix: float | None = None,
    status: str = "completed",
    detail: Any | None = None,
    error_message: str | None = None,
) -> None:
    _ensure_schema_initialized()
    if timing_kind not in {"initialization", "task"}:
        raise ValueError(f"unsupported AI model timing kind: {timing_kind}")
    if status not in {"completed", "failed"}:
        status = "failed" if error_message else "completed"
    if completed_at is None:
        completed_at = _utc_text_from_timestamp(completed_at_unix) if completed_at_unix is not None else _utc_now_text()
    if started_at is None:
        started_at = _utc_text_from_timestamp(started_at_unix) if started_at_unix is not None else completed_at
    detail_json = json.dumps(detail if detail is not None else {}, ensure_ascii=False)
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {AI_MODEL_TIMING_TABLE} (
                service_name, timing_kind, stage_name, task_id, status, started_at,
                completed_at, duration_ms, detail_json, error_message, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(service_name),
                timing_kind,
                str(stage_name) if stage_name is not None else None,
                str(task_id) if task_id is not None else None,
                status,
                started_at,
                completed_at,
                int(round(float(duration_ms))),
                detail_json,
                error_message,
            ),
        )
        conn.commit()


def get_ai_model_timings_for_task(task_id: str) -> List[Dict[str, Any]]:
    _ensure_schema_initialized()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {AI_MODEL_TIMING_TABLE}
            WHERE task_id = ?
            ORDER BY started_at ASC, id ASC
            """,
            (str(task_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def create_display_object(
    *,
    display_object_id: str | None = None,
    canonical_capture_instance_id: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    initialize_task_table()
    display_object_id = str(display_object_id or "").strip() or None
    if display_object_id is None:
        import uuid

        display_object_id = str(uuid.uuid4())
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {DISPLAY_OBJECT_TABLE} (
                display_object_id,
                canonical_capture_instance_id,
                notes,
                updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(display_object_id) DO UPDATE SET
                canonical_capture_instance_id = COALESCE(
                    {DISPLAY_OBJECT_TABLE}.canonical_capture_instance_id,
                    excluded.canonical_capture_instance_id
                ),
                notes = COALESCE(excluded.notes, {DISPLAY_OBJECT_TABLE}.notes),
                updated_at = CURRENT_TIMESTAMP
            """,
            (display_object_id, canonical_capture_instance_id, notes),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
    return dict(row)


def get_display_object(display_object_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_TABLE} WHERE display_object_id = ?",
            (str(display_object_id),),
        ).fetchone()
    return _row_to_dict(row)


def get_capture_instance(capture_instance_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {CAPTURE_INSTANCE_TABLE} WHERE capture_instance_id = ?",
            (str(capture_instance_id),),
        ).fetchone()
    return _row_to_dict(row)


def get_capture_instance_by_task_id(task_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {CAPTURE_INSTANCE_TABLE} WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
    return _row_to_dict(row)


def _refresh_display_object_capture_count(
    conn: sqlite3.Connection,
    display_object_id: str,
    *,
    canonical_capture_instance_id: str | None = None,
) -> None:
    if not display_object_id:
        return
    conn.execute(
        f"""
        UPDATE {DISPLAY_OBJECT_TABLE}
        SET
            capture_count = (
                SELECT COUNT(*)
                FROM {CAPTURE_INSTANCE_TABLE}
                WHERE display_object_id = ? AND binding_status = 'bound'
            ),
            canonical_capture_instance_id = COALESCE(canonical_capture_instance_id, ?),
            updated_at = CURRENT_TIMESTAMP
        WHERE display_object_id = ?
        """,
        (display_object_id, canonical_capture_instance_id, display_object_id),
    )


def upsert_capture_instance(
    *,
    capture_instance_id: str,
    task_id: str,
    display_object_id: str | None = None,
    source: str = "hololens",
    timestamp: str | None = None,
    yolo_object_id: str | None = None,
    binding_status: str = "unbound",
    binding_reason: str | None = None,
    identity_distance: float | None = None,
    candidate_scores: Any | None = None,
    feature: Any | None = None,
    evidence: Any | None = None,
) -> Dict[str, Any]:
    if binding_status not in CAPTURE_BINDING_STATUSES:
        raise ValueError(f"Invalid capture binding status: {binding_status}")
    initialize_task_table()
    capture_instance_id = str(capture_instance_id)
    task_id = str(task_id)
    display_object_id = str(display_object_id).strip() if display_object_id else None
    now = _utc_now_text()
    with _get_connection() as conn:
        old_row = conn.execute(
            f"SELECT display_object_id FROM {CAPTURE_INSTANCE_TABLE} WHERE capture_instance_id = ?",
            (capture_instance_id,),
        ).fetchone()
        old_display_object_id = str(old_row["display_object_id"] or "") if old_row else ""
        conn.execute(
            f"""
            INSERT INTO {CAPTURE_INSTANCE_TABLE} (
                capture_instance_id,
                display_object_id,
                task_id,
                source,
                timestamp,
                yolo_object_id,
                binding_status,
                binding_reason,
                identity_distance,
                candidate_scores_json,
                feature_json,
                evidence_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(capture_instance_id) DO UPDATE SET
                display_object_id = excluded.display_object_id,
                task_id = excluded.task_id,
                source = excluded.source,
                timestamp = excluded.timestamp,
                yolo_object_id = excluded.yolo_object_id,
                binding_status = excluded.binding_status,
                binding_reason = excluded.binding_reason,
                identity_distance = excluded.identity_distance,
                candidate_scores_json = excluded.candidate_scores_json,
                feature_json = excluded.feature_json,
                evidence_json = excluded.evidence_json,
                updated_at = excluded.updated_at
            """,
            (
                capture_instance_id,
                display_object_id,
                task_id,
                source,
                timestamp,
                yolo_object_id,
                binding_status,
                binding_reason,
                identity_distance,
                json.dumps(candidate_scores if candidate_scores is not None else [], ensure_ascii=False),
                json.dumps(feature if feature is not None else {}, ensure_ascii=False),
                json.dumps(evidence if evidence is not None else {}, ensure_ascii=False),
                now,
            ),
        )
        if old_display_object_id and old_display_object_id != (display_object_id or ""):
            _refresh_display_object_capture_count(conn, old_display_object_id)
        if display_object_id:
            _refresh_display_object_capture_count(
                conn,
                display_object_id,
                canonical_capture_instance_id=capture_instance_id,
            )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {CAPTURE_INSTANCE_TABLE} WHERE capture_instance_id = ?",
            (capture_instance_id,),
        ).fetchone()
    return dict(row)


def list_identity_candidate_captures(
    *,
    limit: int = 500,
    exclude_capture_instance_id: str | None = None,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit or 500), 5000))
    where = ["display_object_id IS NOT NULL", "binding_status = 'bound'"]
    params: List[Any] = []
    if exclude_capture_instance_id:
        where.append("capture_instance_id != ?")
        params.append(str(exclude_capture_instance_id))
    params.append(limit)
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {CAPTURE_INSTANCE_TABLE}
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [dict(row) for row in rows]


def update_capture_instance_feature(
    capture_instance_id: str,
    *,
    feature: Any,
    evidence: Any | None = None,
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    capture_instance_id = str(capture_instance_id or "").strip()
    if not capture_instance_id:
        return None
    now = _utc_now_text()
    with _get_connection() as conn:
        if evidence is None:
            conn.execute(
                f"""
                UPDATE {CAPTURE_INSTANCE_TABLE}
                SET feature_json = ?, updated_at = ?
                WHERE capture_instance_id = ?
                """,
                (json.dumps(feature if feature is not None else {}, ensure_ascii=False), now, capture_instance_id),
            )
        else:
            conn.execute(
                f"""
                UPDATE {CAPTURE_INSTANCE_TABLE}
                SET feature_json = ?, evidence_json = ?, updated_at = ?
                WHERE capture_instance_id = ?
                """,
                (
                    json.dumps(feature if feature is not None else {}, ensure_ascii=False),
                    json.dumps(evidence if evidence is not None else {}, ensure_ascii=False),
                    now,
                    capture_instance_id,
                ),
            )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {CAPTURE_INSTANCE_TABLE} WHERE capture_instance_id = ?",
            (capture_instance_id,),
        ).fetchone()
    return _row_to_dict(row)


def record_capture_binding_log(
    *,
    capture_instance_id: str,
    task_id: str,
    display_object_id: str | None,
    decision: str,
    binding_status: str,
    reason: str | None = None,
    candidate_scores: Any | None = None,
    detail: Any | None = None,
) -> None:
    if binding_status not in CAPTURE_BINDING_STATUSES:
        raise ValueError(f"Invalid capture binding status: {binding_status}")
    initialize_task_table()
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {CAPTURE_BINDING_LOG_TABLE} (
                capture_instance_id,
                task_id,
                display_object_id,
                decision,
                binding_status,
                reason,
                candidate_scores_json,
                detail_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(capture_instance_id),
                str(task_id),
                str(display_object_id) if display_object_id else None,
                str(decision),
                binding_status,
                reason,
                json.dumps(candidate_scores if candidate_scores is not None else [], ensure_ascii=False),
                json.dumps(detail if detail is not None else {}, ensure_ascii=False),
            ),
        )
        conn.commit()


def get_capture_binding_logs(capture_instance_id: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit or 50), 500))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {CAPTURE_BINDING_LOG_TABLE}
            WHERE capture_instance_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(capture_instance_id), limit),
        ).fetchall()
    return [dict(row) for row in rows]


def create_history_placement_request(
    *,
    request_id: str,
    request_timestamp: str,
    startup_session_id: str | None = None,
    target_time: str | None = None,
    model_limit: int | None = None,
) -> Dict[str, Any]:
    initialize_task_table()
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {HISTORY_PLACEMENT_REQUEST_TABLE} (
                request_id, request_timestamp, status, startup_session_id, target_time, model_limit
            )
            VALUES (?, ?, 'running', ?, ?, ?)
            """,
            (request_id, request_timestamp, startup_session_id, target_time, model_limit),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {HISTORY_PLACEMENT_REQUEST_TABLE} WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    return dict(row)


def update_history_placement_request(
    request_id: str,
    *,
    status: str,
    selected_task_count: int | None = None,
    result_count: int | None = None,
    error_message: str | None = None,
) -> bool:
    if status not in {'running', 'completed', 'cancelled', 'failed', 'cleanup_pending'}:
        raise ValueError(f"Invalid history placement request status: {status}")
    initialize_task_table()
    set_parts = ["status = ?", "updated_at = CURRENT_TIMESTAMP", "error_message = ?"]
    params: List[Any] = [status, error_message]
    if selected_task_count is not None:
        set_parts.append("selected_task_count = ?")
        params.append(int(selected_task_count))
    if result_count is not None:
        set_parts.append("result_count = ?")
        params.append(int(result_count))
    if status in {"completed", "cancelled", "failed", "cleanup_pending"}:
        set_parts.append("completed_at = CURRENT_TIMESTAMP")
    params.append(request_id)
    with _get_connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE {HISTORY_PLACEMENT_REQUEST_TABLE}
            SET {', '.join(set_parts)}
            WHERE request_id = ?
            """,
            tuple(params),
        )
        conn.commit()
    return cursor.rowcount > 0


def get_latest_history_placement_request(status: str = "completed") -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {HISTORY_PLACEMENT_REQUEST_TABLE}
            WHERE status = ?
            ORDER BY request_timestamp DESC, id DESC
            LIMIT 1
            """,
            (status,),
        ).fetchone()
    return _row_to_dict(row)


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
    task_timestamp: str | None = None,
    status: str = "pending",
    artifact_schema_version: int = 1,
    debug_enabled: bool = True,
    logs_enabled: bool = True,
) -> Dict[str, Any]:
    initialize_task_table()
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    json_path_str = normalize_path_for_storage(json_path)
    startup_session_id = str(startup_session_id or "").strip() or None
    task_timestamp = str(task_timestamp or "").strip() or None
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                task_id,
                status,
                json_path,
                task_timestamp,
                startup_session_id,
                aruco_coordinate_synced,
                artifact_schema_version,
                debug_enabled,
                logs_enabled
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                task_id,
                status,
                json_path_str,
                task_timestamp,
                startup_session_id,
                int(artifact_schema_version),
                1 if debug_enabled else 0,
                1 if logs_enabled else 0,
            ),
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
              AND status != 'uploading'
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
    raw_limit = int(limit if limit is not None else 5)
    bounded_limit = None if raw_limit <= 0 else max(1, min(raw_limit, 50))
    where_clauses = ["status = 'completed'"]
    params: List[Any] = []
    if startup_session_id:
        where_clauses.append("startup_session_id = ?")
        params.append(startup_session_id)
    if require_aruco_coordinate_synced:
        where_clauses.append("aruco_coordinate_synced = 1")

    limit_sql = "" if bounded_limit is None else "LIMIT ?"
    query_params = list(params)
    if bounded_limit is not None:
        query_params.append(bounded_limit)

    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY id DESC
            {limit_sql}
            """,
            tuple(query_params),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_completed_task_for_display_object(display_object_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    display_object_id = str(display_object_id or "").strip()
    if not display_object_id:
        return None
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT t.*
            FROM {TABLE_NAME} AS t
            INNER JOIN {CAPTURE_INSTANCE_TABLE} AS c
                ON c.task_id = t.task_id
            WHERE c.display_object_id = ?
              AND c.binding_status = 'bound'
              AND t.status = 'completed'
            ORDER BY t.id DESC
            LIMIT 1
            """,
            (display_object_id,),
        ).fetchone()
    return _row_to_dict(row)


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


def get_tasks_for_startup_statuses(
    startup_session_id: str,
    statuses: Iterable[str],
) -> List[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
    normalized_statuses = [str(status or "").strip() for status in statuses]
    normalized_statuses = [status for status in normalized_statuses if status in ALLOWED_STATUSES]
    if not startup_session_id or not normalized_statuses:
        return []

    placeholders = ", ".join("?" for _ in normalized_statuses)
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE startup_session_id = ?
              AND status IN ({placeholders})
            ORDER BY id DESC
            """,
            tuple([startup_session_id] + normalized_statuses),
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
              AND status != 'uploading'
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

    if status not in {"pending", "uploading", "upload_failed"}:
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
