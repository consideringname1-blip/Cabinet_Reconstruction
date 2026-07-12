import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

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
_TASK_SCHEMA_UPGRADE_TABLE = "tasks__current_schema"
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
DISPLAY_OBJECT_STATE_TABLE = "display_object_states"
DISPLAY_OBJECT_MODEL_REVISION_TABLE = "display_object_model_revisions"
DISPLAY_OBJECT_POSE_HISTORY_TABLE = "display_object_pose_history"
REALTIME_TRACKING_EVENT_TABLE = "realtime_tracking_events"
AUXILIARY_JOB_TABLE = "auxiliary_jobs"
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
    "model_generation",
    "depthpointcloud",
    "modelscale",
    "object_alignment",
    "pose",
    "aruco_sync",
    "runtime_mesh",
    "model_bounds",
    "display_identity",
    "completed",
    "aruco_completed",
    "failed",
)
TERMINAL_STATUSES = ("completed", "aruco_completed", "failed", "upload_failed")
_TASK_TABLE_COLUMNS = (
    "id",
    "task_id",
    "status",
    "json_path",
    "task_timestamp",
    "startup_session_id",
    "aruco_coordinate_synced",
    "debug_enabled",
    "logs_enabled",
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
    "error_message",
)
_SCHEMA_INITIALIZED = False


@contextmanager
def _get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transactional SQLite connection and always close its handle.

    ``sqlite3.Connection.__exit__`` commits or rolls back but does not close the
    connection.  Every caller in this module uses ``with _get_connection()``,
    so owning the close here avoids descriptor/file-lock leaks without changing
    transaction semantics at the call sites.
    """

    ensure_database_root()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


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


def _create_task_table_sql(table_name: str = TABLE_NAME) -> str:
    if table_name not in {TABLE_NAME, _TASK_SCHEMA_UPGRADE_TABLE}:
        raise ValueError(f"Unsupported task table name: {table_name}")
    return f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ({_status_list_sql()})),
            json_path TEXT NOT NULL,
            task_timestamp TEXT,
            startup_session_id TEXT,
            aruco_coordinate_synced INTEGER NOT NULL DEFAULT 0,
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


def _create_display_object_state_table_sql() -> str:
    return f"""
        CREATE TABLE {DISPLAY_OBJECT_STATE_TABLE} (
            display_object_id TEXT PRIMARY KEY,
            active_model_revision INTEGER NOT NULL DEFAULT 0,
            active_model_task_id TEXT,
            active_model_asset_hash TEXT,
            latest_hololens_pose_revision INTEGER NOT NULL DEFAULT 0,
            latest_hololens_pose_aruco_json TEXT,
            latest_hololens_task_id TEXT,
            latest_hololens_captured_at TEXT,
            latest_tracking_pose_revision INTEGER NOT NULL DEFAULT 0,
            latest_tracking_model_revision INTEGER NOT NULL DEFAULT 0,
            latest_tracking_pose_aruco_json TEXT,
            latest_tracking_observation_seq INTEGER NOT NULL DEFAULT 0,
            latest_body_revision INTEGER NOT NULL DEFAULT 0,
            latest_body_task_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_display_object_model_revision_table_sql() -> str:
    return f"""
        CREATE TABLE {DISPLAY_OBJECT_MODEL_REVISION_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_object_id TEXT NOT NULL,
            model_revision INTEGER NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            asset_hash TEXT,
            source TEXT NOT NULL DEFAULT 'hololens',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(display_object_id, model_revision)
        )
    """


def _create_display_object_pose_history_table_sql() -> str:
    return f"""
        CREATE TABLE {DISPLAY_OBJECT_POSE_HISTORY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_object_id TEXT NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            model_revision INTEGER NOT NULL,
            pose_revision INTEGER NOT NULL,
            pose_aruco_json TEXT NOT NULL,
            body_revision INTEGER,
            captured_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_realtime_tracking_event_table_sql() -> str:
    return f"""
        CREATE TABLE {REALTIME_TRACKING_EVENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_object_id TEXT,
            startup_session_id TEXT,
            ingress_session_id TEXT,
            shigure_object_id TEXT,
            observation_seq INTEGER NOT NULL DEFAULT 0,
            tracking_epoch INTEGER NOT NULL DEFAULT 0,
            mode_epoch INTEGER NOT NULL DEFAULT 0,
            model_revision INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            reason TEXT,
            source_stamp_json TEXT,
            pose_aruco_json TEXT,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_auxiliary_job_table_sql() -> str:
    return f"""
        CREATE TABLE {AUXILIARY_JOB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL,
            branch_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
            result_path TEXT,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(task_id, branch_name)
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


def _normalized_table_definition(sql: str) -> str:
    definition_start = sql.find("(")
    if definition_start < 0:
        return ""
    return " ".join(sql[definition_start:].split())


def _task_table_schema_is_current(conn: sqlite3.Connection) -> bool:
    current_sql = _table_sql(conn, TABLE_NAME)
    if current_sql is None:
        return False
    expected_sql = _create_task_table_sql()
    return _normalized_table_definition(current_sql) == _normalized_table_definition(expected_sql)


def _unsupported_task_status_error(status: str) -> str:
    return (
        f'Task was failed during schema upgrade because status "{status}" '
        "is not part of the current protocol."
    )


def _rebuild_task_table_with_current_schema(conn: sqlite3.Connection) -> None:
    existing_columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    }
    missing_columns = set(_TASK_TABLE_COLUMNS) - existing_columns
    if missing_columns:
        raise RuntimeError(
            "Cannot upgrade tasks table; required columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    rows = conn.execute(
        f"SELECT {', '.join(_TASK_TABLE_COLUMNS)} FROM {TABLE_NAME} ORDER BY id ASC"
    ).fetchall()
    migrated_rows = []
    allowed_statuses = set(ALLOWED_STATUSES)
    migration_time = _utc_now_text()
    for row in rows:
        values = dict(row)
        old_status = str(values["status"])
        if old_status not in allowed_statuses:
            reason = _unsupported_task_status_error(old_status)
            existing_error = str(values["error_message"] or "").strip()
            values["status"] = "failed"
            values["updated_at"] = migration_time
            values["error_message"] = f"{existing_error}\n{reason}" if existing_error else reason
        migrated_rows.append(tuple(values[column] for column in _TASK_TABLE_COLUMNS))

    conn.execute(f"DROP TABLE IF EXISTS {_TASK_SCHEMA_UPGRADE_TABLE}")
    conn.execute(_create_task_table_sql(_TASK_SCHEMA_UPGRADE_TABLE))
    if migrated_rows:
        placeholders = ", ".join("?" for _ in _TASK_TABLE_COLUMNS)
        conn.executemany(
            f"""
            INSERT INTO {_TASK_SCHEMA_UPGRADE_TABLE} ({', '.join(_TASK_TABLE_COLUMNS)})
            VALUES ({placeholders})
            """,
            migrated_rows,
        )
    conn.execute(f"DROP TABLE {TABLE_NAME}")
    conn.execute(
        f"ALTER TABLE {_TASK_SCHEMA_UPGRADE_TABLE} RENAME TO {TABLE_NAME}"
    )


def _load_aruco_template_config() -> Dict[str, Any]:
    if not ARUCO_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"ArUco marker config not found: {ARUCO_TEMPLATE_PATH}")
    try:
        with ARUCO_TEMPLATE_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ArUco marker config JSON: {exc}") from exc

    if not isinstance(loaded, dict) or set(loaded) != {"markers"}:
        raise ValueError("ArUco marker config must contain exactly one top-level field: markers")
    if not isinstance(loaded["markers"], list) or not loaded["markers"]:
        raise ValueError("ArUco marker config markers must be a non-empty array")
    return loaded


def _marker_configs_by_id(template: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    required_fields = {
        "marker_id",
        "dictionary",
        "marker_size_mm",
        "reference_image_name",
        "enabled",
    }
    configs: Dict[int, Dict[str, Any]] = {}
    image_names: set[str] = set()
    for index, marker in enumerate(template["markers"]):
        field_name = f"markers[{index}]"
        if not isinstance(marker, dict) or set(marker) != required_fields:
            raise ValueError(
                f"{field_name} must contain exactly: {', '.join(sorted(required_fields))}"
            )

        marker_id = marker["marker_id"]
        if isinstance(marker_id, bool) or not isinstance(marker_id, int) or marker_id < 0:
            raise ValueError(f"{field_name}.marker_id must be a non-negative integer")
        if marker_id in configs:
            raise ValueError(f"Duplicate ArUco marker_id: {marker_id}")

        dictionary = marker["dictionary"]
        if not isinstance(dictionary, str) or not dictionary.strip():
            raise ValueError(f"{field_name}.dictionary must be a non-empty string")

        marker_size_mm = marker["marker_size_mm"]
        if (
            isinstance(marker_size_mm, bool)
            or not isinstance(marker_size_mm, (int, float))
            or not math.isfinite(float(marker_size_mm))
            or float(marker_size_mm) <= 0.0
        ):
            raise ValueError(f"{field_name}.marker_size_mm must be a positive finite number")

        image_name = marker["reference_image_name"]
        if (
            not isinstance(image_name, str)
            or not image_name.strip()
            or Path(image_name).name != image_name
            or Path(image_name).suffix.lower() not in {".png", ".jpg", ".jpeg"}
        ):
            raise ValueError(f"{field_name}.reference_image_name must be a local PNG or JPEG filename")
        if image_name in image_names:
            raise ValueError(f"Duplicate ArUco reference_image_name: {image_name}")

        enabled = marker["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError(f"{field_name}.enabled must be a boolean")

        normalized = {
            "marker_id": marker_id,
            "dictionary": dictionary.strip(),
            "marker_size_mm": float(marker_size_mm),
            "reference_image_name": image_name,
            "enabled": enabled,
        }
        configs[marker_id] = normalized
        image_names.add(image_name)
    return configs


def _sync_marker_registry_from_reference_folder(conn: sqlite3.Connection) -> int:
    template = _load_aruco_template_config()
    marker_configs = _marker_configs_by_id(template)

    seen_marker_ids: set[int] = set()
    synced_count = 0
    for marker_id, marker_config in marker_configs.items():
        reference_image_name = marker_config["reference_image_name"]
        image_path = ARUCO_REFERENCE_ROOT / reference_image_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"ArUco reference image for marker {marker_id} not found: {image_path}"
            )
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
                marker_config["dictionary"],
                marker_config["marker_size_mm"],
                reference_image_name,
                1 if marker_config["enabled"] else 0,
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
    if _SCHEMA_INITIALIZED:
        return
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")

        if _table_sql(conn, TABLE_NAME) is None:
            conn.execute(_create_task_table_sql())
        elif not _task_table_schema_is_current(conn):
            _rebuild_task_table_with_current_schema(conn)

        # This request table belonged to the removed server-side history placement
        # pipeline.  Current history/live mode is represented by display-object
        # state, so keeping this table would imply a protocol that no longer exists.
        conn.execute("DROP TABLE IF EXISTS history_placement_requests")

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

        if _table_sql(conn, DISPLAY_OBJECT_STATE_TABLE) is None:
            conn.execute(_create_display_object_state_table_sql())
        if _table_sql(conn, DISPLAY_OBJECT_MODEL_REVISION_TABLE) is None:
            conn.execute(_create_display_object_model_revision_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_MODEL_REVISION_TABLE}_display_revision
            ON {DISPLAY_OBJECT_MODEL_REVISION_TABLE} (display_object_id, model_revision DESC)
            """
        )
        if _table_sql(conn, DISPLAY_OBJECT_POSE_HISTORY_TABLE) is None:
            conn.execute(_create_display_object_pose_history_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_POSE_HISTORY_TABLE}_display_revision
            ON {DISPLAY_OBJECT_POSE_HISTORY_TABLE} (display_object_id, pose_revision DESC)
            """
        )
        if _table_sql(conn, REALTIME_TRACKING_EVENT_TABLE) is None:
            conn.execute(_create_realtime_tracking_event_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{REALTIME_TRACKING_EVENT_TABLE}_display_created
            ON {REALTIME_TRACKING_EVENT_TABLE} (display_object_id, created_at DESC)
            """
        )
        if _table_sql(conn, AUXILIARY_JOB_TABLE) is None:
            conn.execute(_create_auxiliary_job_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{AUXILIARY_JOB_TABLE}_task_branch
            ON {AUXILIARY_JOB_TABLE} (task_id, branch_name, updated_at DESC)
            """
        )

        if ARUCO_SYNC_MARKER_REGISTRY_ON_START:
            _sync_marker_registry_from_reference_folder(conn)

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
                debug_enabled,
                logs_enabled
            )
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                task_id,
                status,
                json_path_str,
                task_timestamp,
                startup_session_id,
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


def commit_display_object_capture_state(
    *,
    display_object_id: str,
    capture_task_id: str,
    pose_aruco: Any,
    captured_at: str | None = None,
    generated_new_model: bool,
    active_model_task_id: str | None = None,
    asset_hash: str | None = None,
) -> Dict[str, Any]:
    """Atomically commit a HoloLens capture and its pinned model revision.

    A reused geometry capture advances only the HoloLens pose revision.  A
    successfully regenerated geometry advances both the model and pose
    revisions.  Replaying the same capture task is idempotent.
    """

    initialize_task_table()
    display_object_id = str(display_object_id or "").strip()
    capture_task_id = str(capture_task_id or "").strip()
    if not display_object_id or not capture_task_id:
        raise ValueError("display_object_id and capture_task_id are required")
    if not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco is required")
    model_task_id = str(active_model_task_id or capture_task_id).strip()
    capture_time = str(captured_at or _utc_now_text())
    now = _utc_now_text()

    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"""
            INSERT INTO {DISPLAY_OBJECT_STATE_TABLE} (display_object_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(display_object_id) DO NOTHING
            """,
            (display_object_id, now),
        )
        state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
        existing_history = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE} WHERE task_id = ?",
            (capture_task_id,),
        ).fetchone()
        if existing_history is not None:
            existing_display_object_id = str(existing_history["display_object_id"] or "").strip()
            if existing_display_object_id != display_object_id:
                conn.rollback()
                raise ValueError(
                    f"capture task {capture_task_id} is already committed to "
                    f"display object {existing_display_object_id}"
                )
            serialized_pose = json.dumps(pose_aruco, ensure_ascii=False)
            conn.execute(
                f"""
                UPDATE {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
                SET pose_aruco_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (serialized_pose, now, capture_task_id),
            )
            # ArUco retro-sync can legitimately re-express an already committed
            # capture.  Refresh the latest pointer without allocating another
            # user-visible pose revision.
            if str(state["latest_hololens_task_id"] or "").strip() == capture_task_id:
                conn.execute(
                    f"""
                    UPDATE {DISPLAY_OBJECT_STATE_TABLE}
                    SET latest_hololens_pose_aruco_json = ?, updated_at = ?
                    WHERE display_object_id = ?
                    """,
                    (serialized_pose, now, display_object_id),
                )
            conn.commit()
            row = conn.execute(
                f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
                (display_object_id,),
            ).fetchone()
            return dict(row)

        active_revision = int(state["active_model_revision"] or 0)
        resolved_model_task_id = str(state["active_model_task_id"] or "").strip() or None
        if generated_new_model or active_revision <= 0:
            revision_row = conn.execute(
                f"SELECT * FROM {DISPLAY_OBJECT_MODEL_REVISION_TABLE} WHERE task_id = ?",
                (model_task_id,),
            ).fetchone()
            if revision_row is None:
                maximum = conn.execute(
                    f"SELECT COALESCE(MAX(model_revision), 0) AS value FROM {DISPLAY_OBJECT_MODEL_REVISION_TABLE} WHERE display_object_id = ?",
                    (display_object_id,),
                ).fetchone()
                active_revision = int(maximum["value"] or 0) + 1
                conn.execute(
                    f"""
                    INSERT INTO {DISPLAY_OBJECT_MODEL_REVISION_TABLE} (
                        display_object_id, model_revision, task_id, asset_hash, source
                    ) VALUES (?, ?, ?, ?, 'hololens')
                    """,
                    (display_object_id, active_revision, model_task_id, asset_hash),
                )
            else:
                if str(revision_row["display_object_id"] or "").strip() != display_object_id:
                    conn.rollback()
                    raise ValueError(
                        f"model task {model_task_id} is already committed to another display object"
                    )
                active_revision = int(revision_row["model_revision"])
            resolved_model_task_id = model_task_id
        elif model_task_id:
            revision_row = conn.execute(
                f"SELECT * FROM {DISPLAY_OBJECT_MODEL_REVISION_TABLE} WHERE task_id = ?",
                (model_task_id,),
            ).fetchone()
            if revision_row is not None:
                if str(revision_row["display_object_id"] or "").strip() != display_object_id:
                    conn.rollback()
                    raise ValueError(
                        f"model task {model_task_id} is already committed to another display object"
                    )
                active_revision = int(revision_row["model_revision"])
                resolved_model_task_id = model_task_id

        pose_revision = int(state["latest_hololens_pose_revision"] or 0) + 1
        conn.execute(
            f"""
            INSERT INTO {DISPLAY_OBJECT_POSE_HISTORY_TABLE} (
                display_object_id, task_id, model_revision, pose_revision,
                pose_aruco_json, body_revision, captured_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                display_object_id,
                capture_task_id,
                active_revision,
                pose_revision,
                json.dumps(pose_aruco, ensure_ascii=False),
                capture_time,
                now,
            ),
        )
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET active_model_revision = ?,
                active_model_task_id = ?,
                active_model_asset_hash = COALESCE(?, active_model_asset_hash),
                latest_hololens_pose_revision = ?,
                latest_hololens_pose_aruco_json = ?,
                latest_hololens_task_id = ?,
                latest_hololens_captured_at = ?,
                latest_tracking_model_revision = 0,
                latest_tracking_pose_aruco_json = NULL,
                latest_tracking_observation_seq = 0,
                updated_at = ?
            WHERE display_object_id = ?
            """,
            (
                active_revision,
                resolved_model_task_id,
                asset_hash,
                pose_revision,
                json.dumps(pose_aruco, ensure_ascii=False),
                capture_task_id,
                capture_time,
                now,
                display_object_id,
            ),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
    return dict(row)


def get_display_object_state(display_object_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (str(display_object_id),),
        ).fetchone()
    return _row_to_dict(row)


def list_display_object_states(*, limit: int) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit), 50))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {DISPLAY_OBJECT_STATE_TABLE}
            WHERE latest_hololens_pose_aruco_json IS NOT NULL
            ORDER BY latest_hololens_captured_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_display_object_pose_history(display_object_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit or 50), 1000))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
            WHERE display_object_id = ?
            ORDER BY pose_revision DESC
            LIMIT ?
            """,
            (str(display_object_id), limit),
        ).fetchall()
    return [dict(row) for row in rows]


def commit_realtime_tracking_pose(
    *,
    display_object_id: str,
    model_revision: int,
    hololens_pose_revision: int,
    observation_seq: int,
    pose_aruco: Any,
) -> Optional[Dict[str, Any]]:
    """Commit a live pose only against its exact HoloLens anchor revision."""

    initialize_task_table()
    if not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco is required")
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (str(display_object_id),),
        ).fetchone()
        if (
            state is None
            or int(state["active_model_revision"] or 0) != int(model_revision)
            or int(state["latest_hololens_pose_revision"] or 0) != int(hololens_pose_revision)
        ):
            conn.rollback()
            return None
        pose_revision = int(state["latest_tracking_pose_revision"] or 0) + 1
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET latest_tracking_pose_revision = ?,
                latest_tracking_model_revision = ?,
                latest_tracking_pose_aruco_json = ?,
                latest_tracking_observation_seq = ?,
                updated_at = ?
            WHERE display_object_id = ?
                AND active_model_revision = ?
                AND latest_hololens_pose_revision = ?
            """,
            (
                pose_revision,
                int(model_revision),
                json.dumps(pose_aruco, ensure_ascii=False),
                int(observation_seq),
                now,
                str(display_object_id),
                int(model_revision),
                int(hololens_pose_revision),
            ),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (str(display_object_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def record_realtime_tracking_event(
    *,
    status: str,
    display_object_id: str | None = None,
    startup_session_id: str | None = None,
    ingress_session_id: str | None = None,
    shigure_object_id: str | None = None,
    observation_seq: int = 0,
    tracking_epoch: int = 0,
    mode_epoch: int = 0,
    model_revision: int = 0,
    reason: str | None = None,
    source_stamp: Any = None,
    pose_aruco: Any = None,
    detail: Any = None,
) -> int:
    initialize_task_table()
    with _get_connection() as conn:
        cursor = conn.execute(
            f"""
            INSERT INTO {REALTIME_TRACKING_EVENT_TABLE} (
                display_object_id, startup_session_id, ingress_session_id,
                shigure_object_id, observation_seq, tracking_epoch, mode_epoch,
                model_revision, status, reason, source_stamp_json,
                pose_aruco_json, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                display_object_id,
                startup_session_id,
                ingress_session_id,
                shigure_object_id,
                int(observation_seq),
                int(tracking_epoch),
                int(mode_epoch),
                int(model_revision),
                str(status),
                reason,
                _dump_optional_json(source_stamp),
                _dump_optional_json(pose_aruco),
                json.dumps(detail if detail is not None else {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_latest_realtime_tracking_event(display_object_id: str) -> Optional[Dict[str, Any]]:
    """Return the newest durable tracking diagnostic for one display object."""

    initialize_task_table()
    display_object_id = str(display_object_id or "").strip()
    if not display_object_id:
        return None
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {REALTIME_TRACKING_EVENT_TABLE}
            WHERE display_object_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (display_object_id,),
        ).fetchone()
    return _row_to_dict(row)


def set_latest_body_revision(
    *, display_object_id: str, task_id: str, body_revision: int | None = None
) -> Optional[Dict[str, Any]]:
    """Attach a body result to its capture without letting late jobs win.

    Auxiliary body jobs can finish out of order.  The durable "latest" pointer
    therefore follows capture ``pose_revision`` order, not completion order.
    Replaying the same successful auxiliary job is idempotent.
    """

    initialize_task_table()
    display_object_id = str(display_object_id or "").strip()
    task_id = str(task_id or "").strip()
    if not display_object_id or not task_id:
        raise ValueError("display_object_id and task_id are required")
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
        if state is None:
            conn.rollback()
            return None

        capture = conn.execute(
            f"""
            SELECT *
            FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
            WHERE display_object_id = ? AND task_id = ?
            """,
            (display_object_id, task_id),
        ).fetchone()
        if capture is None:
            # The auxiliary branch may complete before DisplayIdentity commits
            # this capture.  Its caller replays the link after that stage.
            conn.rollback()
            return None

        existing_revision = capture["body_revision"]
        revision = int(
            existing_revision
            if existing_revision is not None
            else (body_revision if body_revision is not None else capture["pose_revision"])
        )
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
            SET body_revision = ?, updated_at = ?
            WHERE display_object_id = ? AND task_id = ?
            """,
            (revision, now, display_object_id, task_id),
        )

        current_body_capture = None
        current_body_task_id = str(state["latest_body_task_id"] or "").strip()
        if current_body_task_id:
            current_body_capture = conn.execute(
                f"""
                SELECT pose_revision
                FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
                WHERE display_object_id = ? AND task_id = ?
                """,
                (display_object_id, current_body_task_id),
            ).fetchone()
        current_pose_revision = (
            int(current_body_capture["pose_revision"])
            if current_body_capture is not None
            else -1
        )
        if int(capture["pose_revision"]) >= current_pose_revision:
            conn.execute(
                f"""
                UPDATE {DISPLAY_OBJECT_STATE_TABLE}
                SET latest_body_revision = ?, latest_body_task_id = ?, updated_at = ?
                WHERE display_object_id = ?
                """,
                (revision, task_id, now, display_object_id),
            )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def upsert_auxiliary_job(
    *,
    task_id: str,
    branch_name: str,
    status: str,
    result_path: str | Path | None = None,
    detail: Any = None,
    error_message: str | None = None,
) -> Dict[str, Any]:
    allowed = {"pending", "running", "completed", "failed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"Invalid auxiliary job status: {status}")
    initialize_task_table()
    now = _utc_now_text()
    job_id = f"{task_id}:{branch_name}"
    started_at = now if status == "running" else None
    completed_at = now if status in {"completed", "failed", "cancelled"} else None
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {AUXILIARY_JOB_TABLE} (
                job_id, task_id, branch_name, status, result_path, detail_json,
                error_message, started_at, completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, branch_name) DO UPDATE SET
                status = excluded.status,
                result_path = COALESCE(excluded.result_path, result_path),
                detail_json = excluded.detail_json,
                error_message = excluded.error_message,
                started_at = COALESCE(started_at, excluded.started_at),
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                str(task_id),
                str(branch_name),
                status,
                normalize_path_for_storage(result_path) if result_path else None,
                json.dumps(detail if detail is not None else {}, ensure_ascii=False),
                error_message,
                started_at,
                completed_at,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {AUXILIARY_JOB_TABLE} WHERE task_id = ? AND branch_name = ?",
            (str(task_id), str(branch_name)),
        ).fetchone()
    return dict(row)


def get_auxiliary_jobs(task_id: str) -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {AUXILIARY_JOB_TABLE} WHERE task_id = ? ORDER BY id ASC",
            (str(task_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def list_auxiliary_jobs(*, statuses: Iterable[str] = ("pending", "running")) -> List[Dict[str, Any]]:
    initialize_task_table()
    requested = tuple(dict.fromkeys(str(value).strip() for value in statuses if str(value).strip()))
    if not requested:
        return []
    placeholders = ", ".join("?" for _ in requested)
    with _get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {AUXILIARY_JOB_TABLE} WHERE status IN ({placeholders}) ORDER BY id ASC",
            requested,
        ).fetchall()
    return [dict(row) for row in rows]
