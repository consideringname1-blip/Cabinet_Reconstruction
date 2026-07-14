import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from config import (
    SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD,
    SHIGURE_EXAMPLE_DINO_SECOND_MARGIN,
    ARUCO_ANCHOR_MARKER_ID,
    ARUCO_SYNC_MARKER_REGISTRY_ON_START,
)
from artifact_layout import (
    ARUCO_REFERENCE_ROOT,
    ARUCO_TEMPLATE_PATH,
    DATABASE_PATH,
    IDENTITY_REFERENCE_ROOT,
    ensure_database_root,
)
from task_json import normalize_path_for_storage, resolve_project_path


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
SCHEMA_METADATA_TABLE = "schema_metadata"
SHIGURE_RUNTIME_SESSION_TABLE = "shigure_runtime_sessions"
SHIGURE_SOURCE_EPOCH_TABLE = "shigure_source_epochs"
SHIGURE_OBJECT_BINDING_TABLE = "shigure_object_bindings"
SHIGURE_CANONICAL_EVENT_TABLE = "shigure_canonical_events"
OBJECT_LIFECYCLE_EVENT_TABLE = "object_lifecycle_events"
OBJECT_IDENTITY_REFERENCE_TABLE = "object_identity_references"
IDENTITY_SYNC_JOB_TABLE = "identity_sync_jobs"
SHIGURE_SCHEMA_VERSION = 2
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


def _create_display_object_state_table_sql(table_name: str = DISPLAY_OBJECT_STATE_TABLE) -> str:
    return f"""
        CREATE TABLE {table_name} (
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
            latest_spatial_observation_seq INTEGER NOT NULL DEFAULT 0,
            latest_skeleton_observation_seq INTEGER NOT NULL DEFAULT 0,
            presence TEXT NOT NULL DEFAULT 'UNKNOWN'
                CHECK (presence IN ('UNKNOWN', 'PRESENT', 'ABSENT')),
            presence_epoch INTEGER NOT NULL DEFAULT 0,
            active_shigure_binding_id TEXT,
            last_lifecycle_event_uid TEXT,
            latest_spatial_box_aruco_json TEXT,
            latest_skeleton_json TEXT,
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


def _create_display_object_pose_history_table_sql(
    table_name: str = DISPLAY_OBJECT_POSE_HISTORY_TABLE,
) -> str:
    return f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_object_id TEXT NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            model_revision INTEGER NOT NULL,
            pose_revision INTEGER NOT NULL,
            pose_aruco_json TEXT NOT NULL,
            captured_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_schema_metadata_table_sql() -> str:
    return f"""
        CREATE TABLE {SCHEMA_METADATA_TABLE} (
            schema_name TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            migrated_at TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}'
        )
    """


def _create_shigure_runtime_session_table_sql() -> str:
    return f"""
        CREATE TABLE {SHIGURE_RUNTIME_SESSION_TABLE} (
            runtime_session_id TEXT PRIMARY KEY,
            server_boot_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'CLOSED')),
            started_at TEXT NOT NULL,
            ended_at TEXT,
            close_reason TEXT,
            config_json TEXT NOT NULL DEFAULT '{{}}'
        )
    """


def _create_shigure_source_epoch_table_sql() -> str:
    return f"""
        CREATE TABLE {SHIGURE_SOURCE_EPOCH_TABLE} (
            source_epoch_id TEXT PRIMARY KEY,
            runtime_session_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'CLOSED')),
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            open_reason TEXT NOT NULL,
            close_reason TEXT,
            publisher_fingerprint_json TEXT NOT NULL DEFAULT '{{}}',
            raw_id_prefix TEXT,
            UNIQUE(runtime_session_id, generation)
        )
    """


def _create_shigure_object_binding_table_sql() -> str:
    return f"""
        CREATE TABLE {SHIGURE_OBJECT_BINDING_TABLE} (
            binding_id TEXT PRIMARY KEY,
            runtime_session_id TEXT NOT NULL,
            source_epoch_id TEXT NOT NULL,
            raw_shigure_object_id TEXT NOT NULL,
            display_object_id TEXT NOT NULL,
            binding_epoch INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVOKED')),
            established_by TEXT NOT NULL,
            established_event_uid TEXT,
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            revoke_reason TEXT,
            confidence REAL,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            UNIQUE(source_epoch_id, raw_shigure_object_id, binding_epoch)
        )
    """


def _create_shigure_canonical_event_table_sql() -> str:
    return f"""
        CREATE TABLE {SHIGURE_CANONICAL_EVENT_TABLE} (
            event_uid TEXT PRIMARY KEY,
            runtime_session_id TEXT NOT NULL,
            source_epoch_id TEXT NOT NULL,
            stamp_sec INTEGER NOT NULL,
            stamp_nanosec INTEGER NOT NULL,
            frame_id TEXT NOT NULL,
            detection_index INTEGER NOT NULL,
            action TEXT NOT NULL CHECK (action IN ('BRING_IN', 'TAKE_OUT', 'MOVE')),
            raw_shigure_object_id TEXT,
            binding_id TEXT,
            display_object_id TEXT,
            resolution_status TEXT NOT NULL
                CHECK (resolution_status IN ('RESOLVED', 'UNRESOLVED', 'AMBIGUOUS', 'CONFLICT', 'REJECTED')),
            resolution_method TEXT,
            bbox_json TEXT NOT NULL,
            collider_json TEXT,
            mask_artifact_path TEXT,
            scene_image_path TEXT,
            object_crop_path TEXT,
            skeleton_json TEXT,
            source_stamp_json TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_epoch_id, stamp_sec, stamp_nanosec, frame_id, detection_index)
        )
    """


def _create_object_lifecycle_event_table_sql() -> str:
    return f"""
        CREATE TABLE {OBJECT_LIFECYCLE_EVENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lifecycle_event_uid TEXT NOT NULL UNIQUE,
            canonical_event_uid TEXT NOT NULL UNIQUE,
            display_object_id TEXT NOT NULL,
            binding_id TEXT NOT NULL,
            source_epoch_id TEXT NOT NULL,
            raw_shigure_object_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK (action IN ('BRING_IN', 'TAKE_OUT', 'MOVE')),
            presence_before TEXT NOT NULL CHECK (presence_before IN ('UNKNOWN', 'PRESENT', 'ABSENT')),
            presence_after TEXT NOT NULL CHECK (presence_after IN ('PRESENT', 'ABSENT')),
            presence_epoch INTEGER NOT NULL,
            model_revision INTEGER NOT NULL DEFAULT 0,
            pose_revision INTEGER NOT NULL DEFAULT 0,
            pose_aruco_json TEXT,
            spatial_box_corners_aruco_json TEXT,
            scene_image_path TEXT,
            object_crop_path TEXT,
            mask_artifact_path TEXT,
            skeleton_json TEXT,
            calibration_revision TEXT,
            source_stamp_json TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """


def _create_object_identity_reference_table_sql() -> str:
    return f"""
        CREATE TABLE {OBJECT_IDENTITY_REFERENCE_TABLE} (
            reference_id TEXT PRIMARY KEY,
            display_object_id TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('SHIGURE', 'HOLOLENS')),
            source_event_uid TEXT,
            source_task_id TEXT,
            image_path TEXT NOT NULL,
            mask_path TEXT,
            embedding_path TEXT,
            view_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            quality_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(display_object_id, view_hash)
        )
    """


def _create_identity_sync_job_table_sql() -> str:
    return f"""
        CREATE TABLE {IDENTITY_SYNC_JOB_TABLE} (
            sync_job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('STARTUP_RECOVERY', 'HOLOLENS_CAPTURE')),
            status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
            runtime_session_id TEXT,
            source_epoch_id TEXT,
            display_object_id TEXT,
            candidate_limit INTEGER NOT NULL DEFAULT 5,
            result_json TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
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


def _v2_table_builders() -> tuple[tuple[str, Any], ...]:
    """Return every application table that belongs to the strict v2 schema."""

    return (
        (TABLE_NAME, _create_task_table_sql),
        (ARUCO_REFERENCE_TABLE, _create_aruco_reference_table_sql),
        (STAGE_RUN_TABLE, _create_stage_run_table_sql),
        (TASK_TIMING_EVENT_TABLE, _create_task_timing_event_table_sql),
        (AI_MODEL_TIMING_TABLE, _create_ai_model_timing_table_sql),
        (ARUCO_MARKER_TABLE, _create_aruco_marker_table_sql),
        (ARUCO_MARKER_RELATION_TABLE, _create_aruco_marker_relation_table_sql),
        (MODEL_BOUNDS_TABLE, _create_model_bounds_table_sql),
        (DISPLAY_OBJECT_TABLE, _create_display_object_table_sql),
        (CAPTURE_INSTANCE_TABLE, _create_capture_instance_table_sql),
        (CAPTURE_BINDING_LOG_TABLE, _create_capture_binding_log_table_sql),
        (DISPLAY_OBJECT_STATE_TABLE, _create_display_object_state_table_sql),
        (
            DISPLAY_OBJECT_MODEL_REVISION_TABLE,
            _create_display_object_model_revision_table_sql,
        ),
        (
            DISPLAY_OBJECT_POSE_HISTORY_TABLE,
            _create_display_object_pose_history_table_sql,
        ),
        (SCHEMA_METADATA_TABLE, _create_schema_metadata_table_sql),
        (SHIGURE_RUNTIME_SESSION_TABLE, _create_shigure_runtime_session_table_sql),
        (SHIGURE_SOURCE_EPOCH_TABLE, _create_shigure_source_epoch_table_sql),
        (SHIGURE_OBJECT_BINDING_TABLE, _create_shigure_object_binding_table_sql),
        (SHIGURE_CANONICAL_EVENT_TABLE, _create_shigure_canonical_event_table_sql),
        (OBJECT_LIFECYCLE_EVENT_TABLE, _create_object_lifecycle_event_table_sql),
        (
            OBJECT_IDENTITY_REFERENCE_TABLE,
            _create_object_identity_reference_table_sql,
        ),
        (IDENTITY_SYNC_JOB_TABLE, _create_identity_sync_job_table_sql),
    )


def _v2_index_sql() -> tuple[str, ...]:
    return (
        f"""
        CREATE INDEX IF NOT EXISTS idx_{ARUCO_REFERENCE_TABLE}_startup_session
        ON {ARUCO_REFERENCE_TABLE} (startup_session_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{STAGE_RUN_TABLE}_task
        ON {STAGE_RUN_TABLE} (task_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TASK_TIMING_EVENT_TABLE}_task_stage
        ON {TASK_TIMING_EVENT_TABLE} (task_id, stage_name, event_name)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{AI_MODEL_TIMING_TABLE}_task
        ON {AI_MODEL_TIMING_TABLE} (task_id, service_name, timing_kind)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{AI_MODEL_TIMING_TABLE}_service
        ON {AI_MODEL_TIMING_TABLE} (service_name, timing_kind, started_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{ARUCO_MARKER_RELATION_TABLE}_marker
        ON {ARUCO_MARKER_RELATION_TABLE} (marker_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{MODEL_BOUNDS_TABLE}_status_uploaded
        ON {MODEL_BOUNDS_TABLE} (status, uploaded_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{MODEL_BOUNDS_TABLE}_task
        ON {MODEL_BOUNDS_TABLE} (task_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CAPTURE_INSTANCE_TABLE}_display_object
        ON {CAPTURE_INSTANCE_TABLE} (display_object_id, binding_status, updated_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CAPTURE_INSTANCE_TABLE}_task
        ON {CAPTURE_INSTANCE_TABLE} (task_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CAPTURE_BINDING_LOG_TABLE}_capture
        ON {CAPTURE_BINDING_LOG_TABLE} (capture_instance_id, created_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{CAPTURE_BINDING_LOG_TABLE}_task
        ON {CAPTURE_BINDING_LOG_TABLE} (task_id, created_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_MODEL_REVISION_TABLE}_display_revision
        ON {DISPLAY_OBJECT_MODEL_REVISION_TABLE} (display_object_id, model_revision DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_POSE_HISTORY_TABLE}_display_revision
        ON {DISPLAY_OBJECT_POSE_HISTORY_TABLE} (display_object_id, pose_revision DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SHIGURE_RUNTIME_SESSION_TABLE}_status
        ON {SHIGURE_RUNTIME_SESSION_TABLE} (status, started_at DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SHIGURE_SOURCE_EPOCH_TABLE}_runtime_status
        ON {SHIGURE_SOURCE_EPOCH_TABLE} (runtime_session_id, status, generation DESC)
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_raw
        ON {SHIGURE_OBJECT_BINDING_TABLE} (source_epoch_id, raw_shigure_object_id)
        WHERE status = 'ACTIVE'
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_display
        ON {SHIGURE_OBJECT_BINDING_TABLE} (source_epoch_id, display_object_id)
        WHERE status = 'ACTIVE'
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SHIGURE_CANONICAL_EVENT_TABLE}_display_created
        ON {SHIGURE_CANONICAL_EVENT_TABLE} (display_object_id, created_at DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{OBJECT_LIFECYCLE_EVENT_TABLE}_display_history
        ON {OBJECT_LIFECYCLE_EVENT_TABLE} (display_object_id, id DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{OBJECT_IDENTITY_REFERENCE_TABLE}_display_active
        ON {OBJECT_IDENTITY_REFERENCE_TABLE} (display_object_id, active, created_at DESC)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{IDENTITY_SYNC_JOB_TABLE}_status_created
        ON {IDENTITY_SYNC_JOB_TABLE} (status, created_at)
        """,
    )


def _application_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _schema_object_sql(
    conn: sqlite3.Connection,
    object_type: str,
    object_name: str,
) -> Optional[str]:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
        (object_type, object_name),
    ).fetchone()
    return str(row["sql"]) if row is not None and row["sql"] is not None else None


def _normalized_schema_sql(sql: str) -> str:
    normalized = " ".join(sql.strip().split())
    # sqlite_master omits this creation-time guard from stored index SQL.
    return normalized.replace(" INDEX IF NOT EXISTS ", " INDEX ")


def _migration_required(detail: str) -> RuntimeError:
    return RuntimeError(
        "Task database is not the strict Shigure v2 schema: "
        f"{detail}. Run: python code/migrate_shigure_v2_data.py --apply"
    )


def _validate_v2_schema(conn: sqlite3.Connection) -> None:
    builders = dict(_v2_table_builders())
    expected_tables = set(builders)
    actual_tables = _application_table_names(conn)
    missing = sorted(expected_tables - actual_tables)
    unexpected = sorted(actual_tables - expected_tables)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing tables=" + ",".join(missing))
        if unexpected:
            details.append("legacy/unknown tables=" + ",".join(unexpected))
        raise _migration_required("; ".join(details))

    mismatched = []
    for table_name, builder in builders.items():
        actual_sql = _table_sql(conn, table_name)
        expected_sql = builder()
        if actual_sql is None or _normalized_table_definition(
            actual_sql
        ) != _normalized_table_definition(expected_sql):
            mismatched.append(table_name)
    if mismatched:
        raise _migration_required("table DDL mismatch=" + ",".join(sorted(mismatched)))

    for expected_sql in _v2_index_sql():
        tokens = _normalized_schema_sql(expected_sql).split()
        index_token = tokens.index("INDEX") + 1
        if tokens[index_token : index_token + 3] == ["IF", "NOT", "EXISTS"]:
            index_token += 3
        index_name = tokens[index_token]
        actual_sql = _schema_object_sql(conn, "index", index_name)
        if actual_sql is None or _normalized_schema_sql(
            actual_sql
        ) != _normalized_schema_sql(expected_sql):
            raise _migration_required(f"index DDL mismatch={index_name}")

    metadata = conn.execute(
        f"""
        SELECT schema_version, migrated_at, detail_json
        FROM {SCHEMA_METADATA_TABLE}
        WHERE schema_name = 'shigure_runtime'
        """
    ).fetchone()
    found_version = None if metadata is None else metadata["schema_version"]
    try:
        version_matches = int(found_version) == SHIGURE_SCHEMA_VERSION
    except (TypeError, ValueError):
        version_matches = False
    if not version_matches:
        raise _migration_required(
            f"schema_metadata shigure_runtime version must be {SHIGURE_SCHEMA_VERSION}, "
            f"got {found_version}"
        )
    if not str(metadata["migrated_at"] or "").strip():
        raise _migration_required("schema_metadata.migrated_at is empty")
    try:
        detail = json.loads(str(metadata["detail_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise _migration_required("schema_metadata.detail_json is invalid JSON") from exc
    if not isinstance(detail, dict):
        raise _migration_required("schema_metadata.detail_json must be an object")


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


_DISPLAY_STATE_V2_COPY_COLUMNS = (
    "display_object_id",
    "active_model_revision",
    "active_model_task_id",
    "active_model_asset_hash",
    "latest_hololens_pose_revision",
    "latest_hololens_pose_aruco_json",
    "latest_hololens_task_id",
    "latest_hololens_captured_at",
    "latest_tracking_pose_revision",
    "latest_tracking_model_revision",
    "latest_tracking_pose_aruco_json",
    "latest_tracking_observation_seq",
    "created_at",
    "updated_at",
)
_POSE_HISTORY_V2_COPY_COLUMNS = (
    "id",
    "display_object_id",
    "task_id",
    "model_revision",
    "pose_revision",
    "pose_aruco_json",
    "captured_at",
    "created_at",
    "updated_at",
)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _rebuild_display_state_table_v2(conn: sqlite3.Connection) -> None:
    old_columns = _table_columns(conn, DISPLAY_OBJECT_STATE_TABLE)
    missing = set(_DISPLAY_STATE_V2_COPY_COLUMNS) - old_columns
    if missing:
        raise RuntimeError(
            "Cannot migrate display object state; missing columns: "
            + ", ".join(sorted(missing))
        )
    temporary = "display_object_states__v2"
    conn.execute(f"DROP TABLE IF EXISTS {temporary}")
    conn.execute(_create_display_object_state_table_sql(temporary))
    columns = ", ".join(_DISPLAY_STATE_V2_COPY_COLUMNS)
    conn.execute(
        f"INSERT INTO {temporary} ({columns}) "
        f"SELECT {columns} FROM {DISPLAY_OBJECT_STATE_TABLE}"
    )
    conn.execute(f"DROP TABLE {DISPLAY_OBJECT_STATE_TABLE}")
    conn.execute(f"ALTER TABLE {temporary} RENAME TO {DISPLAY_OBJECT_STATE_TABLE}")


def _rebuild_pose_history_table_v2(conn: sqlite3.Connection) -> None:
    old_columns = _table_columns(conn, DISPLAY_OBJECT_POSE_HISTORY_TABLE)
    missing = set(_POSE_HISTORY_V2_COPY_COLUMNS) - old_columns
    if missing:
        raise RuntimeError(
            "Cannot migrate display object pose history; missing columns: "
            + ", ".join(sorted(missing))
        )
    temporary = "display_object_pose_history__v2"
    conn.execute(f"DROP TABLE IF EXISTS {temporary}")
    conn.execute(_create_display_object_pose_history_table_sql(temporary))
    columns = ", ".join(_POSE_HISTORY_V2_COPY_COLUMNS)
    conn.execute(
        f"INSERT INTO {temporary} ({columns}) "
        f"SELECT {columns} FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE}"
    )
    conn.execute(f"DROP TABLE {DISPLAY_OBJECT_POSE_HISTORY_TABLE}")
    conn.execute(f"ALTER TABLE {temporary} RENAME TO {DISPLAY_OBJECT_POSE_HISTORY_TABLE}")


def _ensure_display_tables_v2(conn: sqlite3.Connection) -> None:
    state_sql = _table_sql(conn, DISPLAY_OBJECT_STATE_TABLE)
    if state_sql is None:
        conn.execute(_create_display_object_state_table_sql())
    elif _normalized_table_definition(state_sql) != _normalized_table_definition(
        _create_display_object_state_table_sql()
    ):
        _rebuild_display_state_table_v2(conn)

    history_sql = _table_sql(conn, DISPLAY_OBJECT_POSE_HISTORY_TABLE)
    if history_sql is None:
        conn.execute(_create_display_object_pose_history_table_sql())
    elif _normalized_table_definition(history_sql) != _normalized_table_definition(
        _create_display_object_pose_history_table_sql()
    ):
        _rebuild_pose_history_table_v2(conn)


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


def migrate_legacy_task_database_to_v2_once() -> None:
    """Explicitly rebuild a legacy database into the strict v2 schema.

    Runtime code must never call this function.  Destructive legacy cleanup is
    intentionally reachable only from the offline migration command.
    """

    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")

        try:
            _validate_v2_schema(conn)
        except RuntimeError:
            pass
        else:
            conn.commit()
            _SCHEMA_INITIALIZED = True
            return

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

        # V2 is a clean protocol cut. Old event/body rows cannot prove a
        # Shigure runtime identity, so they are intentionally not adapted.
        conn.execute("DROP TABLE IF EXISTS realtime_tracking_events")
        conn.execute("DROP TABLE IF EXISTS auxiliary_jobs")

        _ensure_display_tables_v2(conn)
        if _table_sql(conn, DISPLAY_OBJECT_MODEL_REVISION_TABLE) is None:
            conn.execute(_create_display_object_model_revision_table_sql())
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_MODEL_REVISION_TABLE}_display_revision
            ON {DISPLAY_OBJECT_MODEL_REVISION_TABLE} (display_object_id, model_revision DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_POSE_HISTORY_TABLE}_display_revision
            ON {DISPLAY_OBJECT_POSE_HISTORY_TABLE} (display_object_id, pose_revision DESC)
            """
        )

        table_builders = (
            (SCHEMA_METADATA_TABLE, _create_schema_metadata_table_sql),
            (SHIGURE_RUNTIME_SESSION_TABLE, _create_shigure_runtime_session_table_sql),
            (SHIGURE_SOURCE_EPOCH_TABLE, _create_shigure_source_epoch_table_sql),
            (SHIGURE_OBJECT_BINDING_TABLE, _create_shigure_object_binding_table_sql),
            (SHIGURE_CANONICAL_EVENT_TABLE, _create_shigure_canonical_event_table_sql),
            (OBJECT_LIFECYCLE_EVENT_TABLE, _create_object_lifecycle_event_table_sql),
            (OBJECT_IDENTITY_REFERENCE_TABLE, _create_object_identity_reference_table_sql),
            (IDENTITY_SYNC_JOB_TABLE, _create_identity_sync_job_table_sql),
        )
        for table_name, builder in table_builders:
            if _table_sql(conn, table_name) is None:
                conn.execute(builder())

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{SHIGURE_RUNTIME_SESSION_TABLE}_status
            ON {SHIGURE_RUNTIME_SESSION_TABLE} (status, started_at DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{SHIGURE_SOURCE_EPOCH_TABLE}_runtime_status
            ON {SHIGURE_SOURCE_EPOCH_TABLE} (runtime_session_id, status, generation DESC)
            """
        )
        conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_raw
            ON {SHIGURE_OBJECT_BINDING_TABLE} (source_epoch_id, raw_shigure_object_id)
            WHERE status = 'ACTIVE'
            """
        )
        conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_display
            ON {SHIGURE_OBJECT_BINDING_TABLE} (source_epoch_id, display_object_id)
            WHERE status = 'ACTIVE'
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{SHIGURE_CANONICAL_EVENT_TABLE}_display_created
            ON {SHIGURE_CANONICAL_EVENT_TABLE} (display_object_id, created_at DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{OBJECT_LIFECYCLE_EVENT_TABLE}_display_history
            ON {OBJECT_LIFECYCLE_EVENT_TABLE} (display_object_id, id DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{OBJECT_IDENTITY_REFERENCE_TABLE}_display_active
            ON {OBJECT_IDENTITY_REFERENCE_TABLE} (display_object_id, active, created_at DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{IDENTITY_SYNC_JOB_TABLE}_status_created
            ON {IDENTITY_SYNC_JOB_TABLE} (status, created_at)
            """
        )

        conn.execute(
            f"""
            INSERT INTO {SCHEMA_METADATA_TABLE} (
                schema_name, schema_version, migrated_at, detail_json
            ) VALUES ('shigure_runtime', ?, ?, ?)
            ON CONFLICT(schema_name) DO UPDATE SET
                schema_version = excluded.schema_version,
                migrated_at = excluded.migrated_at,
                detail_json = excluded.detail_json
            """,
            (
                SHIGURE_SCHEMA_VERSION,
                _utc_now_text(),
                json.dumps(
                    {
                        "legacy_realtime_events": "deleted_unmigratable",
                        "legacy_body_jobs": "deleted_retired",
                        "hololens_capture_history": "preserved",
                    },
                    ensure_ascii=False,
                ),
            ),
        )

        # Some legacy databases contain canonical tables but lack indexes
        # because older startup code created an index only with its table.
        for index_sql in _v2_index_sql():
            conn.execute(index_sql)

        _validate_v2_schema(conn)
        conn.commit()
        _SCHEMA_INITIALIZED = True


def initialize_task_table() -> None:
    """Create a brand-new v2 database or validate an existing one read-only.

    An existing database is never repaired, rebuilt, dropped, or version-
    upserted here.  Operators must run the explicit one-shot migration command
    when this strict boundary rejects a legacy or divergent schema.
    """

    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    with _get_connection() as conn:
        existing_tables = _application_table_names(conn)
        if not existing_tables:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            for _, builder in _v2_table_builders():
                conn.execute(builder())
            for index_sql in _v2_index_sql():
                conn.execute(index_sql)
            conn.execute(
                f"""
                INSERT INTO {SCHEMA_METADATA_TABLE} (
                    schema_name, schema_version, migrated_at, detail_json
                ) VALUES ('shigure_runtime', ?, ?, ?)
                """,
                (
                    SHIGURE_SCHEMA_VERSION,
                    _utc_now_text(),
                    json.dumps({"installation": "fresh_v2"}, ensure_ascii=False),
                ),
            )
            _validate_v2_schema(conn)
            conn.commit()
        else:
            _validate_v2_schema(conn)

        if ARUCO_SYNC_MARKER_REGISTRY_ON_START:
            conn.execute("BEGIN IMMEDIATE")
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
                pose_aruco_json, captured_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
            SELECT state.*
            FROM {DISPLAY_OBJECT_STATE_TABLE} AS state
            WHERE state.active_model_revision > 0
              AND EXISTS (
                  SELECT 1 FROM {OBJECT_IDENTITY_REFERENCE_TABLE} AS reference
                  WHERE reference.display_object_id = state.display_object_id
                    AND reference.active = 1
              )
            ORDER BY state.updated_at DESC, state.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_live_display_object_states(*, limit: int) -> List[Dict[str, Any]]:
    """Return lifecycle-known objects ordered only by Shigure activity."""

    initialize_task_table()
    limit = max(1, min(int(limit), 50))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {DISPLAY_OBJECT_STATE_TABLE}
            WHERE presence IN ('PRESENT', 'ABSENT')
              AND (
                    latest_tracking_pose_aruco_json IS NOT NULL
                    OR latest_hololens_pose_aruco_json IS NOT NULL
                  )
            ORDER BY presence_epoch DESC,
                     MAX(latest_tracking_observation_seq,
                         latest_spatial_observation_seq,
                         latest_skeleton_observation_seq) DESC,
                     updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_live_spatial_box_states() -> List[Dict[str, Any]]:
    """Return the complete Shigure box registry without a display limit."""

    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {DISPLAY_OBJECT_STATE_TABLE}
            WHERE active_model_revision > 0
              AND presence IN ('PRESENT', 'ABSENT')
            ORDER BY presence_epoch DESC,
                     latest_spatial_observation_seq DESC,
                     updated_at DESC,
                     display_object_id ASC
            """
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


def start_shigure_runtime_session(
    *, server_boot_id: str | None = None, config: Any = None
) -> Dict[str, Any]:
    """Start a server-local identity session and invalidate every older raw ID."""

    initialize_task_table()
    now = _utc_now_text()
    runtime_session_id = uuid.uuid4().hex
    resolved_boot_id = str(server_boot_id or uuid.uuid4().hex).strip()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        stale_sync_jobs = conn.execute(
            f"""
            SELECT sync_job_id, result_json
            FROM {IDENTITY_SYNC_JOB_TABLE}
            WHERE status IN ('PENDING', 'RUNNING')
            """
        ).fetchall()
        for stale_job in stale_sync_jobs:
            try:
                stale_result = (
                    json.loads(stale_job["result_json"])
                    if stale_job["result_json"]
                    else {}
                )
            except (json.JSONDecodeError, TypeError):
                stale_result = {}
            if not isinstance(stale_result, dict):
                stale_result = {}
            stale_result.update(
                {
                    "status": "FAILED",
                    "reason": "server_restart",
                    "updated_utc": now,
                }
            )
            conn.execute(
                f"""
                UPDATE {IDENTITY_SYNC_JOB_TABLE}
                SET status = 'FAILED',
                    result_json = ?,
                    error_message = 'server_restart',
                    completed_at = ?,
                    updated_at = ?
                WHERE sync_job_id = ?
                  AND status IN ('PENDING', 'RUNNING')
                """,
                (
                    json.dumps(stale_result, ensure_ascii=False),
                    now,
                    now,
                    str(stale_job["sync_job_id"]),
                ),
            )
        conn.execute(
            f"""
            UPDATE {SHIGURE_RUNTIME_SESSION_TABLE}
            SET status = 'CLOSED', ended_at = ?, close_reason = 'server_restart'
            WHERE status = 'ACTIVE'
            """,
            (now,),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_SOURCE_EPOCH_TABLE}
            SET status = 'CLOSED', closed_at = ?, close_reason = 'server_restart'
            WHERE status = 'ACTIVE'
            """,
            (now,),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
            SET status = 'REVOKED', valid_until = ?, revoke_reason = 'server_restart'
            WHERE status = 'ACTIVE'
            """,
            (now,),
        )
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = 'UNKNOWN',
                presence_epoch = presence_epoch + 1,
                active_shigure_binding_id = NULL,
                latest_tracking_model_revision = 0,
                latest_tracking_observation_seq = 0,
                latest_spatial_observation_seq = 0,
                latest_skeleton_observation_seq = 0,
                latest_spatial_box_aruco_json = NULL,
                latest_skeleton_json = NULL,
                updated_at = ?
            WHERE active_shigure_binding_id IS NOT NULL
            """,
            (now,),
        )
        conn.execute(
            f"""
            INSERT INTO {SHIGURE_RUNTIME_SESSION_TABLE} (
                runtime_session_id, server_boot_id, status, started_at, config_json
            ) VALUES (?, ?, 'ACTIVE', ?, ?)
            """,
            (
                runtime_session_id,
                resolved_boot_id,
                now,
                json.dumps(config if config is not None else {}, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_RUNTIME_SESSION_TABLE} WHERE runtime_session_id = ?",
            (runtime_session_id,),
        ).fetchone()
    return dict(row)


def close_shigure_runtime_session(runtime_session_id: str, *, reason: str) -> None:
    initialize_task_table()
    now = _utc_now_text()
    runtime_session_id = str(runtime_session_id or "").strip()
    if not runtime_session_id or not str(reason or "").strip():
        raise ValueError("runtime_session_id and reason are required")
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = 'UNKNOWN',
                presence_epoch = presence_epoch + 1,
                active_shigure_binding_id = NULL,
                latest_tracking_model_revision = 0,
                latest_tracking_observation_seq = 0,
                latest_spatial_observation_seq = 0,
                latest_skeleton_observation_seq = 0,
                latest_spatial_box_aruco_json = NULL,
                latest_skeleton_json = NULL,
                updated_at = ?
            WHERE active_shigure_binding_id IN (
                SELECT binding_id
                FROM {SHIGURE_OBJECT_BINDING_TABLE}
                WHERE runtime_session_id = ? AND status = 'ACTIVE'
            )
            """,
            (now, runtime_session_id),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
            SET status = 'REVOKED', valid_until = ?, revoke_reason = ?
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (now, reason, runtime_session_id),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_SOURCE_EPOCH_TABLE}
            SET status = 'CLOSED', closed_at = ?, close_reason = ?
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (now, reason, runtime_session_id),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_RUNTIME_SESSION_TABLE}
            SET status = 'CLOSED', ended_at = ?, close_reason = ?
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (now, reason, runtime_session_id),
        )


def open_shigure_source_epoch(
    *,
    runtime_session_id: str,
    reason: str,
    publisher_fingerprint: Any = None,
    raw_id_prefix: str | None = None,
) -> Dict[str, Any]:
    """Open one short-lived raw-ID namespace and revoke its predecessor."""

    initialize_task_table()
    runtime_session_id = str(runtime_session_id or "").strip()
    reason = str(reason or "").strip()
    if not runtime_session_id or not reason:
        raise ValueError("runtime_session_id and reason are required")
    now = _utc_now_text()
    source_epoch_id = uuid.uuid4().hex
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        session = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_RUNTIME_SESSION_TABLE}
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (runtime_session_id,),
        ).fetchone()
        if session is None:
            raise ValueError("runtime session is not active")
        previous = conn.execute(
            f"""
            SELECT COALESCE(MAX(generation), 0) AS generation
            FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE runtime_session_id = ?
            """,
            (runtime_session_id,),
        ).fetchone()
        generation = int(previous["generation"] or 0) + 1
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = 'UNKNOWN',
                presence_epoch = presence_epoch + 1,
                active_shigure_binding_id = NULL,
                latest_tracking_model_revision = 0,
                latest_tracking_observation_seq = 0,
                latest_spatial_observation_seq = 0,
                latest_skeleton_observation_seq = 0,
                latest_spatial_box_aruco_json = NULL,
                latest_skeleton_json = NULL,
                updated_at = ?
            WHERE active_shigure_binding_id IN (
                SELECT binding_id
                FROM {SHIGURE_OBJECT_BINDING_TABLE}
                WHERE runtime_session_id = ? AND status = 'ACTIVE'
            )
            """,
            (now, runtime_session_id),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
            SET status = 'REVOKED', valid_until = ?, revoke_reason = 'source_epoch_changed'
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (now, runtime_session_id),
        )
        conn.execute(
            f"""
            UPDATE {SHIGURE_SOURCE_EPOCH_TABLE}
            SET status = 'CLOSED', closed_at = ?, close_reason = 'superseded'
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (now, runtime_session_id),
        )
        conn.execute(
            f"""
            INSERT INTO {SHIGURE_SOURCE_EPOCH_TABLE} (
                source_epoch_id, runtime_session_id, generation, status,
                opened_at, open_reason, publisher_fingerprint_json, raw_id_prefix
            ) VALUES (?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
            """,
            (
                source_epoch_id,
                runtime_session_id,
                generation,
                now,
                reason,
                json.dumps(
                    publisher_fingerprint if publisher_fingerprint is not None else {},
                    ensure_ascii=False,
                ),
                str(raw_id_prefix).strip() if raw_id_prefix else None,
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_SOURCE_EPOCH_TABLE} WHERE source_epoch_id = ?",
            (source_epoch_id,),
        ).fetchone()
    return dict(row)


def get_active_shigure_source_epoch(runtime_session_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            ORDER BY generation DESC
            LIMIT 1
            """,
            (str(runtime_session_id),),
        ).fetchone()
    return _row_to_dict(row)


def establish_shigure_binding(
    *,
    runtime_session_id: str,
    source_epoch_id: str,
    raw_shigure_object_id: str,
    display_object_id: str,
    established_by: str,
    established_event_uid: str | None = None,
    confidence: float | None = None,
    detail: Any = None,
) -> Dict[str, Any]:
    """Create an epoch-scoped one-to-one binding; ambiguity fails closed."""

    initialize_task_table()
    runtime_session_id = str(runtime_session_id or "").strip()
    source_epoch_id = str(source_epoch_id or "").strip()
    raw_id = str(raw_shigure_object_id or "").strip()
    display_object_id = str(display_object_id or "").strip()
    established_by = str(established_by or "").strip()
    if not all((runtime_session_id, source_epoch_id, raw_id, display_object_id, established_by)):
        raise ValueError("all binding identity fields are required")
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        epoch = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE source_epoch_id = ? AND runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (source_epoch_id, runtime_session_id),
        ).fetchone()
        if epoch is None:
            raise ValueError("source epoch is not active")
        if conn.execute(
            f"SELECT 1 FROM {DISPLAY_OBJECT_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone() is None:
            raise ValueError(f"unknown display object: {display_object_id}")

        raw_binding = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE source_epoch_id = ? AND raw_shigure_object_id = ? AND status = 'ACTIVE'
            """,
            (source_epoch_id, raw_id),
        ).fetchone()
        display_binding = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE source_epoch_id = ? AND display_object_id = ? AND status = 'ACTIVE'
            """,
            (source_epoch_id, display_object_id),
        ).fetchone()
        if raw_binding is not None or display_binding is not None:
            if (
                raw_binding is not None
                and display_binding is not None
                and raw_binding["binding_id"] == display_binding["binding_id"]
            ):
                return dict(raw_binding)
            raise ValueError("binding conflicts with the active one-to-one assignment")

        previous = conn.execute(
            f"""
            SELECT COALESCE(MAX(binding_epoch), 0) AS binding_epoch
            FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE source_epoch_id = ? AND raw_shigure_object_id = ?
            """,
            (source_epoch_id, raw_id),
        ).fetchone()
        binding_epoch = int(previous["binding_epoch"] or 0) + 1
        binding_id = uuid.uuid4().hex
        conn.execute(
            f"""
            INSERT INTO {SHIGURE_OBJECT_BINDING_TABLE} (
                binding_id, runtime_session_id, source_epoch_id,
                raw_shigure_object_id, display_object_id, binding_epoch,
                status, established_by, established_event_uid, valid_from,
                confidence, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?)
            """,
            (
                binding_id,
                runtime_session_id,
                source_epoch_id,
                raw_id,
                display_object_id,
                binding_epoch,
                established_by,
                established_event_uid,
                now,
                float(confidence) if confidence is not None else None,
                json.dumps(detail if detail is not None else {}, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE} WHERE binding_id = ?",
            (binding_id,),
        ).fetchone()
    return dict(row)


def get_active_shigure_binding(
    *, source_epoch_id: str, raw_shigure_object_id: str
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE source_epoch_id = ? AND raw_shigure_object_id = ? AND status = 'ACTIVE'
            """,
            (str(source_epoch_id), str(raw_shigure_object_id)),
        ).fetchone()
    return _row_to_dict(row)


def list_active_shigure_bindings(source_epoch_id: str) -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE source_epoch_id = ? AND status = 'ACTIVE'
            ORDER BY valid_from ASC
            """,
            (str(source_epoch_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def revoke_shigure_binding(binding_id: str, *, reason: str) -> None:
    initialize_task_table()
    if not str(reason or "").strip():
        raise ValueError("reason is required")
    with _get_connection() as conn:
        conn.execute(
            f"""
            UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
            SET status = 'REVOKED', valid_until = ?, revoke_reason = ?
            WHERE binding_id = ? AND status = 'ACTIVE'
            """,
            (_utc_now_text(), str(reason), str(binding_id)),
        )


def _canonical_shigure_action(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    actions = {
        "bring_in": "BRING_IN",
        "bringin": "BRING_IN",
        "take_out": "TAKE_OUT",
        "takeout": "TAKE_OUT",
        "move": "MOVE",
        "obj_move": "MOVE",
    }
    try:
        return actions[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Shigure action: {value}") from exc


def record_shigure_canonical_event(
    *,
    runtime_session_id: str,
    source_epoch_id: str,
    stamp_sec: int,
    stamp_nanosec: int,
    frame_id: str,
    detection_index: int,
    action: str,
    bbox: Any,
    resolution_status: str,
    raw_shigure_object_id: str | None = None,
    binding_id: str | None = None,
    display_object_id: str | None = None,
    resolution_method: str | None = None,
    collider: Any = None,
    mask_artifact_path: str | Path | None = None,
    scene_image_path: str | Path | None = None,
    object_crop_path: str | Path | None = None,
    skeleton: Any = None,
    source_stamp: Any = None,
    detail: Any = None,
) -> Dict[str, Any]:
    """Persist one immutable exact-stamp canonical event slot and its resolution."""

    initialize_task_table()
    status = str(resolution_status or "").strip().upper()
    if status not in {"RESOLVED", "UNRESOLVED", "AMBIGUOUS", "CONFLICT", "REJECTED"}:
        raise ValueError("invalid resolution_status")
    if not isinstance(bbox, dict):
        raise ValueError("bbox must be an object")
    action_value = _canonical_shigure_action(action)
    frame_value = str(frame_id or "")
    stamp_sec = int(stamp_sec)
    stamp_nanosec = int(stamp_nanosec)
    detection_index = int(detection_index)
    source_stamp_value = source_stamp or {
        "sec": stamp_sec,
        "nanosec": stamp_nanosec,
        "frame_id": frame_value,
    }
    key = (
        f"{runtime_session_id}|{source_epoch_id}|{stamp_sec}|"
        f"{stamp_nanosec}|{frame_value}|{detection_index}"
    )
    event_uid = uuid.uuid5(uuid.NAMESPACE_URL, f"shigure-detection:{key}").hex
    if status == "RESOLVED" and not all(
        (raw_shigure_object_id, binding_id, display_object_id, resolution_method)
    ):
        raise ValueError("resolved events require raw ID, binding, display object and method")

    def stored_path(value: str | Path | None) -> str | None:
        return normalize_path_for_storage(value) if value is not None else None

    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {SHIGURE_CANONICAL_EVENT_TABLE} (
                event_uid, runtime_session_id, source_epoch_id,
                stamp_sec, stamp_nanosec, frame_id, detection_index, action,
                raw_shigure_object_id, binding_id, display_object_id,
                resolution_status, resolution_method, bbox_json, collider_json,
                mask_artifact_path, scene_image_path, object_crop_path,
                skeleton_json, source_stamp_json, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_uid) DO UPDATE SET
                raw_shigure_object_id = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.raw_shigure_object_id
                    ELSE excluded.raw_shigure_object_id
                END,
                binding_id = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.binding_id
                    ELSE excluded.binding_id
                END,
                display_object_id = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.display_object_id
                    ELSE excluded.display_object_id
                END,
                resolution_status = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                    ELSE excluded.resolution_status
                END,
                resolution_method = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_method
                    ELSE excluded.resolution_method
                END,
                collider_json = COALESCE(
                    excluded.collider_json,
                    {SHIGURE_CANONICAL_EVENT_TABLE}.collider_json
                ),
                mask_artifact_path = COALESCE(
                    excluded.mask_artifact_path,
                    {SHIGURE_CANONICAL_EVENT_TABLE}.mask_artifact_path
                ),
                scene_image_path = COALESCE(
                    excluded.scene_image_path,
                    {SHIGURE_CANONICAL_EVENT_TABLE}.scene_image_path
                ),
                object_crop_path = COALESCE(
                    excluded.object_crop_path,
                    {SHIGURE_CANONICAL_EVENT_TABLE}.object_crop_path
                ),
                skeleton_json = COALESCE(
                    excluded.skeleton_json,
                    {SHIGURE_CANONICAL_EVENT_TABLE}.skeleton_json
                ),
                detail_json = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status = 'RESOLVED'
                         AND excluded.resolution_status != 'RESOLVED'
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.detail_json
                    ELSE excluded.detail_json
                END
            """,
            (
                event_uid,
                str(runtime_session_id),
                str(source_epoch_id),
                stamp_sec,
                stamp_nanosec,
                frame_value,
                detection_index,
                action_value,
                str(raw_shigure_object_id) if raw_shigure_object_id else None,
                str(binding_id) if binding_id else None,
                str(display_object_id) if display_object_id else None,
                status,
                str(resolution_method) if resolution_method else None,
                json.dumps(bbox, ensure_ascii=False),
                _dump_optional_json(collider),
                stored_path(mask_artifact_path),
                stored_path(scene_image_path),
                stored_path(object_crop_path),
                _dump_optional_json(skeleton),
                json.dumps(source_stamp_value, ensure_ascii=False),
                json.dumps(detail if detail is not None else {}, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid,),
        ).fetchone()
    return dict(row)


def get_shigure_canonical_event(event_uid: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (str(event_uid),),
        ).fetchone()
    return _row_to_dict(row)


def activate_recovered_shigure_binding(binding_id: str) -> Dict[str, Any]:
    """Mark a DINO-recovered static object present without fabricating an event."""

    initialize_task_table()
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        binding = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE binding_id = ? AND status = 'ACTIVE'
            """,
            (str(binding_id),),
        ).fetchone()
        if binding is None:
            raise ValueError("binding is not active")
        display_object_id = str(binding["display_object_id"])
        conn.execute(
            f"""
            INSERT INTO {DISPLAY_OBJECT_STATE_TABLE} (display_object_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(display_object_id) DO NOTHING
            """,
            (display_object_id, now),
        )
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = 'PRESENT',
                presence_epoch = presence_epoch + 1,
                active_shigure_binding_id = ?,
                latest_tracking_model_revision = 0,
                latest_tracking_pose_aruco_json = NULL,
                latest_tracking_observation_seq = 0,
                latest_spatial_box_aruco_json = NULL,
                latest_spatial_observation_seq = 0,
                latest_skeleton_json = NULL,
                latest_skeleton_observation_seq = 0,
                updated_at = ?
            WHERE display_object_id = ?
            """,
            (str(binding_id), now, display_object_id),
        )
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
    return dict(row)


def update_shigure_live_observation(
    *,
    binding_id: str,
    observation_seq: int,
    model_revision: int,
    pose_aruco: Any = None,
    spatial_box_corners_aruco: Any = None,
    spatial_box_observed: bool = False,
    skeleton: Any = None,
) -> Optional[Dict[str, Any]]:
    """Commit live compute while presentation may independently show history."""

    initialize_task_table()
    if pose_aruco is not None and not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco must be an object")
    if spatial_box_corners_aruco is not None and (
        not isinstance(spatial_box_corners_aruco, list)
        or len(spatial_box_corners_aruco) != 8
    ):
        raise ValueError("spatial_box_corners_aruco must contain exactly 8 points")
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        binding = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE binding_id = ? AND status = 'ACTIVE'
            """,
            (str(binding_id),),
        ).fetchone()
        if binding is None:
            return None
        display_object_id = str(binding["display_object_id"])
        state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
        if (
            state is None
            or str(state["presence"]) != "PRESENT"
            or str(state["active_shigure_binding_id"] or "") != str(binding_id)
            or int(state["active_model_revision"] or 0) != int(model_revision)
        ):
            return None
        accept_pose = pose_aruco is not None and int(state["latest_tracking_observation_seq"] or 0) < int(observation_seq)
        accept_box = bool(spatial_box_observed) and int(state["latest_spatial_observation_seq"] or 0) < int(observation_seq)
        accept_skeleton = skeleton is not None and int(state["latest_skeleton_observation_seq"] or 0) < int(observation_seq)
        if not any((accept_pose, accept_box, accept_skeleton)):
            return None
        pose_revision = int(state["latest_tracking_pose_revision"] or 0)
        if accept_pose:
            pose_revision += 1
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET latest_tracking_pose_revision = ?,
                latest_tracking_model_revision = ?,
                latest_tracking_pose_aruco_json = COALESCE(?, latest_tracking_pose_aruco_json),
                latest_tracking_observation_seq = ?,
                latest_spatial_box_aruco_json = CASE WHEN ? THEN ? ELSE latest_spatial_box_aruco_json END,
                latest_spatial_observation_seq = ?,
                latest_skeleton_json = COALESCE(?, latest_skeleton_json),
                latest_skeleton_observation_seq = ?,
                updated_at = ?
            WHERE display_object_id = ?
            """,
            (
                pose_revision,
                int(model_revision),
                _dump_optional_json(pose_aruco) if accept_pose else None,
                int(observation_seq) if accept_pose else int(state["latest_tracking_observation_seq"] or 0),
                1 if accept_box else 0,
                _dump_optional_json(spatial_box_corners_aruco) if accept_box else None,
                int(observation_seq) if accept_box else int(state["latest_spatial_observation_seq"] or 0),
                _dump_optional_json(skeleton) if accept_skeleton else None,
                int(observation_seq) if accept_skeleton else int(state["latest_skeleton_observation_seq"] or 0),
                now,
                display_object_id,
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
    result = dict(row)
    result["_pose_accepted"] = bool(accept_pose)
    result["_spatial_box_accepted"] = bool(accept_box)
    result["_skeleton_accepted"] = bool(accept_skeleton)
    return result


def apply_object_lifecycle_event(
    *,
    canonical_event_uid: str,
    pose_aruco: Any = None,
    spatial_box_corners_aruco: Any = None,
    skeleton: Any = None,
    calibration_revision: str | None = None,
    occurred_at: str | None = None,
) -> Dict[str, Any]:
    """Atomically authorize a resolved Shigure event and change presence."""

    initialize_task_table()
    if pose_aruco is not None and not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco must be an object")
    if spatial_box_corners_aruco is not None and (
        not isinstance(spatial_box_corners_aruco, list)
        or len(spatial_box_corners_aruco) != 8
    ):
        raise ValueError("spatial_box_corners_aruco must contain exactly 8 points")
    now = _utc_now_text()
    event_time = str(occurred_at or now)
    lifecycle_uid = uuid.uuid5(
        uuid.NAMESPACE_URL, f"shigure-lifecycle:{canonical_event_uid}"
    ).hex
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE}
            WHERE event_uid = ? AND resolution_status = 'RESOLVED'
            """,
            (str(canonical_event_uid),),
        ).fetchone()
        if event is None:
            raise ValueError("canonical event is not resolved")
        replay = conn.execute(
            f"""
            SELECT * FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
            WHERE canonical_event_uid = ?
            """,
            (str(canonical_event_uid),),
        ).fetchone()
        if replay is not None:
            # The compatibility adapter may emit the lifecycle event before
            # camera/person/contact inputs for the same exact ROS stamp arrive.
            # Replay supplements evidence without advancing presence or creating
            # a second history row. A late exact event skeleton supersedes the
            # state fallback captured by the first emission.
            supplemental_skeleton = (
                _dump_optional_json(skeleton) if skeleton is not None else None
            )
            conn.execute(
                f"""
                UPDATE {OBJECT_LIFECYCLE_EVENT_TABLE}
                SET pose_aruco_json = COALESCE(pose_aruco_json, ?),
                    spatial_box_corners_aruco_json = COALESCE(
                        spatial_box_corners_aruco_json, ?
                    ),
                    scene_image_path = COALESCE(scene_image_path, ?),
                    object_crop_path = COALESCE(object_crop_path, ?),
                    mask_artifact_path = COALESCE(mask_artifact_path, ?),
                    skeleton_json = COALESCE(?, skeleton_json),
                    calibration_revision = COALESCE(calibration_revision, ?)
                WHERE canonical_event_uid = ?
                """,
                (
                    _dump_optional_json(pose_aruco),
                    _dump_optional_json(spatial_box_corners_aruco),
                    event["scene_image_path"],
                    event["object_crop_path"],
                    event["mask_artifact_path"],
                    supplemental_skeleton,
                    str(calibration_revision) if calibration_revision else None,
                    str(canonical_event_uid),
                ),
            )
            updated = conn.execute(
                f"""
                SELECT * FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
                WHERE canonical_event_uid = ?
                """,
                (str(canonical_event_uid),),
            ).fetchone()
            result = dict(updated)
            result["_replayed"] = True
            return result

        binding = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE}
            WHERE binding_id = ? AND status = 'ACTIVE'
            """,
            (str(event["binding_id"]),),
        ).fetchone()
        if binding is None or any(
            (
                str(binding["source_epoch_id"]) != str(event["source_epoch_id"]),
                str(binding["raw_shigure_object_id"]) != str(event["raw_shigure_object_id"]),
                str(binding["display_object_id"]) != str(event["display_object_id"]),
            )
        ):
            raise ValueError("canonical event no longer matches the active binding")

        display_object_id = str(binding["display_object_id"])
        state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
        if state is None:
            raise ValueError("display object has no model/capture state")
        action = str(event["action"])
        before = str(state["presence"])
        if action == "BRING_IN":
            if before == "PRESENT":
                raise ValueError("duplicate bring-in for a present object")
            after = "PRESENT"
        elif action == "TAKE_OUT":
            if before != "PRESENT":
                raise ValueError("take-out requires a present object")
            after = "ABSENT"
        else:
            if before != "PRESENT":
                raise ValueError("move requires a present object")
            after = "PRESENT"

        resolved_pose = pose_aruco
        if resolved_pose is None:
            raw_pose = (
                state["latest_tracking_pose_aruco_json"]
                or state["latest_hololens_pose_aruco_json"]
            )
            resolved_pose = json.loads(raw_pose) if raw_pose else None
        resolved_skeleton = skeleton
        if resolved_skeleton is None and action != "BRING_IN":
            raw_skeleton = event["skeleton_json"] or state["latest_skeleton_json"]
            resolved_skeleton = json.loads(raw_skeleton) if raw_skeleton else None
        resolved_box = spatial_box_corners_aruco
        if resolved_box is None and action != "BRING_IN":
            raw_box = state["latest_spatial_box_aruco_json"]
            resolved_box = json.loads(raw_box) if raw_box else None
        presence_epoch = int(state["presence_epoch"] or 0) + 1
        pose_revision = int(
            state["latest_tracking_pose_revision"]
            or state["latest_hololens_pose_revision"]
            or 0
        )
        conn.execute(
            f"""
            INSERT INTO {OBJECT_LIFECYCLE_EVENT_TABLE} (
                lifecycle_event_uid, canonical_event_uid, display_object_id,
                binding_id, source_epoch_id, raw_shigure_object_id, action,
                presence_before, presence_after, presence_epoch, model_revision,
                pose_revision, pose_aruco_json, spatial_box_corners_aruco_json,
                scene_image_path, object_crop_path, mask_artifact_path,
                skeleton_json, calibration_revision, source_stamp_json, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lifecycle_uid,
                str(canonical_event_uid),
                display_object_id,
                str(binding["binding_id"]),
                str(binding["source_epoch_id"]),
                str(binding["raw_shigure_object_id"]),
                action,
                before,
                after,
                presence_epoch,
                int(state["active_model_revision"] or 0),
                pose_revision,
                _dump_optional_json(resolved_pose),
                _dump_optional_json(resolved_box),
                event["scene_image_path"],
                event["object_crop_path"],
                event["mask_artifact_path"],
                _dump_optional_json(resolved_skeleton),
                str(calibration_revision) if calibration_revision else None,
                str(event["source_stamp_json"]),
                event_time,
            ),
        )
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = ?,
                presence_epoch = ?,
                active_shigure_binding_id = ?,
                last_lifecycle_event_uid = ?,
                latest_tracking_model_revision = CASE
                    WHEN ? = 'BRING_IN' THEN 0
                    ELSE latest_tracking_model_revision
                END,
                latest_tracking_pose_aruco_json = CASE
                    WHEN ? = 'BRING_IN' THEN NULL
                    ELSE latest_tracking_pose_aruco_json
                END,
                latest_tracking_observation_seq = CASE
                    WHEN ? = 'BRING_IN' THEN 0
                    ELSE latest_tracking_observation_seq
                END,
                latest_spatial_box_aruco_json = CASE
                    WHEN ? IN ('BRING_IN', 'TAKE_OUT') THEN NULL
                    ELSE COALESCE(?, latest_spatial_box_aruco_json)
                END,
                latest_spatial_observation_seq = CASE
                    WHEN ? IN ('BRING_IN', 'TAKE_OUT') THEN 0
                    ELSE latest_spatial_observation_seq
                END,
                latest_skeleton_json = CASE
                    WHEN ? = 'BRING_IN' THEN ?
                    ELSE COALESCE(?, latest_skeleton_json)
                END,
                latest_skeleton_observation_seq = CASE
                    WHEN ? = 'BRING_IN' THEN 0
                    ELSE latest_skeleton_observation_seq
                END,
                updated_at = ?
            WHERE display_object_id = ?
            """,
            (
                after,
                presence_epoch,
                None if action == "TAKE_OUT" else str(binding["binding_id"]),
                lifecycle_uid,
                action,
                action,
                action,
                action,
                _dump_optional_json(resolved_box),
                action,
                action,
                _dump_optional_json(resolved_skeleton),
                _dump_optional_json(resolved_skeleton),
                action,
                now,
                display_object_id,
            ),
        )
        if action == "TAKE_OUT":
            # Shigure allocates a fresh raw ID for a later bring-in. Release
            # this epoch-local one-to-one slot immediately after the take-out
            # history row has captured the old binding and pre-take pose.
            conn.execute(
                f"""
                UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
                SET status = 'REVOKED',
                    valid_until = ?,
                    revoke_reason = 'take_out_completed'
                WHERE binding_id = ? AND status = 'ACTIVE'
                """,
                (now, str(binding["binding_id"])),
            )
        row = conn.execute(
            f"""
            SELECT * FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
            WHERE lifecycle_event_uid = ?
            """,
            (lifecycle_uid,),
        ).fetchone()
    result = dict(row)
    result["_replayed"] = False
    return result


def list_object_lifecycle_history(
    display_object_id: str,
    *,
    limit: int = 20,
    before_id: int | None = None,
    take_out_only: bool = True,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit), 100))
    where = ["display_object_id = ?"]
    values: list[Any] = [str(display_object_id)]
    if before_id is not None:
        where.append("id < ?")
        values.append(int(before_id))
    if take_out_only:
        where.append("action = 'TAKE_OUT'")
    values.append(limit)
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(values),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_shigure_canonical_event(
    display_object_id: str,
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE}
            WHERE display_object_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(display_object_id),),
        ).fetchone()
    return _row_to_dict(row)


def add_object_identity_reference(
    *,
    display_object_id: str,
    source: str,
    image_path: str | Path,
    view_hash: str,
    mask_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
    source_event_uid: str | None = None,
    source_task_id: str | None = None,
    quality: Any = None,
) -> Dict[str, Any]:
    initialize_task_table()
    source_value = str(source or "").strip().upper()
    if source_value not in {"SHIGURE", "HOLOLENS"}:
        raise ValueError("source must be SHIGURE or HOLOLENS")

    admission: Dict[str, Any] | None = None
    if source_value == "SHIGURE":
        if mask_path is None:
            raise ValueError("SHIGURE identity references require a mask")
        if not isinstance(quality, dict):
            raise ValueError("SHIGURE identity references require admission quality")
        admission = quality
        if (
            str(admission.get("admission_method") or "")
            != "DINOV2_STRICT_SHIGURE_EXAMPLE"
            or str(admission.get("admission_status") or "") != "MATCHED"
            or str(admission.get("admission_display_object_id") or "")
            != str(display_object_id)
            or not str(admission.get("admission_reference_id") or "").strip()
            or not str(admission.get("admission_source_epoch_id") or "").strip()
            or not str(admission.get("admission_binding_id") or "").strip()
            or not str(
                admission.get("admission_raw_shigure_object_id") or ""
            ).strip()
        ):
            raise ValueError("SHIGURE identity reference admission is invalid")
        try:
            admission_distance = float(admission["admission_distance"])
            admission_margin_value = admission.get("admission_margin")
            admission_margin = (
                None
                if admission_margin_value is None
                else float(admission_margin_value)
            )
            admission_competitor_count = int(
                admission["admission_competitor_count"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "SHIGURE identity reference admission scores are invalid"
            ) from exc
        if (
            not math.isfinite(admission_distance)
            or admission_distance < 0.0
            or admission_distance > SHIGURE_EXAMPLE_DINO_DISTANCE_THRESHOLD
            or admission_competitor_count < 0
            or (
                admission_competitor_count > 0
                and admission_margin is None
            )
            or (
                admission_margin is not None
                and (
                    not math.isfinite(admission_margin)
                    or admission_margin < SHIGURE_EXAMPLE_DINO_SECOND_MARGIN
                )
            )
        ):
            raise ValueError(
                "SHIGURE identity reference admission is not strict enough"
            )

    reference_id = uuid.uuid4().hex
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if admission is not None:
            binding = conn.execute(
                f"""
                SELECT binding_id
                FROM {SHIGURE_OBJECT_BINDING_TABLE}
                WHERE binding_id = ?
                  AND source_epoch_id = ?
                  AND raw_shigure_object_id = ?
                  AND display_object_id = ?
                  AND status = 'ACTIVE'
                """,
                (
                    str(admission["admission_binding_id"]),
                    str(admission["admission_source_epoch_id"]),
                    str(admission["admission_raw_shigure_object_id"]),
                    str(display_object_id),
                ),
            ).fetchone()
            if binding is None:
                raise ValueError(
                    "SHIGURE identity admission binding is unavailable"
                )
            anchor = conn.execute(
                f"""
                SELECT reference_id
                FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
                WHERE reference_id = ?
                  AND display_object_id = ?
                  AND active = 1
                """,
                (
                    str(admission["admission_reference_id"]),
                    str(display_object_id),
                ),
            ).fetchone()
            if anchor is None:
                raise ValueError(
                    "SHIGURE identity admission anchor is unavailable"
                )
        conn.execute(
            f"""
            INSERT INTO {OBJECT_IDENTITY_REFERENCE_TABLE} (
                reference_id, display_object_id, source, source_event_uid,
                source_task_id, image_path, mask_path, embedding_path,
                view_hash, quality_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(display_object_id, view_hash) DO UPDATE SET
                active = 1,
                image_path = excluded.image_path,
                mask_path = COALESCE(excluded.mask_path, mask_path),
                embedding_path = COALESCE(excluded.embedding_path, embedding_path),
                quality_json = excluded.quality_json
            """,
            (
                reference_id,
                str(display_object_id),
                source_value,
                source_event_uid,
                source_task_id,
                normalize_path_for_storage(image_path),
                normalize_path_for_storage(mask_path) if mask_path else None,
                normalize_path_for_storage(embedding_path) if embedding_path else None,
                str(view_hash),
                json.dumps(quality if quality is not None else {}, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            f"""
            SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
            WHERE display_object_id = ? AND view_hash = ?
            """,
            (str(display_object_id), str(view_hash)),
        ).fetchone()
    return dict(row)


def list_object_identity_references(
    display_object_id: str, *, limit: int = 20
) -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
            WHERE display_object_id = ? AND active = 1
            ORDER BY created_at DESC, reference_id DESC
            LIMIT ?
            """,
            (str(display_object_id), max(1, min(int(limit), 100))),
        ).fetchall()
    return [dict(row) for row in rows]


def set_object_identity_reference_embedding(
    reference_id: str,
    *,
    embedding_path: str | Path,
) -> Dict[str, Any]:
    """Attach a durable embedding sidecar to one active identity reference."""

    initialize_task_table()
    reference_id = str(reference_id or "").strip()
    if not reference_id:
        raise ValueError("reference_id is required")
    resolved = resolve_project_path(embedding_path, require_exists=True)
    root = IDENTITY_REFERENCE_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("embedding_path must be inside the identity reference root") from exc
    if resolved.name != "embedding.json":
        raise ValueError("identity embedding sidecar must be named embedding.json")
    normalized = normalize_path_for_storage(resolved)
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE} WHERE reference_id = ? AND active = 1",
            (reference_id,),
        ).fetchone()
        if row is None:
            raise ValueError("identity reference is missing or inactive")
        conn.execute(
            f"UPDATE {OBJECT_IDENTITY_REFERENCE_TABLE} SET embedding_path = ? WHERE reference_id = ?",
            (normalized, reference_id),
        )
        updated = conn.execute(
            f"SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE} WHERE reference_id = ?",
            (reference_id,),
        ).fetchone()
    return dict(updated)


def upsert_identity_sync_job(
    *,
    sync_job_id: str,
    kind: str,
    status: str,
    runtime_session_id: str | None = None,
    source_epoch_id: str | None = None,
    display_object_id: str | None = None,
    candidate_limit: int = 5,
    result: Any = None,
    error_message: str | None = None,
) -> Dict[str, Any]:
    initialize_task_table()
    kind_value = str(kind or "").strip().upper()
    status_value = str(status or "").strip().upper()
    if kind_value not in {"STARTUP_RECOVERY", "HOLOLENS_CAPTURE"}:
        raise ValueError("invalid identity sync kind")
    if status_value not in {"PENDING", "RUNNING", "COMPLETED", "FAILED"}:
        raise ValueError("invalid identity sync status")
    now = _utc_now_text()
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {IDENTITY_SYNC_JOB_TABLE} (
                sync_job_id, kind, status, runtime_session_id, source_epoch_id,
                display_object_id, candidate_limit, result_json, error_message,
                started_at, completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sync_job_id) DO UPDATE SET
                status = excluded.status,
                result_json = excluded.result_json,
                error_message = excluded.error_message,
                started_at = COALESCE(started_at, excluded.started_at),
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                str(sync_job_id),
                kind_value,
                status_value,
                runtime_session_id,
                source_epoch_id,
                display_object_id,
                max(1, min(int(candidate_limit), 5)),
                _dump_optional_json(result),
                error_message,
                now if status_value == "RUNNING" else None,
                now if status_value in {"COMPLETED", "FAILED"} else None,
                now,
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {IDENTITY_SYNC_JOB_TABLE} WHERE sync_job_id = ?",
            (str(sync_job_id),),
        ).fetchone()
    return dict(row)


def get_identity_sync_job(sync_job_id: str) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    sync_job_id = str(sync_job_id or "").strip()
    if not sync_job_id:
        raise ValueError("sync_job_id is required")
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {IDENTITY_SYNC_JOB_TABLE} WHERE sync_job_id = ?",
            (sync_job_id,),
        ).fetchone()
    return _row_to_dict(row)
