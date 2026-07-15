import json
import math
import sqlite3
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from config import (
    ARUCO_ANCHOR_MARKER_ID,
    ARUCO_SYNC_MARKER_REGISTRY_ON_START,
    SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
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
DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE = "display_object_origin_history"
OBJECT_IDENTITY_REFERENCE_TABLE = "object_identity_references"
IDENTITY_SYNC_JOB_TABLE = "identity_sync_jobs"
SHIGURE_SCHEMA_VERSION = 3
DISPLAY_OBJECT_ORIGIN_HISTORY_LIMIT = 5
HOLOLENS_IDENTITY_REFERENCE_LIMIT = 5
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


def _create_display_object_origin_history_table_sql() -> str:
    return f"""
        CREATE TABLE {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_uid TEXT NOT NULL UNIQUE,
            display_object_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('INITIALIZATION', 'TAKE_OUT')),
            model_revision INTEGER NOT NULL DEFAULT 0,
            pose_aruco_json TEXT NOT NULL,
            source_epoch_id TEXT,
            canonical_event_uid TEXT,
            binding_id TEXT,
            raw_shigure_object_id TEXT,
            occurred_at TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}',
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


def _v3_table_builders() -> tuple[tuple[str, Any], ...]:
    """Return every application table that belongs to the strict v3 schema."""

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
            DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE,
            _create_display_object_origin_history_table_sql,
        ),
        (
            OBJECT_IDENTITY_REFERENCE_TABLE,
            _create_object_identity_reference_table_sql,
        ),
        (IDENTITY_SYNC_JOB_TABLE, _create_identity_sync_job_table_sql),
    )


def _v3_index_sql() -> tuple[str, ...]:
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
        CREATE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_display_status
        ON {SHIGURE_OBJECT_BINDING_TABLE} (
            source_epoch_id, display_object_id, status, valid_from DESC
        )
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
        CREATE INDEX IF NOT EXISTS idx_{DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}_display_history
        ON {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} (display_object_id, id DESC)
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


def _v2_table_builders() -> tuple[tuple[str, Any], ...]:
    """Describe the former strict v2 schema for the explicit upgrade only."""

    return tuple(
        item
        for item in _v3_table_builders()
        if item[0] != DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE
    )


def _v2_index_sql() -> tuple[str, ...]:
    retained = tuple(
        sql
        for sql in _v3_index_sql()
        if f"idx_{SHIGURE_OBJECT_BINDING_TABLE}_display_status" not in sql
        and f"idx_{DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}_display_history" not in sql
    )
    return retained + (
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_display
        ON {SHIGURE_OBJECT_BINDING_TABLE} (source_epoch_id, display_object_id)
        WHERE status = 'ACTIVE'
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
        "Task database is not the strict Shigure v3 schema: "
        f"{detail}. If this is a strict v2 database, run: "
        "python code/migrate_shigure_v3_data.py --apply"
    )


def _validate_v3_schema(conn: sqlite3.Connection) -> None:
    builders = dict(_v3_table_builders())
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

    for expected_sql in _v3_index_sql():
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
    retired_active_display_index = (
        f"idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_display"
    )
    if _schema_object_sql(
        conn, "index", retired_active_display_index
    ) is not None:
        raise _migration_required(
            f"retired one-to-one index is still present={retired_active_display_index}"
        )

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


def _strict_v2_schema_error(detail: str) -> RuntimeError:
    return RuntimeError(f"Task database is not the strict Shigure v2 schema: {detail}")


def _validate_v2_schema(conn: sqlite3.Connection) -> None:
    """Validate v2 exactly; used only by explicit offline migrations."""

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
        raise _strict_v2_schema_error("; ".join(details))

    mismatched = []
    for table_name, builder in builders.items():
        actual_sql = _table_sql(conn, table_name)
        if actual_sql is None or _normalized_table_definition(
            actual_sql
        ) != _normalized_table_definition(builder()):
            mismatched.append(table_name)
    if mismatched:
        raise _strict_v2_schema_error(
            "table DDL mismatch=" + ",".join(sorted(mismatched))
        )

    for expected_sql in _v2_index_sql():
        tokens = _normalized_schema_sql(expected_sql).split()
        index_name = tokens[tokens.index("INDEX") + 1]
        if index_name == "IF":
            index_name = tokens[tokens.index("INDEX") + 4]
        actual_sql = _schema_object_sql(conn, "index", index_name)
        if actual_sql is None or _normalized_schema_sql(
            actual_sql
        ) != _normalized_schema_sql(expected_sql):
            raise _strict_v2_schema_error(f"index DDL mismatch={index_name}")

    metadata = conn.execute(
        f"""
        SELECT schema_version, migrated_at, detail_json
        FROM {SCHEMA_METADATA_TABLE}
        WHERE schema_name = 'shigure_runtime'
        """
    ).fetchone()
    try:
        version_matches = metadata is not None and int(metadata["schema_version"]) == 2
    except (TypeError, ValueError):
        version_matches = False
    if not version_matches:
        found = None if metadata is None else metadata["schema_version"]
        raise _strict_v2_schema_error(
            f"schema_metadata shigure_runtime version must be 2, got {found}"
        )
    if not str(metadata["migrated_at"] or "").strip():
        raise _strict_v2_schema_error("schema_metadata.migrated_at is empty")
    try:
        detail = json.loads(str(metadata["detail_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise _strict_v2_schema_error(
            "schema_metadata.detail_json is invalid JSON"
        ) from exc
    if not isinstance(detail, dict):
        raise _strict_v2_schema_error(
            "schema_metadata.detail_json must be an object"
        )


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
                2,
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


def _cap_hololens_identity_references(
    conn: sqlite3.Connection,
    display_object_id: str,
) -> int:
    """Deactivate all but the five newest HoloLens identity views."""

    stale = conn.execute(
        f"""
        SELECT reference_id
        FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
        WHERE display_object_id = ?
          AND source = 'HOLOLENS'
          AND active = 1
        ORDER BY created_at DESC, rowid DESC
        LIMIT -1 OFFSET ?
        """,
        (str(display_object_id), HOLOLENS_IDENTITY_REFERENCE_LIMIT),
    ).fetchall()
    if not stale:
        return 0
    reference_ids = [str(row["reference_id"]) for row in stale]
    placeholders = ", ".join("?" for _ in reference_ids)
    conn.execute(
        f"""
        UPDATE {OBJECT_IDENTITY_REFERENCE_TABLE}
        SET active = 0
        WHERE reference_id IN ({placeholders})
        """,
        tuple(reference_ids),
    )
    return len(reference_ids)


def _origin_backfill_pose(raw_pose: Any) -> tuple[dict[str, Any], tuple[float, float, float]]:
    pose = json.loads(str(raw_pose)) if isinstance(raw_pose, str) else raw_pose
    if not isinstance(pose, dict):
        raise ValueError("origin pose must be an object")
    position = _origin_pose_position(pose)
    quaternion = pose.get("rotation_quaternion_xyzw")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("origin pose quaternion must contain four values")
    quaternion_values = tuple(float(value) for value in quaternion)
    if not all(math.isfinite(value) for value in quaternion_values):
        raise ValueError("origin pose quaternion must be finite")
    if sum(value * value for value in quaternion_values) <= 1.0e-12:
        raise ValueError("origin pose quaternion cannot be zero")
    scale = pose.get("scale")
    if scale is not None:
        if not isinstance(scale, (list, tuple)) or len(scale) != 3:
            raise ValueError("origin pose scale must contain three values")
        scale_values = tuple(float(value) for value in scale)
        if not all(math.isfinite(value) for value in scale_values):
            raise ValueError("origin pose scale must be finite")
    return dict(pose), position


def _origin_time_key(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except (OverflowError, ValueError):
        return 0.0


def _backfill_origins_from_hololens_pose_history(
    conn: sqlite3.Connection,
    *,
    remove_unsafe_v2_take_out_rows: bool = False,
) -> Dict[str, int]:
    """Merge calibrated capture poses with native-v3 origins, oldest first.

    The merge is an offline, one-shot operation.  It preserves verified v3
    FoundationPose rows, rejects the unsafe legacy lifecycle import, merges
    adjacent positions within 20 cm, and rebuilds per-object cursor order so
    the newest retained position still has the greatest integer cursor.
    """

    candidates_by_display: Dict[str, list[dict[str, Any]]] = {}
    removed_unsafe = 0
    invalid_pose_count = 0
    native_candidate_count = 0
    existing_rows = conn.execute(
        f"SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} ORDER BY id ASC"
    ).fetchall()
    stable_order = 0
    for row in existing_rows:
        try:
            detail = json.loads(str(row["detail_json"] or "{}"))
        except (TypeError, json.JSONDecodeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        if (
            remove_unsafe_v2_take_out_rows
            and detail.get("migrated_from") == "object_lifecycle_events"
        ):
            removed_unsafe += 1
            continue
        try:
            pose, position = _origin_backfill_pose(row["pose_aruco_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid_pose_count += 1
            continue
        is_hololens_backfill = (
            detail.get("backfilled_from") == "display_object_pose_history"
        )
        if not is_hololens_backfill:
            native_candidate_count += 1
        stable_order += 1
        candidates_by_display.setdefault(str(row["display_object_id"]), []).append(
            {
                "origin_uid": str(row["origin_uid"]),
                "display_object_id": str(row["display_object_id"]),
                "kind": str(row["kind"]),
                "model_revision": int(row["model_revision"] or 0),
                "pose": pose,
                "position": position,
                "source_epoch_id": row["source_epoch_id"],
                "canonical_event_uid": row["canonical_event_uid"],
                "binding_id": row["binding_id"],
                "raw_shigure_object_id": row["raw_shigure_object_id"],
                "occurred_at": str(row["occurred_at"]),
                "detail": detail,
                "created_at": str(row["created_at"] or row["occurred_at"]),
                "native": not is_hololens_backfill,
                "stable_order": stable_order,
            }
        )

    history_rows = conn.execute(
        f"""
        SELECT history.display_object_id, history.task_id,
               history.model_revision, history.pose_revision,
               history.pose_aruco_json, history.captured_at,
               history.created_at, history.updated_at
        FROM {DISPLAY_OBJECT_POSE_HISTORY_TABLE} AS history
        ORDER BY history.display_object_id ASC,
                 history.pose_revision ASC,
                 history.id ASC
        """
    ).fetchall()
    for history in history_rows:
        try:
            pose, position = _origin_backfill_pose(history["pose_aruco_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid_pose_count += 1
            continue
        stable_order += 1
        display_object_id = str(history["display_object_id"])
        task_id = str(history["task_id"])
        occurred_at = str(
            history["captured_at"]
            or history["created_at"]
            or history["updated_at"]
            or _utc_now_text()
        )
        candidates_by_display.setdefault(display_object_id, []).append(
            {
                "origin_uid": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"display-object-origin:hololens-pose-history:{task_id}",
                ).hex,
                "display_object_id": display_object_id,
                "kind": "INITIALIZATION",
                "model_revision": int(history["model_revision"] or 0),
                "pose": pose,
                "position": position,
                "source_epoch_id": None,
                "canonical_event_uid": None,
                "binding_id": None,
                "raw_shigure_object_id": None,
                "occurred_at": occurred_at,
                "detail": {
                    "backfilled_from": "display_object_pose_history",
                    "task_id": task_id,
                    "pose_revision": int(history["pose_revision"] or 0),
                },
                "created_at": str(history["created_at"] or occurred_at),
                "native": False,
                "stable_order": stable_order,
            }
        )

    retained_by_display: Dict[str, list[dict[str, Any]]] = {}
    deduplicated = 0
    for display_object_id, candidates in candidates_by_display.items():
        candidates.sort(
            key=lambda item: (
                _origin_time_key(item["occurred_at"]),
                1 if item["native"] else 0,
                int(item["stable_order"]),
            )
        )
        unique: list[dict[str, Any]] = []
        for candidate in candidates:
            if unique:
                previous = unique[-1]
                distance = math.sqrt(
                    sum(
                        (
                            candidate["position"][index]
                            - previous["position"][index]
                        )
                        ** 2
                        for index in range(3)
                    )
                )
                if distance < 0.2:
                    deduplicated += 1
                    if candidate["native"] and not previous["native"]:
                        unique[-1] = candidate
                    continue
            unique.append(candidate)
        retained_by_display[display_object_id] = unique[
            -DISPLAY_OBJECT_ORIGIN_HISTORY_LIMIT:
        ]

    conn.execute(f"DELETE FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}")
    conn.execute(
        "DELETE FROM sqlite_sequence WHERE name = ?",
        (DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE,),
    )
    retained_native = 0
    retained_hololens = 0
    for display_object_id in sorted(retained_by_display):
        for origin in retained_by_display[display_object_id]:
            retained_native += int(bool(origin["native"]))
            retained_hololens += int(not bool(origin["native"]))
            conn.execute(
                f"""
                INSERT INTO {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} (
                    origin_uid, display_object_id, kind, model_revision,
                    pose_aruco_json, source_epoch_id, canonical_event_uid,
                    binding_id, raw_shigure_object_id, occurred_at,
                    detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    origin["origin_uid"],
                    display_object_id,
                    origin["kind"],
                    int(origin["model_revision"]),
                    json.dumps(origin["pose"], ensure_ascii=False),
                    origin["source_epoch_id"],
                    origin["canonical_event_uid"],
                    origin["binding_id"],
                    origin["raw_shigure_object_id"],
                    origin["occurred_at"],
                    json.dumps(origin["detail"], ensure_ascii=False),
                    origin["created_at"],
                ),
            )

    return {
        "backfilled_hololens_origins": retained_hololens,
        "preserved_native_v3_origins": retained_native,
        "deduplicated_nearby_origin_candidates": deduplicated,
        "skipped_invalid_origin_poses": invalid_pose_count,
        "removed_unsafe_v2_take_out_origins": removed_unsafe,
        "native_origin_candidates": native_candidate_count,
    }


def migrate_strict_v2_database_to_v3_once() -> Dict[str, Any]:
    """Atomically upgrade an already-strict v2 database to strict v3.

    This function intentionally does not create its own backup because it is
    also useful in focused tests.  The operator-facing
    ``migrate_shigure_v3_data.py`` command always creates and verifies a
    consistent SQLite backup before calling it.
    """

    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return {"status": "already_initialized", "schema_version": 3}

    with _get_connection() as conn:
        try:
            _validate_v3_schema(conn)
        except RuntimeError:
            pass
        else:
            _SCHEMA_INITIALIZED = True
            return {"status": "already_v3", "schema_version": 3}

        _validate_v2_schema(conn)
        old_metadata = conn.execute(
            f"""
            SELECT detail_json
            FROM {SCHEMA_METADATA_TABLE}
            WHERE schema_name = 'shigure_runtime'
            """
        ).fetchone()
        try:
            previous_detail = json.loads(str(old_metadata["detail_json"]))
        except (TypeError, json.JSONDecodeError):
            previous_detail = {}

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(_create_display_object_origin_history_table_sql())
        conn.execute(
            f"DROP INDEX idx_{SHIGURE_OBJECT_BINDING_TABLE}_active_display"
        )
        ignored_legacy_take_out_poses = int(
            conn.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
                WHERE action = 'TAKE_OUT' AND pose_aruco_json IS NOT NULL
                """
            ).fetchone()["value"]
            or 0
        )
        origin_backfill = _backfill_origins_from_hololens_pose_history(conn)

        deactivated_shigure_count = conn.execute(
            f"""
            UPDATE {OBJECT_IDENTITY_REFERENCE_TABLE}
            SET active = 0
            WHERE source = 'SHIGURE' AND active = 1
            """
        ).rowcount
        deactivated_hololens_count = 0
        display_rows = conn.execute(
            f"""
            SELECT DISTINCT display_object_id
            FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
            WHERE source = 'HOLOLENS' AND active = 1
            """
        ).fetchall()
        for display_row in display_rows:
            deactivated_hololens_count += _cap_hololens_identity_references(
                conn, str(display_row["display_object_id"])
            )

        for index_sql in _v3_index_sql():
            conn.execute(index_sql)

        detail = {
            "migration": "strict_v2_to_v3",
            "previous_detail": previous_detail,
            "active_display_binding_index": "replaced_with_alias_index",
            "ignored_unverified_legacy_take_out_poses": ignored_legacy_take_out_poses,
            **origin_backfill,
            "hololens_origin_backfill_v1": {
                **origin_backfill,
                "applied_at": _utc_now_text(),
            },
            "deactivated_shigure_identity_references": deactivated_shigure_count,
            "deactivated_excess_hololens_references": deactivated_hololens_count,
        }
        conn.execute(
            f"""
            UPDATE {SCHEMA_METADATA_TABLE}
            SET schema_version = ?, migrated_at = ?, detail_json = ?
            WHERE schema_name = 'shigure_runtime'
            """,
            (
                SHIGURE_SCHEMA_VERSION,
                _utc_now_text(),
                json.dumps(detail, ensure_ascii=False),
            ),
        )
        _validate_v3_schema(conn)
        conn.commit()

    _SCHEMA_INITIALIZED = True
    return {"status": "migrated", "schema_version": 3, **detail}


def repair_v3_origin_history_from_hololens_once() -> Dict[str, Any]:
    """One-shot offline merge of legacy HoloLens poses and native-v3 origins."""

    global _SCHEMA_INITIALIZED
    with _get_connection() as conn:
        _validate_v3_schema(conn)
        metadata = conn.execute(
            f"""
            SELECT detail_json
            FROM {SCHEMA_METADATA_TABLE}
            WHERE schema_name = 'shigure_runtime'
            """
        ).fetchone()
        try:
            detail = json.loads(str(metadata["detail_json"] or "{}"))
        except (TypeError, json.JSONDecodeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        previous_result = detail.get("hololens_origin_backfill_v1")
        if isinstance(previous_result, dict):
            _SCHEMA_INITIALIZED = True
            return {
                "status": "already_repaired",
                "schema_version": 3,
                **previous_result,
            }

        conn.execute("BEGIN IMMEDIATE")
        result = _backfill_origins_from_hololens_pose_history(
            conn,
            remove_unsafe_v2_take_out_rows=True,
        )
        detail["hololens_origin_backfill_v1"] = {
            **result,
            "applied_at": _utc_now_text(),
        }
        conn.execute(
            f"""
            UPDATE {SCHEMA_METADATA_TABLE}
            SET detail_json = ?
            WHERE schema_name = 'shigure_runtime'
            """,
            (json.dumps(detail, ensure_ascii=False),),
        )
        _validate_v3_schema(conn)
        conn.commit()
    _SCHEMA_INITIALIZED = True
    return {"status": "repaired", "schema_version": 3, **result}


def initialize_task_table() -> None:
    """Create a brand-new v3 database or validate an existing one read-only.

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
            for _, builder in _v3_table_builders():
                conn.execute(builder())
            for index_sql in _v3_index_sql():
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
                    json.dumps({"installation": "fresh_v3"}, ensure_ascii=False),
                ),
            )
            _validate_v3_schema(conn)
            conn.commit()
        else:
            _validate_v3_schema(conn)

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
    where = [
        "display_object_id IS NOT NULL",
        "binding_status = 'bound'",
        "LOWER(source) = 'hololens'",
    ]
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


def list_identity_candidate_captures_by_display(
    *,
    display_object_limit: int = SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
    references_per_display: int = HOLOLENS_IDENTITY_REFERENCE_LIMIT,
    exclude_capture_instance_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Return a fair newest-view window for every selected display object.

    A global backward scan can be exhausted by one heavily photographed
    object before another object contributes even one identity view.  Rank
    captures inside each display object first, then retain at most fifty
    objects and five HoloLens views for each object.
    """

    initialize_task_table()
    display_object_limit = max(
        1,
        min(int(display_object_limit), SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS),
    )
    references_per_display = max(
        1,
        min(int(references_per_display), HOLOLENS_IDENTITY_REFERENCE_LIMIT),
    )
    where = [
        "display_object_id IS NOT NULL",
        "binding_status = 'bound'",
        "LOWER(source) = 'hololens'",
    ]
    predicate_values: list[Any] = []
    if exclude_capture_instance_id:
        where.append("capture_instance_id != ?")
        predicate_values.append(str(exclude_capture_instance_id))
    predicate = " AND ".join(where)
    ranked_predicate = (
        predicate.replace("capture_instance_id", "capture.capture_instance_id")
        .replace("binding_status", "capture.binding_status")
        .replace("display_object_id", "capture.display_object_id")
        .replace("LOWER(source)", "LOWER(capture.source)")
    )
    query_values = (
        predicate_values
        + [display_object_limit]
        + predicate_values
        + [references_per_display]
    )
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            WITH selected_displays AS (
                SELECT display_object_id,
                       MAX(COALESCE(NULLIF(timestamp, ''), created_at)) AS newest_capture_at,
                       MAX(id) AS newest_capture_id
                FROM {CAPTURE_INSTANCE_TABLE}
                WHERE {predicate}
                GROUP BY display_object_id
                ORDER BY newest_capture_at DESC,
                         newest_capture_id DESC,
                         display_object_id ASC
                LIMIT ?
            ), ranked_capture_ids AS (
                SELECT capture.id,
                       ROW_NUMBER() OVER (
                           PARTITION BY capture.display_object_id
                           ORDER BY COALESCE(
                                        NULLIF(capture.timestamp, ''),
                                        capture.created_at
                                    ) DESC,
                                    capture.id DESC
                       ) AS view_rank
                FROM {CAPTURE_INSTANCE_TABLE} AS capture
                INNER JOIN selected_displays AS selected
                    ON selected.display_object_id = capture.display_object_id
                WHERE {ranked_predicate}
            )
            SELECT capture.*
            FROM {CAPTURE_INSTANCE_TABLE} AS capture
            INNER JOIN ranked_capture_ids AS ranked
                ON ranked.id = capture.id
            INNER JOIN selected_displays AS selected
                ON selected.display_object_id = capture.display_object_id
            WHERE ranked.view_rank <= ?
            ORDER BY selected.newest_capture_at DESC,
                     selected.newest_capture_id DESC,
                     capture.display_object_id ASC,
                     COALESCE(NULLIF(capture.timestamp, ''), capture.created_at) DESC,
                     capture.id DESC
            """,
            tuple(query_values),
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
    if pose_aruco is not None and not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco must be an object when provided")
    model_task_id = str(active_model_task_id or capture_task_id).strip()
    capture_time = str(captured_at or "").strip() or None
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
        capture_record = conn.execute(
            f"SELECT * FROM {CAPTURE_INSTANCE_TABLE} WHERE task_id = ?",
            (capture_task_id,),
        ).fetchone()
        if capture_record is not None:
            recorded_display_object_id = str(
                capture_record["display_object_id"] or ""
            ).strip()
            if (
                recorded_display_object_id
                and recorded_display_object_id != display_object_id
            ):
                conn.rollback()
                raise ValueError(
                    f"capture task {capture_task_id} is already bound to "
                    f"display object {recorded_display_object_id}"
                )
            capture_time = (
                capture_time
                or str(capture_record["timestamp"] or "").strip()
                or None
            )
        latest_task_id = str(state["latest_hololens_task_id"] or "").strip()
        latest_capture_time = str(
            state["latest_hololens_captured_at"] or ""
        ).strip()
        if capture_time is None:
            capture_time = (
                latest_capture_time
                if latest_task_id == capture_task_id and latest_capture_time
                else now
            )
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
            if pose_aruco is not None:
                serialized_pose = json.dumps(pose_aruco, ensure_ascii=False)
                conn.execute(
                    f"""
                    UPDATE {DISPLAY_OBJECT_POSE_HISTORY_TABLE}
                    SET pose_aruco_json = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    (serialized_pose, now, capture_task_id),
                )
                # ArUco retro-sync can legitimately re-express an already
                # committed capture. Refresh the latest pointer without
                # allocating another user-visible pose revision.
                if (
                    str(state["latest_hololens_task_id"] or "").strip()
                    == capture_task_id
                ):
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

        # Local-only captures have no pose-history row.  Use their durable
        # capture timestamp (and the current task pointer) as the idempotency
        # key so a delayed replay cannot clear or roll back a newer capture.
        if latest_task_id == capture_task_id and pose_aruco is None:
            conn.commit()
            return dict(state)
        if latest_task_id and latest_task_id != capture_task_id and latest_capture_time:
            incoming_time_key = _origin_time_key(capture_time)
            latest_time_key = _origin_time_key(latest_capture_time)
            if incoming_time_key <= latest_time_key:
                conn.commit()
                return dict(state)

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

        if pose_aruco is None:
            conn.execute(
                f"""
                UPDATE {DISPLAY_OBJECT_STATE_TABLE}
                SET active_model_revision = ?,
                    active_model_task_id = ?,
                    active_model_asset_hash = COALESCE(?, active_model_asset_hash),
                    latest_hololens_pose_aruco_json = NULL,
                    latest_hololens_task_id = ?,
                    latest_hololens_captured_at = ?,
                    updated_at = ?
                WHERE display_object_id = ?
                """,
                (
                    active_revision,
                    resolved_model_task_id,
                    asset_hash,
                    capture_task_id,
                    str(capture_time),
                    now,
                    display_object_id,
                ),
            )
        else:
            pose_revision = int(state["latest_hololens_pose_revision"] or 0) + 1
            serialized_pose = json.dumps(pose_aruco, ensure_ascii=False)
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
                    serialized_pose,
                    str(capture_time),
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
                    serialized_pose,
                    capture_task_id,
                    str(capture_time),
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
                    AND reference.source = 'HOLOLENS'
              )
            ORDER BY state.updated_at DESC, state.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_live_display_object_states(*, limit: int) -> List[Dict[str, Any]]:
    """Return every active downloadable model, including local-only captures.

    Model delivery is a catalog concern, not a Shigure-presence concern. In
    particular, a recorder restart can legitimately leave presence UNKNOWN
    until the next tracking observation; that must not make retained models
    or their history disappear from HoloLens.
    """

    initialize_task_table()
    limit = max(1, min(int(limit), 50))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {DISPLAY_OBJECT_STATE_TABLE}
            WHERE active_model_revision > 0
              AND active_model_task_id IS NOT NULL
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


def _close_epoch_lifecycle_authority_conn(
    conn: sqlite3.Connection,
    source_epoch_ids: Iterable[str],
    *,
    reason: str,
    audited_at: str,
) -> Dict[str, int]:
    """Terminally close lifecycle authority before an epoch is revoked.

    Pending lifecycle evidence can be rejected safely.  A RESOLVED event may
    already have authorized external work, so an event lacking its lifecycle
    row is retained as RESOLVED and explicitly marked as an unreplayable
    orphan.  Replaying it after the epoch binding is revoked would be unsafe.
    """

    epoch_ids = sorted(
        {str(value).strip() for value in source_epoch_ids if str(value).strip()}
    )
    result = {
        "rejected_pending_lifecycle_events": 0,
        "audited_resolved_without_lifecycle": 0,
    }
    if not epoch_ids:
        return result
    placeholders = ", ".join("?" for _ in epoch_ids)

    pending = conn.execute(
        f"""
        SELECT *
        FROM {SHIGURE_CANONICAL_EVENT_TABLE}
        WHERE source_epoch_id IN ({placeholders})
          AND action IN ('TAKE_OUT', 'BRING_IN')
          AND resolution_status IN ('UNRESOLVED', 'AMBIGUOUS', 'CONFLICT')
        """,
        tuple(epoch_ids),
    ).fetchall()
    for event in pending:
        try:
            detail = json.loads(str(event["detail_json"] or "{}"))
        except (TypeError, json.JSONDecodeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {"previous_detail": detail}
        detail["lifecycle_terminal_audit"] = {
            "status": "REJECTED_PENDING_EPOCH_CLOSE",
            "reason": str(reason),
            "audited_at": str(audited_at),
            "source_epoch_id": str(event["source_epoch_id"]),
        }
        updated = conn.execute(
            f"""
            UPDATE {SHIGURE_CANONICAL_EVENT_TABLE}
            SET resolution_status = 'REJECTED',
                resolution_method = ?,
                detail_json = ?
            WHERE event_uid = ?
              AND resolution_status IN ('UNRESOLVED', 'AMBIGUOUS', 'CONFLICT')
            """,
            (
                str(reason),
                json.dumps(detail, ensure_ascii=False),
                str(event["event_uid"]),
            ),
        )
        result["rejected_pending_lifecycle_events"] += int(
            updated.rowcount or 0
        )

    resolved_orphans = conn.execute(
        f"""
        SELECT event.*
        FROM {SHIGURE_CANONICAL_EVENT_TABLE} AS event
        LEFT JOIN {OBJECT_LIFECYCLE_EVENT_TABLE} AS lifecycle
          ON lifecycle.canonical_event_uid = event.event_uid
        WHERE event.source_epoch_id IN ({placeholders})
          AND event.action IN ('TAKE_OUT', 'BRING_IN')
          AND event.resolution_status = 'RESOLVED'
          AND lifecycle.canonical_event_uid IS NULL
        """,
        tuple(epoch_ids),
    ).fetchall()
    for event in resolved_orphans:
        try:
            detail = json.loads(str(event["detail_json"] or "{}"))
        except (TypeError, json.JSONDecodeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {"previous_detail": detail}
        detail["lifecycle_terminal_audit"] = {
            "status": "ORPHANED_RESOLVED_WITHOUT_LIFECYCLE",
            "reason": str(reason),
            "audited_at": str(audited_at),
            "source_epoch_id": str(event["source_epoch_id"]),
            "safe_replay": False,
        }
        conn.execute(
            f"""
            UPDATE {SHIGURE_CANONICAL_EVENT_TABLE}
            SET detail_json = ?
            WHERE event_uid = ? AND resolution_status = 'RESOLVED'
            """,
            (
                json.dumps(detail, ensure_ascii=False),
                str(event["event_uid"]),
            ),
        )
        result["audited_resolved_without_lifecycle"] += 1
    return result


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
        stale_epoch_rows = conn.execute(
            f"""
            SELECT source_epoch_id
            FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE status = 'ACTIVE'
            """
        ).fetchall()
        lifecycle_audit = _close_epoch_lifecycle_authority_conn(
            conn,
            (row["source_epoch_id"] for row in stale_epoch_rows),
            reason="SERVER_RESTART_PENDING_LIFECYCLE",
            audited_at=now,
        )
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
        session_config = (
            dict(config) if isinstance(config, dict) else {"config": config}
        )
        session_config["server_restart_lifecycle_audit"] = lifecycle_audit
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
                json.dumps(session_config, ensure_ascii=False),
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
        active_epoch_rows = conn.execute(
            f"""
            SELECT source_epoch_id
            FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (runtime_session_id,),
        ).fetchall()
        _close_epoch_lifecycle_authority_conn(
            conn,
            (row["source_epoch_id"] for row in active_epoch_rows),
            reason="RUNTIME_CLOSED_PENDING_LIFECYCLE",
            audited_at=now,
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
        previous_epoch_rows = conn.execute(
            f"""
            SELECT source_epoch_id
            FROM {SHIGURE_SOURCE_EPOCH_TABLE}
            WHERE runtime_session_id = ? AND status = 'ACTIVE'
            """,
            (runtime_session_id,),
        ).fetchall()
        lifecycle_audit = _close_epoch_lifecycle_authority_conn(
            conn,
            (row["source_epoch_id"] for row in previous_epoch_rows),
            reason="SOURCE_EPOCH_CHANGED_PENDING_LIFECYCLE",
            audited_at=now,
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
    result = dict(row)
    result["_lifecycle_epoch_close_audit"] = lifecycle_audit
    return result


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


def _establish_shigure_binding_conn(
    conn: sqlite3.Connection,
    *,
    runtime_session_id: str,
    source_epoch_id: str,
    raw_shigure_object_id: str,
    display_object_id: str,
    established_by: str,
    established_event_uid: str | None = None,
    confidence: float | None = None,
    detail: Any = None,
    allow_existing_same: bool = True,
    now: str | None = None,
) -> Dict[str, Any]:
    runtime_session_id = str(runtime_session_id or "").strip()
    source_epoch_id = str(source_epoch_id or "").strip()
    raw_id = str(raw_shigure_object_id or "").strip()
    display_object_id = str(display_object_id or "").strip()
    established_by = str(established_by or "").strip()
    if not all((runtime_session_id, source_epoch_id, raw_id, display_object_id, established_by)):
        raise ValueError("all binding identity fields are required")
    now_value = str(now or _utc_now_text())
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
    if raw_binding is not None:
        if (
            allow_existing_same
            and str(raw_binding["display_object_id"]) == display_object_id
        ):
            return dict(raw_binding)
        raise ValueError("raw Shigure ID already has an active epoch binding")

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
            now_value,
            float(confidence) if confidence is not None else None,
            json.dumps(detail if detail is not None else {}, ensure_ascii=False),
        ),
    )
    row = conn.execute(
        f"SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE} WHERE binding_id = ?",
        (binding_id,),
    ).fetchone()
    return dict(row)


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
    """Bind one raw ID while allowing multiple aliases for a display object."""

    initialize_task_table()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return _establish_shigure_binding_conn(
            conn,
            runtime_session_id=runtime_session_id,
            source_epoch_id=source_epoch_id,
            raw_shigure_object_id=raw_shigure_object_id,
            display_object_id=display_object_id,
            established_by=established_by,
            established_event_uid=established_event_uid,
            confidence=confidence,
            detail=detail,
        )


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


def get_shigure_binding(binding_id: str) -> Optional[Dict[str, Any]]:
    """Return one binding by immutable id, including revoked audit rows."""

    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE} WHERE binding_id = ?",
            (str(binding_id),),
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


def list_display_object_alias_bindings(
    source_epoch_id: str,
    display_object_id: str,
) -> List[Dict[str, Any]]:
    """Return active aliases with the presentation-primary binding first."""

    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT binding.*
            FROM {SHIGURE_OBJECT_BINDING_TABLE} AS binding
            LEFT JOIN {DISPLAY_OBJECT_STATE_TABLE} AS state
              ON state.display_object_id = binding.display_object_id
            WHERE binding.source_epoch_id = ?
              AND binding.display_object_id = ?
              AND binding.status = 'ACTIVE'
            ORDER BY CASE
                         WHEN binding.binding_id = state.active_shigure_binding_id
                         THEN 0 ELSE 1
                     END ASC,
                     binding.valid_from DESC,
                     binding.binding_id DESC
            """,
            (str(source_epoch_id), str(display_object_id)),
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
        "takeaway": "TAKE_OUT",
        "take_away": "TAKE_OUT",
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
    raw_id_value = (
        str(raw_shigure_object_id).strip()
        if raw_shigure_object_id
        else None
    )
    binding_id_value = str(binding_id).strip() if binding_id else None
    display_object_id_value = (
        str(display_object_id).strip() if display_object_id else None
    )
    resolution_method_value = (
        str(resolution_method).strip() if resolution_method else None
    )

    def stored_path(value: str | Path | None) -> str | None:
        return normalize_path_for_storage(value) if value is not None else None

    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid,),
        ).fetchone()
        if existing is not None and str(
            existing["resolution_status"] or ""
        ).upper() not in {"RESOLVED", "REJECTED"}:
            for column, incoming in (
                ("raw_shigure_object_id", raw_id_value),
                ("binding_id", binding_id_value),
                ("display_object_id", display_object_id_value),
                ("resolution_method", resolution_method_value),
            ):
                stored = str(existing[column] or "").strip() or None
                if stored is not None and incoming is not None and stored != incoming:
                    raise ValueError(
                        "pending canonical event identity conflict: "
                        f"{column} stored={stored!r} incoming={incoming!r}"
                    )
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
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.raw_shigure_object_id
                    ELSE COALESCE(
                        {SHIGURE_CANONICAL_EVENT_TABLE}.raw_shigure_object_id,
                        excluded.raw_shigure_object_id
                    )
                END,
                binding_id = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.binding_id
                    ELSE COALESCE(
                        {SHIGURE_CANONICAL_EVENT_TABLE}.binding_id,
                        excluded.binding_id
                    )
                END,
                display_object_id = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.display_object_id
                    ELSE COALESCE(
                        {SHIGURE_CANONICAL_EVENT_TABLE}.display_object_id,
                        excluded.display_object_id
                    )
                END,
                resolution_status = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                    ELSE excluded.resolution_status
                END,
                resolution_method = CASE
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                    THEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_method
                    ELSE COALESCE(
                        {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_method,
                        excluded.resolution_method
                    )
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
                    WHEN {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
                         IN ('RESOLVED', 'REJECTED')
                         AND excluded.resolution_status
                             != {SHIGURE_CANONICAL_EVENT_TABLE}.resolution_status
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
                raw_id_value,
                binding_id_value,
                display_object_id_value,
                status,
                resolution_method_value,
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


def _merge_canonical_event_audit_detail(
    stored_detail_json: Any,
    detail: Any,
) -> str:
    """Merge object-shaped transition audit data without losing capture audit."""

    if detail is None:
        return str(stored_detail_json or "{}")
    try:
        stored_detail = json.loads(str(stored_detail_json or "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical event detail_json is invalid") from exc
    if isinstance(stored_detail, dict) and isinstance(detail, dict):
        detail = {**stored_detail, **detail}
    return json.dumps(detail, ensure_ascii=False)


def _resolve_pending_shigure_canonical_event_conn(
    conn: sqlite3.Connection,
    event_uid: str,
    binding_id: str,
    display_object_id: str,
    raw_id: str,
    resolution_method: str,
    detail: Any = None,
) -> Dict[str, Any]:
    event_uid_value = str(event_uid or "").strip()
    binding_id_value = str(binding_id or "").strip()
    display_object_id_value = str(display_object_id or "").strip()
    raw_id_value = str(raw_id or "").strip()
    method_value = str(resolution_method or "").strip()
    if not all(
        (
            event_uid_value,
            binding_id_value,
            display_object_id_value,
            raw_id_value,
            method_value,
        )
    ):
        raise ValueError(
            "event_uid, binding_id, display_object_id, raw_id and "
            "resolution_method are required"
        )

    event = conn.execute(
        f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
        (event_uid_value,),
    ).fetchone()
    if event is None:
        raise ValueError("unknown canonical event")

    status = str(event["resolution_status"] or "").upper()
    if status == "RESOLVED":
        expected = (
            binding_id_value,
            display_object_id_value,
            raw_id_value,
            method_value,
        )
        actual = (
            str(event["binding_id"] or ""),
            str(event["display_object_id"] or ""),
            str(event["raw_shigure_object_id"] or ""),
            str(event["resolution_method"] or ""),
        )
        if actual != expected:
            raise ValueError(
                "canonical event is already resolved to a different identity"
            )
        return dict(event)
    if status == "REJECTED":
        raise ValueError("rejected canonical events cannot be resolved")
    if status not in {"UNRESOLVED", "AMBIGUOUS", "CONFLICT"}:
        raise ValueError(f"canonical event has invalid pending status: {status}")

    binding = conn.execute(
        f"""
        SELECT binding.*
        FROM {SHIGURE_OBJECT_BINDING_TABLE} AS binding
        JOIN {SHIGURE_SOURCE_EPOCH_TABLE} AS epoch
          ON epoch.source_epoch_id = binding.source_epoch_id
         AND epoch.runtime_session_id = binding.runtime_session_id
         AND epoch.status = 'ACTIVE'
        WHERE binding.binding_id = ?
          AND binding.status = 'ACTIVE'
        """,
        (binding_id_value,),
    ).fetchone()
    if binding is None:
        raise ValueError("binding is not active in an active source epoch")
    if any(
        (
            str(binding["source_epoch_id"]) != str(event["source_epoch_id"]),
            str(binding["runtime_session_id"])
            != str(event["runtime_session_id"]),
            str(binding["raw_shigure_object_id"]) != raw_id_value,
            str(binding["display_object_id"]) != display_object_id_value,
        )
    ):
        raise ValueError(
            "active binding does not match the canonical event epoch/raw/display"
        )
    for column, supplied in (
        ("raw_shigure_object_id", raw_id_value),
        ("binding_id", binding_id_value),
        ("display_object_id", display_object_id_value),
        ("resolution_method", method_value),
    ):
        existing = str(event[column] or "").strip()
        if existing and existing != supplied:
            raise ValueError(
                f"canonical event {column} conflicts with the requested resolution"
            )

    cursor = conn.execute(
        f"""
        UPDATE {SHIGURE_CANONICAL_EVENT_TABLE}
        SET raw_shigure_object_id = ?,
            binding_id = ?,
            display_object_id = ?,
            resolution_status = 'RESOLVED',
            resolution_method = ?,
            detail_json = ?
        WHERE event_uid = ?
          AND resolution_status IN ('UNRESOLVED', 'AMBIGUOUS', 'CONFLICT')
        """,
        (
            raw_id_value,
            binding_id_value,
            display_object_id_value,
            method_value,
            _merge_canonical_event_audit_detail(event["detail_json"], detail),
            event_uid_value,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("canonical event resolution transition was not applied")
    resolved = conn.execute(
        f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
        (event_uid_value,),
    ).fetchone()
    return dict(resolved)


def resolve_pending_shigure_canonical_event(
    event_uid: str,
    binding_id: str,
    display_object_id: str,
    raw_id: str,
    resolution_method: str,
    detail: Any = None,
) -> Dict[str, Any]:
    """Resolve pending evidence only through its matching active epoch binding.

    RESOLVED and REJECTED are terminal decisions.  A matching RESOLVED row is
    returned unchanged so replay remains idempotent even after its binding has
    subsequently been revoked by the lifecycle transition.
    """

    initialize_task_table()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return _resolve_pending_shigure_canonical_event_conn(
            conn,
            event_uid,
            binding_id,
            display_object_id,
            raw_id,
            resolution_method,
            detail,
        )


def reject_pending_shigure_canonical_event(
    event_uid: str,
    reason: str,
    detail: Any = None,
) -> Dict[str, Any]:
    """Reject pending evidence while preserving either terminal decision."""

    initialize_task_table()
    event_uid_value = str(event_uid or "").strip()
    reason_value = str(reason or "").strip()
    if not event_uid_value or not reason_value:
        raise ValueError("event_uid and reason are required")

    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid_value,),
        ).fetchone()
        if event is None:
            raise ValueError("unknown canonical event")
        status = str(event["resolution_status"] or "").upper()
        if status in {"RESOLVED", "REJECTED"}:
            return dict(event)
        if status not in {"UNRESOLVED", "AMBIGUOUS", "CONFLICT"}:
            raise ValueError(f"canonical event has invalid pending status: {status}")

        rejection_detail: Any
        if detail is None:
            rejection_detail = {"rejection_reason": reason_value}
        elif isinstance(detail, dict):
            rejection_detail = {**detail, "rejection_reason": reason_value}
        else:
            rejection_detail = {
                "rejection_reason": reason_value,
                "rejection_detail": detail,
            }
        cursor = conn.execute(
            f"""
            UPDATE {SHIGURE_CANONICAL_EVENT_TABLE}
            SET resolution_status = 'REJECTED',
                resolution_method = ?,
                detail_json = ?
            WHERE event_uid = ?
              AND resolution_status IN ('UNRESOLVED', 'AMBIGUOUS', 'CONFLICT')
            """,
            (
                reason_value,
                _merge_canonical_event_audit_detail(
                    event["detail_json"], rejection_detail
                ),
                event_uid_value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("canonical event rejection transition was not applied")
        rejected = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid_value,),
        ).fetchone()
    return dict(rejected)


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
        current_state = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_STATE_TABLE} WHERE display_object_id = ?",
            (display_object_id,),
        ).fetchone()
        if (
            current_state is not None
            and str(current_state["presence"]) == "PRESENT"
            and str(current_state["active_shigure_binding_id"] or "")
            == str(binding_id)
        ):
            return dict(current_state)
        conn.execute(
            f"""
            UPDATE {DISPLAY_OBJECT_STATE_TABLE}
            SET presence = 'PRESENT',
                presence_epoch = presence_epoch + CASE
                    WHEN presence = 'PRESENT' THEN 0 ELSE 1
                END,
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
    _connection: sqlite3.Connection | None = None,
) -> Dict[str, Any]:
    """Atomically authorize a resolved Shigure event and change presence."""

    if _connection is None:
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
    connection_context = (
        _get_connection() if _connection is None else nullcontext(_connection)
    )
    with connection_context as conn:
        if _connection is None:
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
                SET pose_aruco_json = COALESCE(?, pose_aruco_json),
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
            # All raw IDs for this display object are aliases of the same
            # physical object. A take-out invalidates the complete alias set;
            # a later bring-in may establish a fresh raw ID.
            conn.execute(
                f"""
                UPDATE {SHIGURE_OBJECT_BINDING_TABLE}
                SET status = 'REVOKED',
                    valid_until = ?,
                    revoke_reason = 'take_out_completed'
                WHERE source_epoch_id = ?
                  AND display_object_id = ?
                  AND status = 'ACTIVE'
                """,
                (
                    now,
                    str(binding["source_epoch_id"]),
                    display_object_id,
                ),
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


def _apply_object_lifecycle_event_conn(
    conn: sqlite3.Connection,
    *,
    canonical_event_uid: str,
    pose_aruco: Any = None,
    spatial_box_corners_aruco: Any = None,
    skeleton: Any = None,
    calibration_revision: str | None = None,
    occurred_at: str | None = None,
) -> Dict[str, Any]:
    """Apply lifecycle semantics inside an existing IMMEDIATE transaction."""

    return apply_object_lifecycle_event(
        canonical_event_uid=canonical_event_uid,
        pose_aruco=pose_aruco,
        spatial_box_corners_aruco=spatial_box_corners_aruco,
        skeleton=skeleton,
        calibration_revision=calibration_revision,
        occurred_at=occurred_at,
        _connection=conn,
    )


def commit_pending_take_out_lifecycle_event(
    event_uid: str,
    binding_id: str,
    display_object_id: str,
    raw_id: str,
    resolution_method: str,
    *,
    detail: Any = None,
    skeleton: Any = None,
    calibration_revision: str | None = None,
    occurred_at: str | None = None,
) -> Dict[str, Any]:
    """Resolve and apply one TAKE_OUT in a single rollback-safe transaction."""

    initialize_task_table()
    event_uid_value = str(event_uid or "").strip()
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute(
            f"SELECT action FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid_value,),
        ).fetchone()
        if event is None:
            raise ValueError("unknown canonical event")
        if str(event["action"]) != "TAKE_OUT":
            raise ValueError("canonical event is not TAKE_OUT")
        canonical = _resolve_pending_shigure_canonical_event_conn(
            conn,
            event_uid_value,
            binding_id,
            display_object_id,
            raw_id,
            resolution_method,
            detail,
        )
        lifecycle = _apply_object_lifecycle_event_conn(
            conn,
            canonical_event_uid=event_uid_value,
            skeleton=skeleton,
            calibration_revision=calibration_revision,
            occurred_at=occurred_at,
        )
        binding = conn.execute(
            f"SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE} WHERE binding_id = ?",
            (str(binding_id),),
        ).fetchone()
        if binding is None:
            raise RuntimeError("committed TAKE_OUT binding audit row is missing")
        return {
            "canonical_event": canonical,
            "lifecycle_event": lifecycle,
            "binding": dict(binding),
        }


def commit_pending_bring_in_lifecycle_event(
    event_uid: str,
    runtime_session_id: str,
    source_epoch_id: str,
    raw_id: str,
    display_object_id: str,
    resolution_method: str,
    *,
    confidence: float | None = None,
    binding_detail: Any = None,
    resolution_detail: Any = None,
    skeleton: Any = None,
    calibration_revision: str | None = None,
    occurred_at: str | None = None,
) -> Dict[str, Any]:
    """Establish, resolve, and apply one BRING_IN in one transaction."""

    initialize_task_table()
    event_uid_value = str(event_uid or "").strip()
    runtime_session_id_value = str(runtime_session_id or "").strip()
    source_epoch_id_value = str(source_epoch_id or "").strip()
    raw_id_value = str(raw_id or "").strip()
    display_object_id_value = str(display_object_id or "").strip()
    method_value = str(resolution_method or "").strip()
    if not all(
        (
            event_uid_value,
            runtime_session_id_value,
            source_epoch_id_value,
            raw_id_value,
            display_object_id_value,
            method_value,
        )
    ):
        raise ValueError("all BRING_IN transaction identity fields are required")

    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute(
            f"SELECT * FROM {SHIGURE_CANONICAL_EVENT_TABLE} WHERE event_uid = ?",
            (event_uid_value,),
        ).fetchone()
        if event is None:
            raise ValueError("unknown canonical event")
        if str(event["action"]) != "BRING_IN":
            raise ValueError("canonical event is not BRING_IN")
        if any(
            (
                str(event["runtime_session_id"]) != runtime_session_id_value,
                str(event["source_epoch_id"]) != source_epoch_id_value,
            )
        ):
            raise ValueError("BRING_IN event does not belong to the supplied epoch")

        if str(event["resolution_status"] or "").upper() == "RESOLVED":
            expected = (
                raw_id_value,
                display_object_id_value,
                method_value,
            )
            actual = (
                str(event["raw_shigure_object_id"] or ""),
                str(event["display_object_id"] or ""),
                str(event["resolution_method"] or ""),
            )
            if actual != expected:
                raise ValueError(
                    "canonical event is already resolved to a different identity"
                )
            lifecycle = conn.execute(
                f"""
                SELECT * FROM {OBJECT_LIFECYCLE_EVENT_TABLE}
                WHERE canonical_event_uid = ?
                """,
                (event_uid_value,),
            ).fetchone()
            if lifecycle is None:
                raise ValueError(
                    "resolved BRING_IN has no lifecycle row and cannot be replayed"
                )
            binding = conn.execute(
                f"SELECT * FROM {SHIGURE_OBJECT_BINDING_TABLE} WHERE binding_id = ?",
                (str(event["binding_id"] or ""),),
            ).fetchone()
            if binding is None:
                raise ValueError("resolved BRING_IN binding audit row is missing")
            lifecycle_result = dict(lifecycle)
            lifecycle_result["_replayed"] = True
            return {
                "canonical_event": dict(event),
                "lifecycle_event": lifecycle_result,
                "binding": dict(binding),
            }

        binding = _establish_shigure_binding_conn(
            conn,
            runtime_session_id=runtime_session_id_value,
            source_epoch_id=source_epoch_id_value,
            raw_shigure_object_id=raw_id_value,
            display_object_id=display_object_id_value,
            established_by=method_value,
            established_event_uid=event_uid_value,
            confidence=confidence,
            detail=binding_detail,
            allow_existing_same=False,
        )
        canonical = _resolve_pending_shigure_canonical_event_conn(
            conn,
            event_uid_value,
            str(binding["binding_id"]),
            display_object_id_value,
            raw_id_value,
            method_value,
            resolution_detail,
        )
        lifecycle = _apply_object_lifecycle_event_conn(
            conn,
            canonical_event_uid=event_uid_value,
            skeleton=skeleton,
            calibration_revision=calibration_revision,
            occurred_at=occurred_at,
        )
        return {
            "canonical_event": canonical,
            "lifecycle_event": lifecycle,
            "binding": binding,
        }


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


def _origin_pose_position(pose_aruco: Any) -> tuple[float, float, float]:
    if not isinstance(pose_aruco, dict):
        raise ValueError("pose_aruco must be an object")
    position = pose_aruco.get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("pose_aruco.position must contain exactly 3 values")
    try:
        values = tuple(float(component) for component in position)
    except (TypeError, ValueError) as exc:
        raise ValueError("pose_aruco.position must be numeric") from exc
    if not all(math.isfinite(component) for component in values):
        raise ValueError("pose_aruco.position must be finite")
    return values  # type: ignore[return-value]


def add_display_object_origin(
    display_object_id: str,
    model_revision: int,
    pose_aruco: Any,
    kind: str,
    source_epoch_id: str | None = None,
    canonical_event_uid: str | None = None,
    binding_id: str | None = None,
    raw_shigure_object_id: str | None = None,
    occurred_at: str | None = None,
    dedup_distance_m: float | None = 0.2,
) -> Dict[str, Any]:
    """Persist one origin, idempotently, and retain at most five per object."""

    initialize_task_table()
    display_object_id = str(display_object_id or "").strip()
    if not display_object_id:
        raise ValueError("display_object_id is required")
    kind_value = str(kind or "").strip().upper()
    if kind_value not in {"INITIALIZATION", "TAKE_OUT"}:
        raise ValueError("kind must be INITIALIZATION or TAKE_OUT")
    model_revision = int(model_revision)
    if model_revision < 0:
        raise ValueError("model_revision cannot be negative")
    position = _origin_pose_position(pose_aruco)
    if dedup_distance_m is not None:
        dedup_distance_m = float(dedup_distance_m)
        if not math.isfinite(dedup_distance_m) or dedup_distance_m < 0.0:
            raise ValueError("dedup_distance_m must be finite and non-negative")

    source_epoch_value = str(source_epoch_id or "").strip() or None
    canonical_event_value = str(canonical_event_uid or "").strip() or None
    binding_value = str(binding_id or "").strip() or None
    raw_id_value = str(raw_shigure_object_id or "").strip() or None
    event_time = str(occurred_at or _utc_now_text())
    if canonical_event_value:
        identity_key = f"canonical:{canonical_event_value}"
    elif binding_value or source_epoch_value or raw_id_value:
        identity_key = "|".join(
            (
                kind_value,
                source_epoch_value or "",
                binding_value or "",
                raw_id_value or "",
                display_object_id,
                str(model_revision),
            )
        )
    else:
        identity_key = "|".join(
            (
                kind_value,
                display_object_id,
                str(model_revision),
                event_time,
                json.dumps(pose_aruco, ensure_ascii=False, sort_keys=True),
            )
        )
    origin_uid = uuid.uuid5(
        uuid.NAMESPACE_URL, f"display-object-origin:{identity_key}"
    ).hex

    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            f"""
            SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
            WHERE origin_uid = ?
            """,
            (origin_uid,),
        ).fetchone()
        if existing is not None:
            result = dict(existing)
            result["_inserted"] = False
            result["_deduplicated"] = False
            return result

        processed_canonical_uids: list[str] = []
        if canonical_event_value:
            origin_rows = conn.execute(
                f"""
                SELECT *
                FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                WHERE display_object_id = ?
                ORDER BY occurred_at DESC, id DESC
                """,
                (display_object_id,),
            ).fetchall()
            for origin_row in origin_rows:
                stored_canonical_uid = str(
                    origin_row["canonical_event_uid"] or ""
                ).strip()
                if stored_canonical_uid:
                    processed_canonical_uids.append(stored_canonical_uid)
                if stored_canonical_uid == canonical_event_value:
                    result = dict(origin_row)
                    result["_inserted"] = False
                    result["_deduplicated"] = True
                    return result
                try:
                    origin_detail = json.loads(str(origin_row["detail_json"] or "{}"))
                except (TypeError, json.JSONDecodeError):
                    continue
                deduplicated_uids = (
                    origin_detail.get("processed_canonical_event_uids")
                    if isinstance(origin_detail, dict)
                    else None
                )
                if not isinstance(deduplicated_uids, list) and isinstance(
                    origin_detail, dict
                ):
                    deduplicated_uids = origin_detail.get(
                        "deduplicated_canonical_event_uids"
                    )
                if isinstance(deduplicated_uids, list):
                    processed_canonical_uids.extend(
                        str(value)
                        for value in deduplicated_uids
                        if str(value).strip()
                    )
                if (
                    isinstance(deduplicated_uids, list)
                    and canonical_event_value in deduplicated_uids
                ):
                    result = dict(origin_row)
                    result["_inserted"] = False
                    result["_deduplicated"] = True
                    return result

        state = conn.execute(
            f"""
            SELECT display_object_id
            FROM {DISPLAY_OBJECT_STATE_TABLE}
            WHERE display_object_id = ?
            """,
            (display_object_id,),
        ).fetchone()
        if state is None:
            raise ValueError("display object has no model/capture state")

        latest = conn.execute(
            f"""
            SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
            WHERE display_object_id = ?
            ORDER BY occurred_at DESC, id DESC
            LIMIT 1
            """,
            (display_object_id,),
        ).fetchone()
        if latest is not None:
            try:
                latest_detail = json.loads(str(latest["detail_json"] or "{}"))
            except (TypeError, json.JSONDecodeError):
                latest_detail = {}
            if isinstance(latest_detail, dict):
                carried_uids = latest_detail.get("processed_canonical_event_uids")
                if not isinstance(carried_uids, list):
                    carried_uids = latest_detail.get(
                        "deduplicated_canonical_event_uids"
                    )
                if isinstance(carried_uids, list):
                    processed_canonical_uids.extend(
                        str(value)
                        for value in carried_uids
                        if str(value).strip()
                    )
            latest_canonical_uid = str(
                latest["canonical_event_uid"] or ""
            ).strip()
            if latest_canonical_uid:
                processed_canonical_uids.append(latest_canonical_uid)
        processed_canonical_uids = list(
            dict.fromkeys(processed_canonical_uids)
        )[-256:]
        if dedup_distance_m is not None:
            previous_origin = conn.execute(
                f"""
                SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                WHERE display_object_id = ? AND occurred_at <= ?
                ORDER BY occurred_at DESC, id DESC
                LIMIT 1
                """,
                (display_object_id, event_time),
            ).fetchone()
            next_origin = conn.execute(
                f"""
                SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                WHERE display_object_id = ? AND occurred_at > ?
                ORDER BY occurred_at ASC, id ASC
                LIMIT 1
                """,
                (display_object_id, event_time),
            ).fetchone()
            for adjacent in (previous_origin, next_origin):
                if adjacent is None:
                    continue
                try:
                    adjacent_pose = json.loads(str(adjacent["pose_aruco_json"]))
                    adjacent_position = _origin_pose_position(adjacent_pose)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                distance = math.sqrt(
                    sum(
                        (position[index] - adjacent_position[index]) ** 2
                        for index in range(3)
                    )
                )
                if distance < dedup_distance_m:
                    if canonical_event_value:
                        try:
                            detail = json.loads(str(adjacent["detail_json"] or "{}"))
                        except (TypeError, json.JSONDecodeError):
                            detail = {}
                        if not isinstance(detail, dict):
                            detail = {}
                        deduplicated_uids = detail.get(
                            "processed_canonical_event_uids"
                        )
                        if not isinstance(deduplicated_uids, list):
                            deduplicated_uids = processed_canonical_uids
                        if canonical_event_value not in deduplicated_uids:
                            deduplicated_uids.append(canonical_event_value)
                            detail["processed_canonical_event_uids"] = list(
                                dict.fromkeys(deduplicated_uids)
                            )[-256:]
                            serialized_detail = json.dumps(detail, ensure_ascii=False)
                            conn.execute(
                                f"""
                                UPDATE {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                                SET detail_json = ?
                                WHERE id = ?
                                """,
                                (serialized_detail, int(adjacent["id"])),
                            )
                        else:
                            serialized_detail = str(adjacent["detail_json"] or "{}")
                    result = dict(adjacent)
                    if canonical_event_value:
                        result["detail_json"] = serialized_detail
                    result["_inserted"] = False
                    result["_deduplicated"] = True
                    result["_dedup_distance_m"] = distance
                    return result

        if canonical_event_value:
            processed_canonical_uids.append(canonical_event_value)
        processed_canonical_uids = list(
            dict.fromkeys(processed_canonical_uids)
        )[-256:]
        origin_detail = (
            {"processed_canonical_event_uids": processed_canonical_uids}
            if processed_canonical_uids
            else {}
        )
        conn.execute(
            f"""
            INSERT INTO {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} (
                origin_uid, display_object_id, kind, model_revision,
                pose_aruco_json, source_epoch_id, canonical_event_uid,
                binding_id, raw_shigure_object_id, occurred_at, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                origin_uid,
                display_object_id,
                kind_value,
                model_revision,
                json.dumps(pose_aruco, ensure_ascii=False),
                source_epoch_value,
                canonical_event_value,
                binding_value,
                raw_id_value,
                event_time,
                json.dumps(origin_detail, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE} WHERE origin_uid = ?",
            (origin_uid,),
        ).fetchone()
        conn.execute(
            f"""
            DELETE FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
            WHERE display_object_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                  WHERE display_object_id = ?
                  ORDER BY occurred_at DESC, id DESC
                  LIMIT ?
              )
            """,
            (
                display_object_id,
                display_object_id,
                DISPLAY_OBJECT_ORIGIN_HISTORY_LIMIT,
            ),
        )
    result = dict(row)
    result["_inserted"] = True
    result["_deduplicated"] = False
    return result


def list_display_object_origins(
    display_object_id: str,
    *,
    limit: int = DISPLAY_OBJECT_ORIGIN_HISTORY_LIMIT,
    before_id: int | None = None,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    limit = max(1, min(int(limit), DISPLAY_OBJECT_ORIGIN_HISTORY_LIMIT))
    display_object_id = str(display_object_id)
    with _get_connection() as conn:
        where = ["display_object_id = ?"]
        values: list[Any] = [display_object_id]
        if before_id is not None:
            cursor = conn.execute(
                f"""
                SELECT occurred_at, id
                FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
                WHERE display_object_id = ? AND id = ?
                """,
                (display_object_id, int(before_id)),
            ).fetchone()
            if cursor is None:
                return []
            where.append("(occurred_at < ? OR (occurred_at = ? AND id < ?))")
            values.extend(
                (
                    str(cursor["occurred_at"]),
                    str(cursor["occurred_at"]),
                    int(cursor["id"]),
                )
            )
        values.append(limit)
        rows = conn.execute(
            f"""
            SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY occurred_at DESC, id DESC
            LIMIT ?
            """,
            tuple(values),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_display_object_origin(
    display_object_id: str,
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM {DISPLAY_OBJECT_ORIGIN_HISTORY_TABLE}
            WHERE display_object_id = ?
            ORDER BY occurred_at DESC, id DESC
            LIMIT 1
            """,
            (str(display_object_id),),
        ).fetchone()
    return _row_to_dict(row)


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
    if source_value != "HOLOLENS":
        if source_value != "SHIGURE":
            raise ValueError("source must be HOLOLENS")
        raise ValueError(
            "Shigure masked images are query-only; identity references must come from HoloLens uploads"
        )

    reference_id = uuid.uuid4().hex
    with _get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        created_at = _utc_now_text()
        conn.execute(
            f"""
            INSERT INTO {OBJECT_IDENTITY_REFERENCE_TABLE} (
                reference_id, display_object_id, source, source_event_uid,
                source_task_id, image_path, mask_path, embedding_path,
                view_hash, quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(display_object_id, view_hash) DO UPDATE SET
                active = 1,
                source = excluded.source,
                source_event_uid = excluded.source_event_uid,
                source_task_id = excluded.source_task_id,
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
                created_at,
            ),
        )
        _cap_hololens_identity_references(conn, str(display_object_id))
        row = conn.execute(
            f"""
            SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
            WHERE display_object_id = ? AND view_hash = ?
            """,
            (str(display_object_id), str(view_hash)),
        ).fetchone()
    return dict(row)


def list_object_identity_references(
    display_object_id: str,
    *,
    limit: int = 20,
    source: str | None = None,
) -> List[Dict[str, Any]]:
    initialize_task_table()
    source_value = None
    if source is not None:
        source_value = str(source or "").strip().upper()
        if source_value not in {"SHIGURE", "HOLOLENS"}:
            raise ValueError("source must be SHIGURE or HOLOLENS")
    where = ["display_object_id = ?", "active = 1"]
    values: list[Any] = [str(display_object_id)]
    if source_value is not None:
        where.append("source = ?")
        values.append(source_value)
    values.append(max(1, min(int(limit), 100)))
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM {OBJECT_IDENTITY_REFERENCE_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            tuple(values),
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
                candidate_limit = excluded.candidate_limit,
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
                max(
                    1,
                    min(
                        int(candidate_limit),
                        SHIGURE_IDENTITY_MAX_DISPLAY_OBJECTS,
                    ),
                ),
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
