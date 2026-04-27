import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATABASE_PATH, UPLOAD_FOLDER
from task_json import normalize_path_for_storage


TABLE_NAME = "tasks"
ARUCO_REFERENCE_TABLE = "aruco_references"
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
    "blender",
    "completed",
    "aruco_completed",
    "failed",
)
TERMINAL_STATUSES = ("completed", "aruco_completed", "failed")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


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


def initialize_task_table() -> None:
    with _get_connection() as conn:
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

        _normalize_stored_paths(conn)
        conn.commit()


def get_latest_10_records() -> List[Dict[str, Any]]:
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT 10").fetchall()
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
) -> Optional[Dict[str, Any]]:
    initialize_task_table()
    startup_session_id = str(startup_session_id or "").strip()
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
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
    return _row_to_dict(row)


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
