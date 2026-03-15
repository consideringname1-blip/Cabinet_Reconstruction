import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATABASE_PATH


TABLE_NAME = "tasks"
ALLOWED_STATUSES = (
    "pending",
    "hololens2depth",
    "sam3mask",
    "instantmesh",
    "relocationresize",
    "blender",
    "completed",
    "failed",
)


def _get_connection() -> sqlite3.Connection:
    """获取数据库连接，并确保数据库目录存在。"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """将 sqlite 查询结果行转换为字典。"""
    if row is None:
        return None
    return dict(row)


def initialize_task_table() -> None:
    """初始化任务表；如果表不存在则自动创建。"""
    status_list = ", ".join(f"'{status}'" for status in ALLOWED_STATUSES)
    with _get_connection() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ({status_list})),
                json_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
            """
        )
        conn.commit()


def get_latest_10_records() -> List[Dict[str, Any]]:
    """按主键 id 倒序查询最新的 10 条任务记录。"""
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT 10"
        ).fetchall()
    return [dict(row) for row in rows]


def get_status_by_task_id(task_id: str) -> Optional[str]:
    """根据 task_id 查询任务当前状态。"""
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT status FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return row["status"] if row else None


def get_json_path_by_task_id(task_id: str) -> Optional[str]:
    """根据 task_id 查询对应的 json_path。"""
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT json_path FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return row["json_path"] if row else None


def create_task(task_id: str, json_path: Path | str) -> Dict[str, Any]:
    """使用 task_id 和 json_path 创建一条新的任务记录。"""
    initialize_task_table()
    json_path_str = str(json_path)
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (task_id, status, json_path)
            VALUES (?, 'pending', ?)
            """,
            (task_id, json_path_str),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return dict(row)


def get_latest_unfinished_task() -> Optional[Dict[str, Any]]:
    """查询最新一条未完成且未失败的任务。"""
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"""
            SELECT task_id, json_path, status
            FROM {TABLE_NAME}
            WHERE status NOT IN ('completed', 'failed')
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    return _row_to_dict(row)


def get_unfinished_tasks() -> List[Dict[str, Any]]:
    """按创建顺序查询全部未完成且未失败的任务。"""
    initialize_task_table()
    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE status NOT IN ('completed', 'failed')
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def update_task_status(
    task_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    """根据 task_id 更新任务状态，并按需要写入错误信息与时间戳。"""
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

    if status == "completed":
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


def get_task_by_task_id(task_id: str) -> Optional[Dict[str, Any]]:
    """根据 task_id 查询整条任务记录。"""
    initialize_task_table()
    with _get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return _row_to_dict(row)
