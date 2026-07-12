from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from path_config import PROJECT_ROOT


def resolve_project_path(
    path_arg: str | Path,
    *,
    require_exists: bool = True,
) -> Path:
    raw = str(path_arg or "").strip()
    if not raw:
        raise ValueError("path is empty")
    path = Path(raw).expanduser()
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if require_exists and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {path_arg}")
    return resolved


def normalize_path_for_storage(
    path_arg: str | Path,
) -> str:
    resolved = resolve_project_path(path_arg, require_exists=False)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(Path(path_arg).expanduser())


def resolve_task_json_path(json_arg: str | Path) -> Path:
    return resolve_project_path(json_arg, require_exists=True)


def resolve_task_json_path_from_record(task_record: Mapping[str, Any]) -> Path:
    json_path = str(task_record.get("json_path") or "").strip()
    if not json_path:
        raise ValueError("task record is missing json_path")
    return resolve_task_json_path(json_path)


def load_task_json(json_path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_task_json_path(json_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_task_json(json_path: str | Path, data: Mapping[str, Any]) -> None:
    resolved_path = resolve_project_path(json_path, require_exists=False)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_task_id_in_json(json_path: str | Path, task_id: str) -> None:
    data = load_task_json(json_path)
    if data.get("task_id") == task_id:
        return

    data["task_id"] = task_id
    save_task_json(json_path, data)
