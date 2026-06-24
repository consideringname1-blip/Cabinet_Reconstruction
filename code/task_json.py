from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from path_config import PROJECT_ROOT
from artifact_layout import model_task_json_path


_PROJECT_PATH_HINTS = ("data/", "code/", "H2AI/", "models/")


def resolve_project_path(
    path_arg: str | Path,
    *,
    default_base: Path | None = None,
    require_exists: bool = True,
) -> Path:
    if not str(path_arg or "").strip():
        raise ValueError("path is empty")

    fallback: Path | None = None
    project_fallback: Path | None = None
    for candidate in _iter_candidate_paths(path_arg, default_base=default_base) or []:
        resolved = candidate.expanduser().resolve()
        fallback = fallback or resolved
        if resolved.exists():
            return resolved
        try:
            resolved.relative_to(PROJECT_ROOT)
            project_fallback = project_fallback or resolved
        except ValueError:
            pass

    if require_exists:
        raise FileNotFoundError(f"Path not found: {path_arg}")
    if project_fallback is not None:
        return project_fallback
    if fallback is None:
        raise ValueError("path is empty")
    return fallback


def normalize_path_for_storage(
    path_arg: str | Path,
    *,
    default_base: Path | None = None,
) -> str:
    resolved = resolve_project_path(path_arg, default_base=default_base, require_exists=False)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(Path(path_arg).expanduser())


def resolve_task_json_path(json_arg: str | Path) -> Path:
    return resolve_project_path(json_arg, require_exists=True)


def resolve_task_json_path_from_record(task_record: Mapping[str, Any]) -> Path:
    json_path = str(task_record.get("json_path") or "").strip()
    if json_path:
        return resolve_task_json_path(json_path)

    task_timestamp = str(task_record.get("task_timestamp") or "").strip()
    if task_timestamp:
        candidate = model_task_json_path(task_timestamp).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Task JSON not found for task_timestamp={task_timestamp!r}")


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
