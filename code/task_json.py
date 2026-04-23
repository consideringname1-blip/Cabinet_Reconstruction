from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from config import PROJECT_ROOT, UPLOAD_FOLDER


_PROJECT_PATH_HINTS = ("data/", "code/", "H2AI/", "models/")


def _rewrite_legacy_storage_path(normalized: str) -> str:
    return (
        normalized.replace("data/aruco/raw/", "data/aruco/runtime/")
        .replace("data/aruco/aruco_template.json", "data/aruco/reference/aruco.json")
    )


def _iter_candidate_paths(
    path_arg: str | Path,
    *,
    default_base: Path | None = None,
):
    raw = str(path_arg or "").strip()
    if not raw:
        return

    normalized = raw.replace("\\", "/")
    seen: set[str] = set()

    def emit(candidate: Path):
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        yield candidate

    candidate = Path(raw).expanduser()
    yield from emit(candidate)

    if default_base is not None and not candidate.is_absolute():
        yield from emit(default_base / candidate)

    if normalized.startswith("/workspace/"):
        yield from emit(PROJECT_ROOT / Path(normalized.removeprefix("/workspace/")))
    elif normalized.startswith("workspace/"):
        yield from emit(PROJECT_ROOT / Path(normalized.removeprefix("workspace/")))

    normalized_lstrip = normalized.lstrip("./")
    for hint in _PROJECT_PATH_HINTS:
        marker_index = normalized_lstrip.lower().find(hint.lower())
        if marker_index == -1:
            continue
        yield from emit(PROJECT_ROOT / Path(normalized_lstrip[marker_index:]))
        break

    rewritten = _rewrite_legacy_storage_path(normalized)
    if rewritten == normalized:
        return

    if rewritten.startswith("/workspace/"):
        yield from emit(PROJECT_ROOT / Path(rewritten.removeprefix("/workspace/")))
        return
    if rewritten.startswith("workspace/"):
        yield from emit(PROJECT_ROOT / Path(rewritten.removeprefix("workspace/")))
        return

    rewritten_lstrip = rewritten.lstrip("./")
    for hint in _PROJECT_PATH_HINTS:
        marker_index = rewritten_lstrip.lower().find(hint.lower())
        if marker_index == -1:
            continue
        yield from emit(PROJECT_ROOT / Path(rewritten_lstrip[marker_index:]))
        break


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
    return resolve_project_path(json_arg, default_base=UPLOAD_FOLDER, require_exists=True)


def load_task_json(json_path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_task_json_path(json_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_task_json(json_path: str | Path, data: Mapping[str, Any]) -> None:
    resolved_path = resolve_project_path(json_path, default_base=UPLOAD_FOLDER, require_exists=False)
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
