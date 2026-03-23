from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from config import UPLOAD_FOLDER


def resolve_task_json_path(json_arg: str | Path) -> Path:
    candidate = Path(json_arg).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    upload_candidate = (UPLOAD_FOLDER / candidate).resolve()
    if upload_candidate.is_file():
        return upload_candidate

    raise FileNotFoundError(f"JSON file not found: {json_arg}")


def load_task_json(json_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(json_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_task_json(json_path: str | Path, data: Mapping[str, Any]) -> None:
    resolved_path = Path(json_path).expanduser().resolve()
    with resolved_path.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_task_id_in_json(json_path: str | Path, task_id: str) -> None:
    data = load_task_json(json_path)
    if data.get("task_id") == task_id:
        return

    data["task_id"] = task_id
    save_task_json(json_path, data)
