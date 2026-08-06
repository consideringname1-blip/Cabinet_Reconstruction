"""Configuration loading and resolved-config recording."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .errors import Failure, Phase1Error


REQUIRED_TOP_LEVEL = {
    "schema_version", "dataset", "reference_selection", "validity",
    "proposals", "tracking", "clustering", "classification",
    "known_motion", "visualization", "runtime",
}


def _resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, base, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, base, key) for v in value]
    if isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir") or key.endswith("_root")):
        path = Path(value).expanduser()
        return str((base / path).resolve() if not path.is_absolute() else path.resolve())
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase1Error(Failure("config", "yaml_read_failed", str(exc), details={"path": str(path)})) from exc
    if not isinstance(raw, dict):
        raise Phase1Error(Failure("config", "yaml_root_not_mapping", "YAML root must be a mapping"))
    missing = sorted(REQUIRED_TOP_LEVEL - set(raw))
    if missing:
        raise Phase1Error(Failure("config", "missing_sections", "Required YAML sections are missing", details={"missing": missing}))
    config = _resolve_paths(copy.deepcopy(raw), path.parent.resolve())
    if str(config["schema_version"]) != "1.0":
        raise Phase1Error(Failure("config", "unsupported_schema", "Only schema_version 1.0 is supported"))
    if config["known_motion"].get("type") not in {"prismatic", "revolute"}:
        raise Phase1Error(Failure("config", "invalid_known_motion", "known_motion.type must be prismatic or revolute"))
    forbidden = {"optimize_camera", "estimate_axis", "model_selection", "free_se3", "tsdf", "nksr", "mesh"}
    present = sorted(forbidden & set(config))
    if present:
        raise Phase1Error(Failure("config", "forbidden_phase1_sections", "Config requests work outside stage 1", details={"sections": present}))
    config["_config_source"] = str(path.resolve())
    return config


def save_resolved_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(yaml.safe_dump(copy.deepcopy(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
