from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DATABASE_RELATIVE_PATH = Path("database/tasks.db")
BACKUP_SUFFIX = ".pre_shigure_v2"
BACKUP_METADATA_SUFFIX = ".meta.json"
TARGET_SCHEMA_NAME = "shigure_runtime"
TARGET_SCHEMA_VERSION = 2

LEGACY_TABLES = (
    "realtime_tracking_events",
    "auxiliary_jobs",
)

# These tables contain data whose identity/coordinate meaning remains valid in
# v2.  Their row counts are checked before and after task_db's schema migration.
PRESERVED_TABLES = (
    "tasks",
    "task_stage_runs",
    "task_timing_events",
    "ai_model_timings",
    "aruco_references",
    "aruco_markers",
    "aruco_marker_relations",
    "model_bounds",
    "display_objects",
    "capture_instances",
    "capture_binding_logs",
    "display_object_states",
    "display_object_model_revisions",
    "display_object_pose_history",
)

PRESERVED_FILESYSTEM_ROOTS = (
    Path("model"),
    Path("aruco"),
    Path("aruco_processing"),
    Path("identity_references"),
)


class MigrationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _quoted_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
    }


def _row_counts(path: Path, table_names: Iterable[str]) -> dict[str, int | None]:
    with _read_only_connection(path) as connection:
        existing = _table_names(connection)
        return {
            table_name: (
                int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {_quoted_identifier(table_name)}"
                    ).fetchone()[0]
                )
                if table_name in existing
                else None
            )
            for table_name in table_names
        }


def _schema_version(path: Path) -> int | None:
    with _read_only_connection(path) as connection:
        if "schema_metadata" not in _table_names(connection):
            return None
        row = connection.execute(
            """
            SELECT schema_version
            FROM schema_metadata
            WHERE schema_name = ?
            """,
            (TARGET_SCHEMA_NAME,),
        ).fetchone()
        return int(row[0]) if row is not None else None


def _integrity_check(path: Path) -> None:
    try:
        with _read_only_connection(path) as connection:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise MigrationError(f"SQLite integrity check failed for {path}: {exc}") from exc
    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        raise MigrationError(
            f"SQLite integrity check failed for {path}: {'; '.join(messages)}"
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _encoded_sqlite_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, bytes):
        return {"blob": value.hex()}
    return {"type": type(value).__name__, "text": str(value)}


def _logical_database_fingerprint(path: Path) -> str:
    """Hash SQLite schema and row content, independent of page/WAL layout."""

    digest = hashlib.sha256()
    with _read_only_connection(path) as connection:
        tables = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        for table in tables:
            name = str(table["name"])
            schema_record = ["schema", name, str(table["sql"] or "")]
            digest.update(
                json.dumps(schema_record, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            quoted = _quoted_identifier(name)
            # SQLite's backup API preserves rowids. Ordering by rowid makes the
            # comparison stable even when page layout differs.
            try:
                rows = connection.execute(f"SELECT * FROM {quoted} ORDER BY rowid")
            except sqlite3.OperationalError:
                rows = connection.execute(f"SELECT * FROM {quoted}")
            for row in rows:
                encoded = [_encoded_sqlite_value(value) for value in tuple(row)]
                digest.update(
                    json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
    return digest.hexdigest()


def _backup_path(database_path: Path) -> Path:
    return database_path.with_name(database_path.name + BACKUP_SUFFIX)


def _backup_metadata_path(backup_path: Path) -> Path:
    return backup_path.with_name(backup_path.name + BACKUP_METADATA_SUFFIX)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve_preserved_artifact(raw_path: Any, data_root: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    raw = Path(text).expanduser()
    candidate = raw if raw.is_absolute() else data_root.parent / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root.resolve())
    except (FileNotFoundError, ValueError, OSError):
        return None
    return resolved if resolved.is_file() else None


def _stored_project_path(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.parent.resolve()).as_posix()
    except ValueError as exc:
        raise MigrationError(
            f"Identity artifact is outside the project data root: {path}"
        ) from exc


def _capture_identity_migration_plan(
    database_path: Path,
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract only self-contained, verifiable HoloLens DINO references."""

    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with _read_only_connection(database_path) as connection:
        if "capture_instances" not in _table_names(connection):
            return planned, skipped
        rows = connection.execute(
            """
            SELECT task_id, display_object_id, feature_json, evidence_json, created_at
            FROM capture_instances
            WHERE binding_status = 'bound'
              AND display_object_id IS NOT NULL
              AND TRIM(display_object_id) <> ''
            ORDER BY id
            """
        ).fetchall()

    for row in rows:
        task_id = str(row["task_id"] or "").strip()
        display_object_id = str(row["display_object_id"] or "").strip()
        reason: str | None = None
        try:
            feature = json.loads(str(row["feature_json"] or "{}"))
            evidence = json.loads(str(row["evidence_json"] or "{}"))
        except json.JSONDecodeError:
            feature, evidence = {}, {}
            reason = "invalid_identity_json"
        dino = feature.get("dinov2") if isinstance(feature, dict) else None
        source = evidence.get("dinov2_source") if isinstance(evidence, dict) else None
        if not isinstance(dino, dict) or not isinstance(source, dict):
            reason = reason or "dinov2_reference_missing"
            dino, source = {}, {}
        raw_embedding = dino.get("embedding")
        try:
            embedding = [float(value) for value in raw_embedding]
        except (TypeError, ValueError, OverflowError):
            embedding = []
        norm = math.sqrt(sum(value * value for value in embedding)) if embedding else 0.0
        if (
            not embedding
            or not all(math.isfinite(value) for value in embedding)
            or not math.isfinite(norm)
            or norm <= 1.0e-12
        ):
            reason = reason or "dinov2_embedding_invalid"
        image_path = _resolve_preserved_artifact(source.get("color_path"), data_root)
        mask_path = _resolve_preserved_artifact(source.get("mask_path"), data_root)
        if image_path is None or mask_path is None:
            reason = reason or "identity_image_or_mask_missing"
        if reason:
            skipped.append(
                {
                    "task_id": task_id,
                    "display_object_id": display_object_id,
                    "reason": reason,
                }
            )
            continue

        reference_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"shigure-v2-hololens-reference:{display_object_id}:{task_id}",
        ).hex
        embedding_path = (
            data_root / "identity_references" / reference_id / "embedding.json"
        )
        embedding_payload = dict(dino)
        embedding_payload["embedding"] = embedding
        embedding_payload["dim"] = len(embedding)
        embedding_payload["normalized"] = True
        planned.append(
            {
                "reference_id": reference_id,
                "display_object_id": display_object_id,
                "task_id": task_id,
                "image_path": _stored_project_path(image_path, data_root),
                "mask_path": _stored_project_path(mask_path, data_root),
                "embedding_path": _stored_project_path(embedding_path, data_root),
                "embedding_file": embedding_path,
                "embedding_payload": embedding_payload,
                "view_hash": f"hololens:{task_id}",
                "created_at": str(row["created_at"] or _utc_now()),
            }
        )
    return planned, skipped


def _materialize_identity_references(
    database_path: Path,
    entries: Sequence[dict[str, Any]],
) -> int:
    for entry in entries:
        embedding_path = Path(entry["embedding_file"])
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(embedding_path, dict(entry["embedding_payload"]))

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for entry in entries:
            connection.execute(
                """
                INSERT INTO object_identity_references (
                    reference_id, display_object_id, source, source_task_id,
                    image_path, mask_path, embedding_path, view_hash,
                    active, quality_json, created_at
                ) VALUES (?, ?, 'HOLOLENS', ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(display_object_id, view_hash) DO UPDATE SET
                    active = 1,
                    image_path = excluded.image_path,
                    mask_path = excluded.mask_path,
                    embedding_path = excluded.embedding_path,
                    quality_json = excluded.quality_json
                """,
                (
                    entry["reference_id"],
                    entry["display_object_id"],
                    entry["task_id"],
                    entry["image_path"],
                    entry["mask_path"],
                    entry["embedding_path"],
                    entry["view_hash"],
                    json.dumps(
                        {
                            "role": "auxiliary",
                            "migration": "shigure_v2",
                            "embedding_dim": len(
                                entry["embedding_payload"]["embedding"]
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    entry["created_at"],
                ),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return len(entries)


def _task_json_path(raw_path: Any, data_root: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    raw = Path(text).expanduser()
    candidate = raw if raw.is_absolute() else data_root.parent / raw
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root.resolve())
    except (FileNotFoundError, ValueError, OSError):
        return None
    return resolved if resolved.is_file() else None


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _canonical_pose(value: Any) -> dict[str, list[float]] | None:
    if not isinstance(value, dict):
        return None
    position = _finite_vector(value.get("position"), 3)
    rotation = _finite_vector(value.get("rotation_quaternion_xyzw"), 4)
    scale = _finite_vector(value.get("scale"), 3)
    if position is None or rotation is None or scale is None:
        return None
    norm = math.sqrt(sum(component * component for component in rotation))
    if norm <= 1.0e-12 or any(component <= 0.0 for component in scale):
        return None
    return {
        "position": position,
        "rotation_quaternion_xyzw": [component / norm for component in rotation],
        "scale": scale,
    }


def _capture_state_migration_plan(
    database_path: Path,
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan deterministic state reconstruction only for objects without state."""

    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with _read_only_connection(database_path) as connection:
        tables = _table_names(connection)
        if not {"capture_instances", "tasks", "display_object_states"}.issubset(tables):
            return planned, skipped
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT display_object_id FROM display_object_states"
            ).fetchall()
        }
        rows = connection.execute(
            """
            SELECT ci.id, ci.task_id, ci.display_object_id, ci.timestamp,
                   ci.created_at, t.json_path, t.status
            FROM capture_instances AS ci
            JOIN tasks AS t ON t.task_id = ci.task_id
            WHERE ci.binding_status = 'bound'
              AND ci.display_object_id IS NOT NULL
              AND TRIM(ci.display_object_id) <> ''
            ORDER BY ci.id
            """
        ).fetchall()

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        display_object_id = str(row["display_object_id"] or "").strip()
        if display_object_id not in existing:
            grouped.setdefault(display_object_id, []).append(row)

    generated_decisions = {"create_new", "bind_historical_dinov2_match"}
    reuse_decisions = {"reuse_historical_model_match"}
    for display_object_id, captures in grouped.items():
        model_revision = 0
        pose_revision = 0
        active_model_task_id: str | None = None
        object_entries: list[dict[str, Any]] = []
        for row in captures:
            task_id = str(row["task_id"] or "").strip()
            task_path = _task_json_path(row["json_path"], data_root)
            if str(row["status"] or "") != "completed" or task_path is None:
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "completed_task_json_missing"}
                )
                continue
            try:
                task_json = json.loads(task_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "invalid_task_json"}
                )
                continue
            identity = task_json.get("DisplayIdentity")
            if not isinstance(identity, dict) or str(
                identity.get("display_object_id") or ""
            ).strip() != display_object_id:
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "display_identity_mismatch"}
                )
                continue
            decision = str(identity.get("decision") or "").strip()
            generated = decision in generated_decisions
            if not generated and decision not in reuse_decisions:
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "unsupported_identity_decision"}
                )
                continue
            if generated:
                blender = task_json.get("Blender")
                fbx_name = str((blender or {}).get("fbx") or "").strip()
                fbx_path = task_path.parent / "result" / fbx_name
                if (
                    not isinstance(blender, dict)
                    or blender.get("artifact_root") != "model_result"
                    or not fbx_name
                    or not fbx_path.is_file()
                ):
                    skipped.append(
                        {"task_id": task_id, "display_object_id": display_object_id,
                         "reason": "generated_model_artifact_missing"}
                    )
                    continue
                model_revision += 1
                active_model_task_id = task_id
            elif model_revision <= 0 or active_model_task_id is None:
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "reuse_without_prior_model"}
                )
                continue

            pose = _canonical_pose(task_json.get("object_aruco"))
            if pose is None:
                skipped.append(
                    {"task_id": task_id, "display_object_id": display_object_id,
                     "reason": "valid_aruco_pose_missing"}
                )
                object_entries.append(
                    {
                        "task_id": task_id,
                        "generated_new_model": generated,
                        "model_revision": model_revision,
                        "active_model_task_id": active_model_task_id,
                        "pose": None,
                        "pose_revision": None,
                        "captured_at": str(row["timestamp"] or row["created_at"] or _utc_now()),
                    }
                )
                continue
            pose_revision += 1
            object_entries.append(
                {
                    "task_id": task_id,
                    "generated_new_model": generated,
                    "model_revision": model_revision,
                    "active_model_task_id": active_model_task_id,
                    "pose": pose,
                    "pose_revision": pose_revision,
                    "captured_at": str(row["timestamp"] or row["created_at"] or _utc_now()),
                }
            )
        if object_entries and model_revision > 0:
            planned.append(
                {
                    "display_object_id": display_object_id,
                    "entries": object_entries,
                    "active_model_revision": model_revision,
                    "active_model_task_id": active_model_task_id,
                    "latest_pose_revision": pose_revision,
                }
            )
    return planned, skipped


def _materialize_capture_states(
    database_path: Path,
    objects: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts = {"objects": 0, "model_revisions": 0, "pose_history": 0}
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    now = _utc_now()
    try:
        connection.execute("BEGIN IMMEDIATE")
        for item in objects:
            display_object_id = str(item["display_object_id"])
            existing = connection.execute(
                "SELECT 1 FROM display_object_states WHERE display_object_id = ?",
                (display_object_id,),
            ).fetchone()
            if existing is not None:
                continue
            connection.execute(
                "INSERT INTO display_object_states (display_object_id, updated_at) VALUES (?, ?)",
                (display_object_id, now),
            )
            counts["objects"] += 1
            latest_pose: dict[str, Any] | None = None
            for entry in item["entries"]:
                if entry["generated_new_model"]:
                    connection.execute(
                        """
                        INSERT INTO display_object_model_revisions (
                            display_object_id, model_revision, task_id, source
                        ) VALUES (?, ?, ?, 'hololens')
                        """,
                        (display_object_id, entry["model_revision"], entry["task_id"]),
                    )
                    counts["model_revisions"] += 1
                if entry["pose"] is None:
                    continue
                connection.execute(
                    """
                    INSERT INTO display_object_pose_history (
                        display_object_id, task_id, model_revision, pose_revision,
                        pose_aruco_json, captured_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        display_object_id,
                        entry["task_id"],
                        entry["model_revision"],
                        entry["pose_revision"],
                        json.dumps(entry["pose"], ensure_ascii=False),
                        entry["captured_at"],
                        now,
                    ),
                )
                counts["pose_history"] += 1
                latest_pose = entry
            connection.execute(
                """
                UPDATE display_object_states
                SET active_model_revision = ?, active_model_task_id = ?,
                    latest_hololens_pose_revision = ?,
                    latest_hololens_pose_aruco_json = ?,
                    latest_hololens_task_id = ?, latest_hololens_captured_at = ?,
                    presence = 'UNKNOWN', updated_at = ?
                WHERE display_object_id = ?
                """,
                (
                    item["active_model_revision"],
                    item["active_model_task_id"],
                    int(item["latest_pose_revision"]),
                    json.dumps(latest_pose["pose"], ensure_ascii=False) if latest_pose else None,
                    latest_pose["task_id"] if latest_pose else None,
                    latest_pose["captured_at"] if latest_pose else None,
                    now,
                    display_object_id,
                ),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return counts


def _patch_task_jsons(database_path: Path, data_root: Path) -> dict[str, int]:
    """Write the strict revision contract and remove retired branch payloads."""

    revisions: dict[str, dict[str, Any]] = {}
    with _read_only_connection(database_path) as connection:
        for row in connection.execute(
            "SELECT task_id, display_object_id, model_revision FROM display_object_model_revisions"
        ):
            revisions[str(row["task_id"])] = {
                "display_object_id": str(row["display_object_id"]),
                "model_revision": int(row["model_revision"]),
            }
        for row in connection.execute(
            """
            SELECT task_id, display_object_id, model_revision, pose_revision
            FROM display_object_pose_history
            """
        ):
            revisions[str(row["task_id"])] = {
                "display_object_id": str(row["display_object_id"]),
                "model_revision": int(row["model_revision"]),
                "hololens_pose_revision": int(row["pose_revision"]),
            }
        tasks = connection.execute("SELECT task_id, json_path FROM tasks ORDER BY id").fetchall()

    result = {"updated": 0, "backups_created": 0, "legacy_sections_removed": 0}
    for row in tasks:
        task_id = str(row["task_id"] or "")
        path = _task_json_path(row["json_path"], data_root)
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MigrationError(f"Invalid task JSON {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MigrationError(f"Task JSON root is not an object: {path}")
        changed = False
        for legacy_key in (
            "TakenObjectDetection",
            "SAM3DBodyMesh",
            "HistoryPlacementRestoration",
        ):
            if legacy_key in payload:
                del payload[legacy_key]
                result["legacy_sections_removed"] += 1
                changed = True
        revision = revisions.get(task_id)
        if revision is not None:
            identity = payload.get("DisplayIdentity")
            if not isinstance(identity, dict) or str(
                identity.get("display_object_id") or ""
            ).strip() != revision["display_object_id"]:
                raise MigrationError(f"DisplayIdentity mismatch in {path}")
            for key in ("model_revision", "hololens_pose_revision"):
                if key in revision and identity.get(key) != revision[key]:
                    identity[key] = revision[key]
                    changed = True
        if not changed:
            continue
        backup_path = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup_path.exists():
            try:
                os.link(path, backup_path)
            except FileExistsError:
                pass
            else:
                result["backups_created"] += 1
        _atomic_write_json(path, payload)
        result["updated"] += 1
    return result


def _backup_metadata_payload(database_path: Path, backup_path: Path) -> dict[str, Any]:
    backup_fingerprint = _logical_database_fingerprint(backup_path)
    return {
        "format_version": 1,
        "created_at": _utc_now(),
        "source_database_path": str(database_path.resolve()),
        "source_fingerprint": backup_fingerprint,
        "backup_fingerprint": backup_fingerprint,
        "backup_sha256": _hash_file(backup_path),
    }


def _validate_existing_backup(
    database_path: Path,
    backup_path: Path,
    *,
    register_missing_metadata: bool,
) -> dict[str, Any]:
    _integrity_check(backup_path)
    if _schema_version(backup_path) is not None:
        raise MigrationError(
            f"Existing backup is not a pre-v2 database: {backup_path}"
        )

    metadata_path = _backup_metadata_path(backup_path)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MigrationError(f"Invalid backup metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            raise MigrationError(f"Unsupported backup metadata: {metadata_path}")
        actual_hash = _hash_file(backup_path)
        if metadata.get("backup_sha256") != actual_hash:
            raise MigrationError(f"Backup hash does not match metadata: {backup_path}")
        actual_fingerprint = _logical_database_fingerprint(backup_path)
        if metadata.get("backup_fingerprint") != actual_fingerprint:
            raise MigrationError(
                f"Backup logical fingerprint does not match metadata: {backup_path}"
            )
        if _schema_version(database_path) is None:
            current_fingerprint = _logical_database_fingerprint(database_path)
            if metadata.get("source_fingerprint") != current_fingerprint:
                raise MigrationError(
                    "Existing pre-v2 backup belongs to different database content; "
                    f"refusing to overwrite or continue: {backup_path}"
                )
        return {
            "status": "reused_verified",
            "sha256": actual_hash,
            "metadata_path": str(metadata_path),
        }

    # A backup without our provenance record is accepted only when it is an
    # exact logical copy of the still-unmigrated source. Once the source is v2,
    # provenance can no longer be established safely, so fail closed.
    if _schema_version(database_path) is not None:
        raise MigrationError(
            "Existing backup has no provenance metadata and the source is already v2: "
            f"{backup_path}"
        )
    if _logical_database_fingerprint(database_path) != _logical_database_fingerprint(
        backup_path
    ):
        raise MigrationError(
            "Existing backup is not an exact logical copy of the pre-v2 database: "
            f"{backup_path}"
        )
    if register_missing_metadata:
        _atomic_write_json(
            metadata_path,
            _backup_metadata_payload(database_path, backup_path),
        )
    return {
        "status": "reused_verified",
        "sha256": _hash_file(backup_path),
        "metadata_path": str(metadata_path),
    }


def _create_backup(database_path: Path, backup_path: Path) -> dict[str, Any]:
    if _schema_version(database_path) is not None:
        raise MigrationError(
            "Database is already v2 but its required pre-v2 backup is missing: "
            f"{backup_path}"
        )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup_path.with_name(f".{backup_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with _read_only_connection(database_path) as source:
            destination = sqlite3.connect(temporary)
            try:
                source.backup(destination)
                destination.commit()
            finally:
                destination.close()
        _integrity_check(temporary)
        if _logical_database_fingerprint(database_path) != _logical_database_fingerprint(
            temporary
        ):
            raise MigrationError("SQLite backup does not match the source database")
        # Never replace an existing pre-v2 backup, even under a race.
        try:
            os.link(temporary, backup_path)
        except FileExistsError as exc:
            raise MigrationError(f"Backup appeared concurrently: {backup_path}") from exc
        temporary.unlink()
        metadata_path = _backup_metadata_path(backup_path)
        _atomic_write_json(
            metadata_path,
            _backup_metadata_payload(database_path, backup_path),
        )
        return {
            "status": "created",
            "sha256": _hash_file(backup_path),
            "metadata_path": str(metadata_path),
        }
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _ensure_backup(database_path: Path, *, apply: bool) -> dict[str, Any]:
    backup_path = _backup_path(database_path)
    relative_status = {
        "path": str(backup_path),
        "metadata_path": str(_backup_metadata_path(backup_path)),
    }
    if backup_path.exists():
        relative_status.update(
            _validate_existing_backup(
                database_path,
                backup_path,
                register_missing_metadata=apply,
            )
        )
        return relative_status
    if _schema_version(database_path) == TARGET_SCHEMA_VERSION:
        # A completed migration intentionally retains no legacy backup. This
        # keeps --apply idempotent without restoring pre-v2 data into DVC.
        relative_status["status"] = "not_required_already_v2"
        relative_status["sha256"] = None
        return relative_status

    if not apply:
        relative_status["status"] = "would_create"
        return relative_status
    relative_status.update(_create_backup(database_path, backup_path))
    return relative_status


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_candidate(path: Path, data_root: Path) -> Path:
    absolute_path = path.absolute()
    absolute_root = data_root.absolute()
    if absolute_path == absolute_root or not _is_relative_to(absolute_path, absolute_root):
        raise MigrationError(f"Refusing cleanup path outside data root: {path}")
    # A candidate symlink is safe to unlink, but traversing a symlinked parent
    # could make rmtree operate outside data/. Resolve only the parent for that
    # distinction.
    resolved_root = data_root.resolve()
    resolved_parent = absolute_path.parent.resolve()
    if resolved_parent != resolved_root and not _is_relative_to(
        resolved_parent, resolved_root
    ):
        raise MigrationError(f"Refusing cleanup through symlinked parent: {path}")
    return absolute_path


def _discover_cleanup_candidates(data_root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    model_root = data_root / "model"
    if model_root.is_dir():
        for path in model_root.glob("**/worker/auxiliary_shigure_contact_body"):
            if path.exists() or path.is_symlink():
                candidates.append(("retired_sam3d_body_worker", path))
        for path in model_root.glob("**/result/08_sam3d_body*"):
            if path.exists() or path.is_symlink():
                candidates.append(("retired_sam3d_body_result", path))
        for path in model_root.glob("**/result/07_taken_detection*"):
            if path.exists() or path.is_symlink():
                candidates.append(("retired_taken_detection_result", path))

    database_backup = _backup_path(data_root / DATABASE_RELATIVE_PATH)
    migration_backups = (
        database_backup,
        _backup_metadata_path(database_backup),
        Path(str(database_backup) + "-shm"),
        Path(str(database_backup) + "-wal"),
    )
    for path in migration_backups:
        if path.exists() or path.is_symlink():
            candidates.append(("verified_migration_backup", path))
    if model_root.is_dir():
        for path in model_root.glob(f"**/task.json{BACKUP_SUFFIX}"):
            if path.exists() or path.is_symlink():
                candidates.append(
                    ("verified_task_json_migration_backup", path)
                )


    for pattern in (".tasks.db.pre_shigure_v2.*.tmp-shm", ".tasks.db.pre_shigure_v2.*.tmp-wal"):
        for path in (data_root / "database").glob(pattern):
            candidates.append(("abandoned_migration_sidecar", path))

    fixed = (
        ("legacy_realtime_tracking_cache", data_root / "realtime_tracking"),
        ("legacy_history_placement_cache", data_root / "history_placement_requests"),
        ("legacy_shigure_history_cache", data_root / "shigure_history_cache"),
        ("invalid_mask_overlay_checks", data_root / "shigure_mask_overlay_checks"),
    )
    for category, path in fixed:
        if path.exists() or path.is_symlink():
            candidates.append((category, path))

    unique: dict[Path, str] = {}
    for category, raw_path in candidates:
        path = _safe_candidate(raw_path, data_root)
        unique.setdefault(path, category)

    # If a selected directory contains another candidate, the outer deletion
    # already accounts for it and is the only manifest entry we need.
    result: list[tuple[str, Path]] = []
    for path in sorted(unique, key=lambda value: (len(value.parts), str(value))):
        if any(_is_relative_to(path, selected) for _, selected in result):
            continue
        result.append((unique[path], path))
    return result


def _tree_stats(path: Path, excluded: Sequence[Path] = ()) -> dict[str, int]:
    excluded_absolute = tuple(item.absolute() for item in excluded)

    def is_excluded(item: Path) -> bool:
        absolute = item.absolute()
        return any(absolute == root or _is_relative_to(absolute, root) for root in excluded_absolute)

    if not (path.exists() or path.is_symlink()) or is_excluded(path):
        return {"file_count": 0, "directory_count": 0, "bytes": 0}
    if path.is_symlink() or path.is_file():
        return {
            "file_count": 1,
            "directory_count": 0,
            "bytes": int(path.lstat().st_size),
        }

    file_count = 0
    directory_count = 1
    byte_count = 0
    for current, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            child = current_path / name
            if is_excluded(child):
                continue
            if child.is_symlink():
                file_count += 1
                byte_count += int(child.lstat().st_size)
            else:
                kept_directories.append(name)
                directory_count += 1
        directory_names[:] = kept_directories
        for name in file_names:
            child = current_path / name
            if is_excluded(child):
                continue
            file_count += 1
            try:
                byte_count += int(child.lstat().st_size)
            except FileNotFoundError:
                continue
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "bytes": byte_count,
    }


def _relative(path: Path, data_root: Path) -> str:
    try:
        return path.absolute().relative_to(data_root.absolute()).as_posix()
    except ValueError:
        return str(path)


def _cleanup_manifest_entries(
    candidates: Sequence[tuple[str, Path]],
    data_root: Path,
    *,
    action: str,
) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "path": _relative(path, data_root),
            "action": action,
            **_tree_stats(path),
        }
        for category, path in candidates
    ]


def _delete_candidates(candidates: Sequence[tuple[str, Path]]) -> None:
    failures: list[str] = []
    for _, path in sorted(candidates, key=lambda item: len(item[1].parts), reverse=True):
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise MigrationError("Failed to delete legacy artifacts: " + "; ".join(failures))


def _apply_task_db_v2_schema(database_path: Path) -> None:
    # task_db owns the canonical v2 DDL. Patch only its database target for this
    # one-shot process so tests and offline migrations can use an isolated root.
    import task_db

    previous_path = task_db.DATABASE_PATH
    previous_initialized = task_db._SCHEMA_INITIALIZED
    previous_marker_sync = task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START
    try:
        task_db.DATABASE_PATH = database_path
        task_db._SCHEMA_INITIALIZED = False
        # A schema migration must not mutate the marker registry from current
        # config; ArUco rows are preserved exactly here.
        task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START = False
        task_db.migrate_legacy_task_database_to_v2_once()
    finally:
        task_db.DATABASE_PATH = previous_path
        task_db._SCHEMA_INITIALIZED = previous_initialized
        task_db.ARUCO_SYNC_MARKER_REGISTRY_ON_START = previous_marker_sync


def _filesystem_preservation_manifest(
    data_root: Path,
    candidates: Sequence[tuple[str, Path]],
) -> list[dict[str, Any]]:
    excluded = [path for _, path in candidates]
    result: list[dict[str, Any]] = []
    for relative_root in PRESERVED_FILESYSTEM_ROOTS:
        path = data_root / relative_root
        result.append(
            {
                "path": relative_root.as_posix(),
                **_tree_stats(path, excluded=excluded),
            }
        )
    return result


def _totals(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "path_count": len(entries),
        "file_count": sum(int(entry["file_count"]) for entry in entries),
        "directory_count": sum(int(entry["directory_count"]) for entry in entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
    }


def run_migration(data_root: Path, *, apply: bool = False) -> dict[str, Any]:
    data_root = data_root.expanduser().absolute()
    database_path = data_root / DATABASE_RELATIVE_PATH
    if not database_path.is_file():
        raise MigrationError(f"Task database does not exist: {database_path}")
    _integrity_check(database_path)

    schema_before = _schema_version(database_path)
    candidates = _discover_cleanup_candidates(data_root)
    deleted_entries = _cleanup_manifest_entries(
        candidates,
        data_root,
        action="would_delete" if not apply else "deleted",
    )
    preserved_before = _row_counts(database_path, PRESERVED_TABLES)
    legacy_before = _row_counts(database_path, LEGACY_TABLES)
    identity_plan, identity_skipped = _capture_identity_migration_plan(
        database_path,
        data_root,
    )
    state_plan, state_skipped = _capture_state_migration_plan(
        database_path,
        data_root,
    )
    backup = _ensure_backup(database_path, apply=apply)

    if apply:
        _apply_task_db_v2_schema(database_path)
        _integrity_check(database_path)
        schema_after = _schema_version(database_path)
        if schema_after != TARGET_SCHEMA_VERSION:
            raise MigrationError(
                f"task_db did not install schema v{TARGET_SCHEMA_VERSION}; got {schema_after}"
            )
        preserved_after = _row_counts(database_path, PRESERVED_TABLES)
        changed_counts = {
            table_name: {"before": before, "after": preserved_after[table_name]}
            for table_name, before in preserved_before.items()
            if before is not None and preserved_after[table_name] != before
        }
        if changed_counts:
            raise MigrationError(
                "Preserved database row counts changed: "
                + json.dumps(changed_counts, ensure_ascii=False, sort_keys=True)
            )
        legacy_after = _row_counts(database_path, LEGACY_TABLES)
        if any(value is not None for value in legacy_after.values()):
            raise MigrationError(f"Legacy tables remain after migration: {legacy_after}")
        migrated_state_counts = _materialize_capture_states(database_path, state_plan)
        task_json_migration = _patch_task_jsons(database_path, data_root)
        migrated_identity_reference_count = _materialize_identity_references(
            database_path,
            identity_plan,
        )
        # SQLite/task JSON backups exist only until every migration validation
        # has succeeded. Rediscover so backups created by this run are included,
        # then remove all retired and pre-v2 material from the active data root.
        candidates = _discover_cleanup_candidates(data_root)
        deleted_entries = _cleanup_manifest_entries(
            candidates, data_root, action="deleted"
        )

        preserved_after = _row_counts(database_path, PRESERVED_TABLES)
        _delete_candidates(candidates)
        for _, path in candidates:
            if path.exists() or path.is_symlink():
                raise MigrationError(f"Legacy artifact still exists after deletion: {path}")
    else:
        schema_after = schema_before
        preserved_after = preserved_before
        legacy_after = legacy_before
        migrated_identity_reference_count = 0
        migrated_state_counts = {"objects": 0, "model_revisions": 0, "pose_history": 0}
        task_json_migration = {
            "updated": 0,
            "backups_created": 0,
            "legacy_sections_removed": 0,
        }

    filesystem_preserved = _filesystem_preservation_manifest(data_root, ())
    if not apply:
        filesystem_preserved = _filesystem_preservation_manifest(data_root, candidates)

    return {
        "migration": "shigure_v2",
        "mode": "apply" if apply else "dry-run",
        "status": "completed" if apply else "planned",
        "data_root": str(data_root),
        "database": {
            "path": DATABASE_RELATIVE_PATH.as_posix(),
            "schema_version_before": schema_before,
            "schema_version_after": schema_after,
            "backup": {
                **backup,
                "path": _relative(Path(str(backup["path"])), data_root),
                "metadata_path": _relative(
                    Path(str(backup["metadata_path"])), data_root
                ),
                "retained": (
                    False
                    if apply
                    or backup.get("status") == "not_required_already_v2"
                    else None
                ),
                "cleanup_status": (
                    "not_present_already_v2"
                    if backup.get("status") == "not_required_already_v2"
                    else "deleted_after_success" if apply else "planned"
                ),
            },
            "preserved_row_counts_before": preserved_before,
            "preserved_row_counts_after": preserved_after,
            "deleted_legacy_row_counts": legacy_before,
            "legacy_tables_after": legacy_after,
            "identity_references": {
                "planned": len(identity_plan),
                "migrated": migrated_identity_reference_count,
                "skipped": identity_skipped,
            },
            "capture_state_backfill": {
                "planned_objects": len(state_plan),
                "planned_captures": sum(len(item["entries"]) for item in state_plan),
                "migrated": migrated_state_counts,
                "skipped": state_skipped,
            },
            "task_json": {
                **task_json_migration,
                "planned_revision_updates": sum(len(item["entries"]) for item in state_plan),
                "retired_keys": [
                    "TakenObjectDetection",
                    "SAM3DBodyMesh",
                    "HistoryPlacementRestoration",
                ],
            },
        },
        "preserved": {
            "filesystem": filesystem_preserved,
            "totals": _totals(filesystem_preserved),
        },
        "deleted": {
            "filesystem": deleted_entries,
            "totals": _totals(deleted_entries),
        },
        "cleanup_rules": [
            "model/**/worker/auxiliary_shigure_contact_body",
            "model/**/result/08_sam3d_body*",
            "model/**/result/07_taken_detection*",
            "realtime_tracking",
            "history_placement_requests",
            "shigure_history_cache",
            "shigure_mask_overlay_checks",
            "database/tasks.db.pre_shigure_v2*",
            "model/**/task.json.pre_shigure_v2",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate tasks.db and data artifacts to the Shigure v2 protocol. "
            "The default is a read-only dry run."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Use temporary backups, migrate v2, then delete all pre-v2 data on success.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = run_migration(arguments.data_root, apply=arguments.apply)
    except (MigrationError, OSError, sqlite3.DatabaseError) as exc:
        manifest = {
            "migration": "shigure_v2",
            "mode": "apply" if arguments.apply else "dry-run",
            "status": "failed",
            "data_root": str(arguments.data_root.expanduser().absolute()),
            "error": str(exc),
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
