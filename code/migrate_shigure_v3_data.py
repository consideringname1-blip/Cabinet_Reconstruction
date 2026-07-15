from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "database" / "tasks.db"
BACKUP_SUFFIX = ".pre_shigure_v3"
ORIGIN_REPAIR_BACKUP_SUFFIX = ".pre_origin_backfill_v1"
BACKUP_METADATA_SUFFIX = ".meta.json"


class MigrationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _integrity_check(path: Path) -> None:
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            messages = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
    except sqlite3.DatabaseError as exc:
        raise MigrationError(f"SQLite integrity check failed for {path}: {exc}") from exc
    if messages != ["ok"]:
        raise MigrationError(
            f"SQLite integrity check failed for {path}: {'; '.join(messages)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_version(path: Path) -> int | None:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        row = connection.execute(
            """
            SELECT schema_version
            FROM schema_metadata
            WHERE schema_name = 'shigure_runtime'
            """
        ).fetchone()
    return None if row is None else int(row[0])


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


def _create_backup(
    database_path: Path,
    *,
    backup_suffix: str = BACKUP_SUFFIX,
    source_schema_version: int = 2,
    target_schema_version: int = 3,
) -> tuple[Path, Path]:
    backup_path = database_path.with_name(database_path.name + backup_suffix)
    metadata_path = backup_path.with_name(backup_path.name + BACKUP_METADATA_SUFFIX)
    if backup_path.exists() or metadata_path.exists():
        raise MigrationError(
            "Refusing to overwrite an existing Shigure database backup: "
            f"{backup_path} (move it aside only after verifying it)"
        )

    source_uri = database_path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(source_uri, uri=True)) as source:
            with closing(sqlite3.connect(backup_path)) as destination:
                source.backup(destination)
    except Exception:
        try:
            backup_path.unlink()
        except FileNotFoundError:
            pass
        raise

    _integrity_check(backup_path)
    with backup_path.open("rb") as stream:
        os.fsync(stream.fileno())
    _atomic_write_json(
        metadata_path,
        {
            "backup_path": str(backup_path.resolve()),
            "backup_sha256": _sha256(backup_path),
            "created_at": _utc_now(),
            "source_database": str(database_path.resolve()),
            "source_schema_version": int(source_schema_version),
            "target_schema_version": int(target_schema_version),
        },
    )
    return backup_path, metadata_path


def _task_db_module(database_path: Path):
    code_root = Path(__file__).resolve().parent
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    import task_db

    task_db.DATABASE_PATH = database_path.resolve()
    task_db._SCHEMA_INITIALIZED = False
    return task_db


def _validate_source(database_path: Path) -> int:
    _integrity_check(database_path)
    task_db = _task_db_module(database_path)
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        version = _schema_version(database_path)
        if version == 2:
            task_db._validate_v2_schema(connection)
        elif version == 3:
            task_db._validate_v3_schema(connection)
        else:
            raise MigrationError(
                f"Expected strict Shigure schema version 2 or 3, got {version}"
            )
    return int(version)


def _origin_repair_completed(database_path: Path) -> bool:
    uri = database_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        row = connection.execute(
            """
            SELECT detail_json
            FROM schema_metadata
            WHERE schema_name = 'shigure_runtime'
            """
        ).fetchone()
    if row is None:
        return False
    try:
        detail = json.loads(str(row[0] or "{}"))
    except json.JSONDecodeError:
        return False
    return isinstance(detail, dict) and isinstance(
        detail.get("hololens_origin_backfill_v1"), dict
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely upgrade a strict Shigure v2 SQLite database to v3."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"database path (default: {DEFAULT_DATABASE_PATH})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create a verified backup and apply the atomic migration",
    )
    parser.add_argument(
        "--repair-origin-history",
        action="store_true",
        help=(
            "on an existing v3 database, merge calibrated HoloLens capture "
            "poses with native v3 origin history"
        ),
    )
    args = parser.parse_args(argv)
    database_path = args.database.expanduser().resolve()
    if not database_path.is_file():
        raise MigrationError(f"Database not found: {database_path}")

    version = _validate_source(database_path)
    if version == 3:
        repaired = _origin_repair_completed(database_path)
        if not args.repair_origin_history:
            print(
                json.dumps(
                    {
                        "status": "already_v3",
                        "database": str(database_path),
                        "origin_history_repaired": repaired,
                    }
                )
            )
            return 0
        if repaired:
            print(
                json.dumps(
                    {
                        "status": "already_repaired",
                        "database": str(database_path),
                    }
                )
            )
            return 0
        if not args.apply:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "database": str(database_path),
                        "source_schema_version": 3,
                        "target_schema_version": 3,
                        "apply": (
                            "python code/migrate_shigure_v3_data.py "
                            "--repair-origin-history --apply"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        backup_path, metadata_path = _create_backup(
            database_path,
            backup_suffix=ORIGIN_REPAIR_BACKUP_SUFFIX,
            source_schema_version=3,
            target_schema_version=3,
        )
        task_db = _task_db_module(database_path)
        result = task_db.repair_v3_origin_history_from_hololens_once()
        if _validate_source(database_path) != 3:
            raise MigrationError("Origin repair damaged the strict v3 schema")
        print(
            json.dumps(
                {
                    **result,
                    "database": str(database_path),
                    "backup": str(backup_path),
                    "backup_metadata": str(metadata_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "database": str(database_path),
                    "source_schema_version": 2,
                    "target_schema_version": 3,
                    "apply": "python code/migrate_shigure_v3_data.py --apply",
                },
                ensure_ascii=False,
            )
        )
        return 0

    backup_path, metadata_path = _create_backup(database_path)
    task_db = _task_db_module(database_path)
    result = task_db.migrate_strict_v2_database_to_v3_once()
    if _validate_source(database_path) != 3:
        raise MigrationError("Migration returned without producing strict schema v3")
    print(
        json.dumps(
            {
                **result,
                "database": str(database_path),
                "backup": str(backup_path),
                "backup_metadata": str(metadata_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MigrationError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
