"""Host-wide atomic, reservation-aware GPU placement for worker processes.

The operating system only reports memory after a CUDA process has allocated it.
That is too late for launching several large workers at once: every launcher can
otherwise observe the same apparently-empty GPU.  :class:`GpuLeaseManager`
therefore accounts the caller's *expected* peak memory immediately and keeps that
reservation until it is explicitly released or its bound process exits.

Memory values in this module are mebibytes (MiB).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


DEFAULT_RESERVE_FLOOR_MIB = 3 * 1024
DEFAULT_RESERVE_FRACTION = 0.08
DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "GPU_LEASE_REGISTRY_PATH",
        "/tmp/shigure_gpu_leases_v1.sqlite3",
    )
).expanduser()


class GpuLeaseError(RuntimeError):
    """Base class for GPU lease failures."""


class GpuUnavailableError(GpuLeaseError):
    """Raised by a non-waiting acquisition when no GPU can safely fit it."""


class GpuLeaseTimeout(GpuUnavailableError):
    """Raised when a waiting acquisition reaches its timeout."""


class UnknownGpuLeaseError(GpuLeaseError):
    """Raised when an operation refers to an inactive or foreign lease."""


@dataclass(frozen=True)
class GpuProcessUsage:
    """GPU memory attributed to one operating-system process."""

    pid: int
    used_mib: int

    def __post_init__(self) -> None:
        if int(self.pid) <= 0:
            raise ValueError("pid must be positive")
        if int(self.used_mib) < 0:
            raise ValueError("used_mib must not be negative")


@dataclass(frozen=True)
class GpuSnapshot:
    """One point-in-time view of a physical GPU."""

    index: str
    total_mib: int
    used_mib: int
    processes: tuple[GpuProcessUsage, ...] = ()
    uuid: str | None = None

    def __post_init__(self) -> None:
        if not str(self.index).strip():
            raise ValueError("GPU index must not be empty")
        if int(self.total_mib) <= 0:
            raise ValueError("total_mib must be positive")
        if int(self.used_mib) < 0:
            raise ValueError("used_mib must not be negative")


SnapshotProvider = Callable[[], Iterable[GpuSnapshot]]
PidLivenessProvider = Callable[[int], bool]
PidIdentityProvider = Callable[[int], str | None]


def _parse_nonnegative_int(value: str) -> int | None:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def query_gpu_snapshots() -> list[GpuSnapshot]:
    """Query NVIDIA GPUs and per-process usage.

    Graphics or driver allocations that are absent from the compute-process
    query remain represented by ``memory.used`` and are consequently treated as
    external usage.  An unavailable ``nvidia-smi`` returns an empty snapshot;
    it never causes a best-effort assignment.
    """

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = subprocess.run(
            gpu_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if gpu_result.returncode != 0:
        return []

    gpu_rows: list[tuple[str, str, int, int]] = []
    for line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        total_mib = _parse_nonnegative_int(parts[2])
        used_mib = _parse_nonnegative_int(parts[3])
        if total_mib is None or total_mib <= 0 or used_mib is None:
            continue
        gpu_rows.append((parts[0], parts[1], total_mib, used_mib))

    if not gpu_rows:
        return []

    usage_by_uuid: dict[str, dict[int, int]] = {}
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        process_result = subprocess.run(
            process_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        process_result = None

    if process_result is not None and process_result.returncode == 0:
        for line in process_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            pid = _parse_nonnegative_int(parts[1])
            used_mib = _parse_nonnegative_int(parts[2])
            if pid is None or pid <= 0 or used_mib is None:
                continue
            per_gpu = usage_by_uuid.setdefault(parts[0], {})
            per_gpu[pid] = per_gpu.get(pid, 0) + used_mib

    snapshots: list[GpuSnapshot] = []
    for index, gpu_uuid, total_mib, used_mib in gpu_rows:
        processes = tuple(
            GpuProcessUsage(pid=pid, used_mib=process_mib)
            for pid, process_mib in sorted(usage_by_uuid.get(gpu_uuid, {}).items())
        )
        snapshots.append(
            GpuSnapshot(
                index=index,
                uuid=gpu_uuid,
                total_mib=total_mib,
                used_mib=used_mib,
                processes=processes,
            )
        )
    return snapshots


def pid_is_alive(pid: int) -> bool:
    """Return whether a PID still exists, treating permission denial as alive."""

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pid_start_identity(pid: int) -> str | None:
    """Return boot plus process start identity so PID reuse cannot inherit a lease."""

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        remainder = stat.rsplit(")", 1)[1].strip().split()
        # After the comm field, index 0 is proc field 3. Start time is field 22.
        start_ticks = remainder[19]
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            boot_id = ""
        prefix = f"linux-boot:{boot_id}:" if boot_id else "linux:"
        return f"{prefix}start-ticks:{start_ticks}"
    except (IndexError, OSError, ValueError):
        return None


@dataclass
class _LeaseRecord:
    lease_id: str
    service: str
    gpu_id: str
    required_mib: int
    acquired_monotonic: float
    acquired_unix: float
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_pid: int = 0
    owner_start_identity: str | None = None
    pid: int | None = None
    pid_start_identity: str | None = None
    usage_pids: set[int] = field(default_factory=set)
    usage_pid_identities: dict[int, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class _GpuAccounting:
    snapshot: GpuSnapshot
    reserve_mib: int
    external_used_mib: int
    committed_mib: int
    managed_reported_mib: int
    managed_overage_mib: int
    misplaced_managed_mib: int
    accounted_used_mib: int
    allocatable_free_mib: int
    lease_ids: tuple[str, ...]


class GpuLease:
    """Handle returned by :meth:`GpuLeaseManager.acquire`.

    The handle itself is a context manager.  Leaving its context always releases
    the reservation, including when the body raises.
    """

    def __init__(
        self,
        manager: "GpuLeaseManager",
        *,
        lease_id: str,
        service: str,
        gpu_id: str,
        required_mib: int,
    ) -> None:
        self._manager = manager
        self.lease_id = lease_id
        self.service = service
        self.gpu_id = gpu_id
        self.required_mib = required_mib

    @property
    def cuda_env(self) -> dict[str, str]:
        """Environment fragment suitable for a worker subprocess."""

        return {"CUDA_VISIBLE_DEVICES": self.gpu_id}

    @property
    def pid(self) -> int | None:
        return self._manager._lease_pid(self.lease_id)

    @property
    def active(self) -> bool:
        return self._manager.is_active(self.lease_id)

    def bind_pid(self, pid: int, *, tracks_usage: bool = True) -> None:
        self._manager.bind_pid(self.lease_id, pid, tracks_usage=tracks_usage)

    def bind_usage_pid(self, pid: int) -> None:
        self._manager.bind_usage_pid(self.lease_id, pid)

    def release(self) -> bool:
        return self._manager.release(self.lease_id)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class GpuLeaseManager:
    """Thread- and process-safe expected-memory allocator for host GPUs.

    Placement is *best fit*: among GPUs that can fit the complete request while
    preserving driver/system reserve, choose the one with the least capacity left
    afterward. Shared reservation reads, fitting, and commits use one short SQLite
    write transaction; the slower external GPU snapshot is collected beforehand.
    """

    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider = query_gpu_snapshots,
        pid_liveness_provider: PidLivenessProvider = pid_is_alive,
        reserve_floor_mib: int = DEFAULT_RESERVE_FLOOR_MIB,
        reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
        poll_interval_s: float = 1.0,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        pid_identity_provider: PidIdentityProvider = pid_start_identity,
    ) -> None:
        if int(reserve_floor_mib) < 0:
            raise ValueError("reserve_floor_mib must not be negative")
        if not 0.0 <= float(reserve_fraction) < 1.0:
            raise ValueError("reserve_fraction must be in [0, 1)")
        if float(poll_interval_s) <= 0:
            raise ValueError("poll_interval_s must be positive")
        self._snapshot_provider = snapshot_provider
        self._pid_liveness_provider = pid_liveness_provider
        self._reserve_floor_mib = int(reserve_floor_mib)
        self._reserve_fraction = float(reserve_fraction)
        self._poll_interval_s = float(poll_interval_s)
        self._registry_path = Path(registry_path).expanduser().resolve()
        self._pid_identity_provider = pid_identity_provider
        self._owner_pid = os.getpid()
        self._owner_start_identity = self._safe_pid_identity(self._owner_pid)
        self._condition = threading.Condition(threading.RLock())
        self._leases: dict[str, _LeaseRecord] = {}
        self._pid_to_lease: dict[int, str] = {}
        self._last_snapshot_error: str | None = None

    def reserve_for_total(self, total_mib: int) -> int:
        """Driver/system memory that must remain unallocated on a GPU."""

        return max(
            self._reserve_floor_mib,
            int(math.ceil(int(total_mib) * self._reserve_fraction)),
        )

    def _reset_for_current_process(self) -> None:
        """Discard inherited handles and locks after a fork."""

        self._owner_pid = os.getpid()
        self._owner_start_identity = self._safe_pid_identity(self._owner_pid)
        self._condition = threading.Condition(threading.RLock())
        self._leases = {}
        self._pid_to_lease = {}
        self._last_snapshot_error = None

    def _ensure_current_process(self) -> None:
        if os.getpid() != self._owner_pid:
            self._reset_for_current_process()

    @staticmethod
    def _is_registry_busy(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            return True
        error_code = getattr(exc, "sqlite_errorcode", None)
        # SQLITE_BUSY=5 and SQLITE_LOCKED=6. Mask extended result codes and
        # avoid module constants absent from the deployed Python 3.10 sqlite3.
        return (
            isinstance(error_code, int)
            and (int(error_code) & 0xFF) in {5, 6}
        )

    def _safe_pid_identity(self, pid: int) -> str | None:
        try:
            value = self._pid_identity_provider(int(pid))
        except Exception:
            return None
        text = str(value).strip() if value is not None else ""
        return text or None

    def _pid_matches(
        self, pid: int, expected_identity: str | None
    ) -> bool:
        try:
            alive = bool(self._pid_liveness_provider(int(pid)))
        except Exception:
            # Uncertain liveness is not authority to discard a reservation.
            return True
        if not alive:
            return False
        if expected_identity is None:
            return True
        current = self._safe_pid_identity(int(pid))
        return current is None or current == expected_identity

    @staticmethod
    def _record_to_payload(record: _LeaseRecord) -> dict[str, Any]:
        return {
            "lease_id": record.lease_id,
            "service": record.service,
            "gpu_id": record.gpu_id,
            "required_mib": int(record.required_mib),
            "acquired_monotonic": float(record.acquired_monotonic),
            "acquired_unix": float(record.acquired_unix),
            "metadata": dict(record.metadata),
            "owner_pid": int(record.owner_pid),
            "owner_start_identity": record.owner_start_identity,
            "pid": int(record.pid) if record.pid is not None else None,
            "pid_start_identity": record.pid_start_identity,
            "usage_pids": sorted(int(pid) for pid in record.usage_pids),
            "usage_pid_identities": {
                str(pid): record.usage_pid_identities.get(pid)
                for pid in sorted(record.usage_pids)
            },
        }

    @staticmethod
    def _record_from_payload(payload: Mapping[str, Any]) -> _LeaseRecord:
        required = {
            "lease_id",
            "service",
            "gpu_id",
            "required_mib",
            "acquired_monotonic",
            "acquired_unix",
            "metadata",
            "owner_pid",
            "owner_start_identity",
            "pid",
            "pid_start_identity",
            "usage_pids",
            "usage_pid_identities",
        }
        missing = required.difference(payload)
        if missing:
            raise GpuLeaseError(
                f"shared GPU lease record is missing fields: {sorted(missing)}"
            )
        lease_id = str(payload["lease_id"]).strip()
        service = str(payload["service"]).strip()
        gpu_id = str(payload["gpu_id"]).strip()
        required_mib = int(payload["required_mib"])
        owner_pid = int(payload["owner_pid"])
        if (
            not lease_id
            or not service
            or not gpu_id
            or required_mib <= 0
            or owner_pid <= 0
        ):
            raise GpuLeaseError("shared GPU lease record is invalid")
        raw_metadata = payload["metadata"]
        raw_usage = payload["usage_pids"]
        raw_identities = payload["usage_pid_identities"]
        if (
            not isinstance(raw_metadata, Mapping)
            or not isinstance(raw_usage, list)
            or not isinstance(raw_identities, Mapping)
        ):
            raise GpuLeaseError("shared GPU lease record has invalid containers")
        usage_pids = {int(pid) for pid in raw_usage}
        if any(pid <= 0 for pid in usage_pids):
            raise GpuLeaseError("shared GPU lease contains an invalid usage PID")
        pid_value = payload["pid"]
        bound_pid = int(pid_value) if pid_value is not None else None
        if bound_pid is not None and bound_pid <= 0:
            raise GpuLeaseError("shared GPU lease contains an invalid owner PID")
        return _LeaseRecord(
            lease_id=lease_id,
            service=service,
            gpu_id=gpu_id,
            required_mib=required_mib,
            acquired_monotonic=float(payload["acquired_monotonic"]),
            acquired_unix=float(payload["acquired_unix"]),
            metadata=dict(raw_metadata),
            owner_pid=owner_pid,
            owner_start_identity=(
                str(payload["owner_start_identity"])
                if payload["owner_start_identity"] is not None
                else None
            ),
            pid=bound_pid,
            pid_start_identity=(
                str(payload["pid_start_identity"])
                if payload["pid_start_identity"] is not None
                else None
            ),
            usage_pids=usage_pids,
            usage_pid_identities={
                pid: (
                    str(raw_identities.get(str(pid)))
                    if raw_identities.get(str(pid)) is not None
                    else None
                )
                for pid in usage_pids
            },
        )

    @contextmanager
    def _shared_records_locked(
        self,
        *,
        busy_timeout_s: float | None = None,
    ) -> Iterator[dict[str, _LeaseRecord]]:
        """Lock, load, mutate, and atomically commit the host-wide registry."""

        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_timeout = (
            max(5.0, self._poll_interval_s * 5.0)
            if busy_timeout_s is None
            else max(0.0, float(busy_timeout_s))
        )
        connection = sqlite3.connect(
            str(self._registry_path),
            timeout=sqlite_timeout,
            isolation_level=None,
        )
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gpu_leases (
                    lease_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            records: dict[str, _LeaseRecord] = {}
            for lease_id, payload_json in connection.execute(
                "SELECT lease_id, payload_json FROM gpu_leases"
            ):
                try:
                    payload = json.loads(str(payload_json))
                except json.JSONDecodeError as exc:
                    raise GpuLeaseError(
                        f"shared GPU lease registry is corrupt: {exc}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise GpuLeaseError(
                        "shared GPU lease registry contains a non-object record"
                    )
                record = self._record_from_payload(payload)
                if str(lease_id) != record.lease_id:
                    raise GpuLeaseError(
                        "shared GPU lease registry key does not match payload"
                    )
                records[record.lease_id] = record
            yield records
            connection.execute("DELETE FROM gpu_leases")
            connection.executemany(
                "INSERT INTO gpu_leases (lease_id, payload_json) VALUES (?, ?)",
                [
                    (
                        record.lease_id,
                        json.dumps(
                            self._record_to_payload(record),
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        ),
                    )
                    for record in records.values()
                ],
            )
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def _reap_records_locked(
        self, records: dict[str, _LeaseRecord]
    ) -> list[str]:
        reaped: list[str] = []
        for lease_id, record in list(records.items()):
            lifetime_pid = record.pid or record.owner_pid
            lifetime_identity = (
                record.pid_start_identity
                if record.pid is not None
                else record.owner_start_identity
            )
            if not self._pid_matches(lifetime_pid, lifetime_identity):
                records.pop(lease_id, None)
                reaped.append(lease_id)
                continue
            for usage_pid in tuple(record.usage_pids):
                expected = record.usage_pid_identities.get(usage_pid)
                if self._pid_matches(usage_pid, expected):
                    continue
                record.usage_pids.discard(usage_pid)
                record.usage_pid_identities.pop(usage_pid, None)
        return reaped

    def _sync_local_records_locked(
        self, shared_records: Mapping[str, _LeaseRecord]
    ) -> None:
        for lease_id in tuple(self._leases):
            shared = shared_records.get(lease_id)
            if shared is None:
                self._leases.pop(lease_id, None)
            else:
                self._leases[lease_id] = shared
        self._pid_to_lease.clear()
        for lease_id, record in self._leases.items():
            if record.pid is not None:
                self._pid_to_lease[record.pid] = lease_id
            for usage_pid in record.usage_pids:
                self._pid_to_lease[usage_pid] = lease_id

    @staticmethod
    def _shared_pid_owner(
        records: Mapping[str, _LeaseRecord],
        pid: int,
    ) -> str | None:
        for lease_id, record in records.items():
            if record.pid == pid or pid in record.usage_pids:
                return lease_id
        return None

    def acquire(
        self,
        required_mib: int,
        *,
        service: str,
        allowed_gpu_ids: Iterable[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GpuLease:
        """Atomically reserve expected peak memory and return its GPU assignment.

        ``wait=False`` fails immediately.  A waiting call periodically refreshes
        GPU state so that external jobs ending can make progress even when no
        manager operation signals the condition.  No failure path creates a lease.
        """

        required = int(required_mib)
        if required <= 0:
            raise ValueError("required_mib must be positive")
        service_name = str(service).strip()
        if not service_name:
            raise ValueError("service must not be empty")
        if timeout is not None and float(timeout) < 0:
            raise ValueError("timeout must not be negative")
        allowed = None
        if allowed_gpu_ids is not None:
            allowed = {str(gpu_id).strip() for gpu_id in allowed_gpu_ids}
            allowed.discard("")
            if not allowed:
                raise ValueError("allowed_gpu_ids must contain at least one GPU")

        self._ensure_current_process()
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while True:
                acquired: _LeaseRecord | None = None
                snapshots = self._snapshots_locked()
                try:
                    with self._shared_records_locked(
                        busy_timeout_s=0.0
                    ) as shared_records:
                        self._reap_records_locked(shared_records)
                        self._sync_local_records_locked(shared_records)
                        # External GPU state is sampled before taking the host-wide
                        # write lock. Reservation read, fit, and insert remain one
                        # atomic transaction across every server process.
                        accounting = self._account_locked(
                            snapshots, shared_records
                        )
                        candidates = [
                            item
                            for item in accounting
                            if (allowed is None or item.snapshot.index in allowed)
                            and item.allocatable_free_mib >= required
                        ]
                        if candidates:
                            # Pack existing GPUs before opening an emptier one.
                            chosen = min(
                                candidates,
                                key=lambda item: (
                                    item.allocatable_free_mib - required,
                                    _gpu_sort_key(item.snapshot.index),
                                ),
                            )
                            lease_id = uuid.uuid4().hex
                            acquired = _LeaseRecord(
                                lease_id=lease_id,
                                service=service_name,
                                gpu_id=chosen.snapshot.index,
                                required_mib=required,
                                acquired_monotonic=time.monotonic(),
                                acquired_unix=time.time(),
                                metadata=dict(metadata or {}),
                                owner_pid=self._owner_pid,
                                owner_start_identity=self._owner_start_identity,
                            )
                            shared_records[lease_id] = acquired
                        reason = self._unavailable_reason(
                            required, allowed, accounting
                        )
                except sqlite3.OperationalError as exc:
                    if not self._is_registry_busy(exc):
                        raise
                    reason = (
                        f"cannot reserve {required} MiB: "
                        "shared GPU lease registry is busy"
                    )

                if acquired is not None:
                    self._leases[acquired.lease_id] = acquired
                    return GpuLease(
                        self,
                        lease_id=acquired.lease_id,
                        service=service_name,
                        gpu_id=acquired.gpu_id,
                        required_mib=required,
                    )
                if not wait:
                    raise GpuUnavailableError(reason)

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise GpuLeaseTimeout(reason)
                wait_for = self._poll_interval_s
                if remaining is not None:
                    wait_for = min(wait_for, remaining)
                self._condition.wait(wait_for)

    @contextmanager
    def reservation(
        self,
        required_mib: int,
        *,
        service: str,
        allowed_gpu_ids: Iterable[str] | None = None,
        wait: bool = True,
        timeout: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[GpuLease]:
        """Acquire a lease for the duration of a ``with`` block."""

        lease = self.acquire(
            required_mib,
            service=service,
            allowed_gpu_ids=allowed_gpu_ids,
            wait=wait,
            timeout=timeout,
            metadata=metadata,
        )
        try:
            yield lease
        finally:
            lease.release()

    def bind_pid(
        self,
        lease: str | GpuLease,
        pid: int,
        *,
        tracks_usage: bool = True,
    ) -> None:
        """Bind the lease lifetime owner and optionally its GPU accounting PID."""

        self._ensure_current_process()
        lease_id = _lease_id(lease)
        process_id = int(pid)
        if process_id <= 0:
            raise ValueError("pid must be positive")
        with self._condition:
            if lease_id not in self._leases:
                raise UnknownGpuLeaseError(f"inactive GPU lease: {lease_id}")
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
                record = shared_records.get(lease_id)
                if record is None or lease_id not in self._leases:
                    raise UnknownGpuLeaseError(
                        f"inactive GPU lease: {lease_id}"
                    )
                if record.pid == process_id:
                    if tracks_usage:
                        record.usage_pids.add(process_id)
                        record.usage_pid_identities[process_id] = (
                            self._safe_pid_identity(process_id)
                        )
                else:
                    if record.pid is not None:
                        raise GpuLeaseError(
                            f"lease {lease_id} is already bound to PID "
                            f"{record.pid}"
                        )
                    owner = self._shared_pid_owner(
                        shared_records, process_id
                    )
                    if owner is not None and owner != lease_id:
                        raise GpuLeaseError(
                            f"PID {process_id} is already bound to lease "
                            f"{owner}"
                        )
                    record.pid = process_id
                    record.pid_start_identity = self._safe_pid_identity(
                        process_id
                    )
                    if tracks_usage:
                        record.usage_pids.add(process_id)
                        record.usage_pid_identities[process_id] = (
                            record.pid_start_identity
                        )
                self._leases[lease_id] = record
                self._sync_local_records_locked(shared_records)
            self._condition.notify_all()

    def bind_usage_pid(self, lease: str | GpuLease, pid: int) -> None:
        """Associate a CUDA child while retaining the original owner lifetime."""

        self._ensure_current_process()
        lease_id = _lease_id(lease)
        process_id = int(pid)
        if process_id <= 0:
            raise ValueError("pid must be positive")
        with self._condition:
            if lease_id not in self._leases:
                raise UnknownGpuLeaseError(f"inactive GPU lease: {lease_id}")
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
                record = shared_records.get(lease_id)
                if record is None or lease_id not in self._leases:
                    raise UnknownGpuLeaseError(
                        f"inactive GPU lease: {lease_id}"
                    )
                if process_id not in record.usage_pids:
                    owner = self._shared_pid_owner(
                        shared_records, process_id
                    )
                    if owner is not None and owner != lease_id:
                        raise GpuLeaseError(
                            f"PID {process_id} is already bound to lease "
                            f"{owner}"
                        )
                    record.usage_pids.add(process_id)
                    record.usage_pid_identities[process_id] = (
                        self._safe_pid_identity(process_id)
                    )
                self._leases[lease_id] = record
                self._sync_local_records_locked(shared_records)
            self._condition.notify_all()

    def release(self, lease: str | GpuLease) -> bool:
        """Release an active lease; return False when already inactive."""

        self._ensure_current_process()
        lease_id = _lease_id(lease)
        with self._condition:
            if lease_id not in self._leases:
                return False
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                removed = shared_records.pop(lease_id, None)
                self._leases.pop(lease_id, None)
                self._sync_local_records_locked(shared_records)
            self._condition.notify_all()
            return removed is not None

    def reap(self) -> tuple[str, ...]:
        """Release shared leases whose lifetime PID no longer matches."""

        self._ensure_current_process()
        with self._condition:
            with self._shared_records_locked() as shared_records:
                reaped = self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
            if reaped:
                self._condition.notify_all()
            return tuple(reaped)

    def is_active(self, lease: str | GpuLease) -> bool:
        self._ensure_current_process()
        lease_id = _lease_id(lease)
        with self._condition:
            if lease_id not in self._leases:
                return False
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
                return (
                    lease_id in self._leases
                    and lease_id in shared_records
                )

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable host-wide accounting report."""

        self._ensure_current_process()
        snapshots = self._snapshots_locked()
        with self._condition:
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
                accounting = self._account_locked(snapshots, shared_records)
                records = list(shared_records.values())
            now = time.monotonic()
            return {
                "registry_path": str(self._registry_path),
                "reserve_policy": {
                    "floor_mib": self._reserve_floor_mib,
                    "fraction": self._reserve_fraction,
                },
                "snapshot_error": self._last_snapshot_error,
                "gpus": [
                    {
                        "gpu_id": item.snapshot.index,
                        "uuid": item.snapshot.uuid,
                        "total_mib": item.snapshot.total_mib,
                        "raw_used_mib": item.snapshot.used_mib,
                        "reserve_mib": item.reserve_mib,
                        "external_used_mib": item.external_used_mib,
                        "committed_mib": item.committed_mib,
                        "managed_reported_mib": item.managed_reported_mib,
                        "managed_overage_mib": item.managed_overage_mib,
                        "misplaced_managed_mib": item.misplaced_managed_mib,
                        "accounted_used_mib": item.accounted_used_mib,
                        "allocatable_free_mib": item.allocatable_free_mib,
                        "lease_ids": list(item.lease_ids),
                    }
                    for item in accounting
                ],
                "leases": [
                    {
                        "lease_id": record.lease_id,
                        "service": record.service,
                        "gpu_id": record.gpu_id,
                        "required_mib": record.required_mib,
                        "owner_pid": record.owner_pid,
                        "pid": record.pid,
                        "usage_pids": sorted(record.usage_pids),
                        "age_seconds": max(
                            0.0, now - record.acquired_monotonic
                        ),
                        "acquired_at": record.acquired_unix,
                        "metadata": dict(record.metadata),
                    }
                    for record in sorted(
                        records, key=lambda item: item.acquired_monotonic
                    )
                ],
            }

    def _lease_pid(self, lease_id: str) -> int | None:
        self._ensure_current_process()
        with self._condition:
            if lease_id not in self._leases:
                return None
            with self._shared_records_locked() as shared_records:
                self._reap_records_locked(shared_records)
                self._sync_local_records_locked(shared_records)
                record = self._leases.get(lease_id)
                return None if record is None else record.pid

    def _snapshots_locked(self) -> list[GpuSnapshot]:
        try:
            raw_snapshots = list(self._snapshot_provider())
            snapshots = [_normalize_snapshot(item) for item in raw_snapshots]
            self._last_snapshot_error = None
        except Exception as exc:  # A monitor failure must never over-assign a GPU.
            self._last_snapshot_error = f"{type(exc).__name__}: {exc}"
            return []
        # Duplicate indices make placement ambiguous, so fail this snapshot closed.
        indices = [item.index for item in snapshots]
        if len(indices) != len(set(indices)):
            self._last_snapshot_error = "duplicate GPU indices in snapshot"
            return []
        return sorted(snapshots, key=lambda item: _gpu_sort_key(item.index))

    def _account_locked(
        self,
        snapshots: Sequence[GpuSnapshot],
        lease_records: Mapping[str, _LeaseRecord] | None = None,
    ) -> list[_GpuAccounting]:
        records = self._leases if lease_records is None else lease_records
        managed_pids = {
            pid
            for record in records.values()
            for pid in record.usage_pids
        }
        process_usage: dict[str, dict[int, int]] = {}
        for snapshot in snapshots:
            per_gpu: dict[int, int] = {}
            for process in snapshot.processes:
                per_gpu[process.pid] = per_gpu.get(process.pid, 0) + process.used_mib
            process_usage[snapshot.index] = per_gpu

        accounting: list[_GpuAccounting] = []
        for snapshot in snapshots:
            per_gpu = process_usage[snapshot.index]
            managed_reported = sum(
                used for pid, used in per_gpu.items() if pid in managed_pids
            )
            external_used = max(0, snapshot.used_mib - managed_reported)
            assigned = [
                record
                for record in records.values()
                if record.gpu_id == snapshot.index
            ]
            committed = sum(record.required_mib for record in assigned)
            overage = 0
            for record in assigned:
                if not record.usage_pids:
                    continue
                reported = sum(per_gpu.get(pid, 0) for pid in record.usage_pids)
                overage += max(0, reported - record.required_mib)

            # If a bound process violates its assignment, count that real memory on
            # the unexpected GPU in addition to keeping its expected reservation on
            # the assigned GPU.
            assigned_pids = {
                pid
                for record in assigned
                for pid in record.usage_pids
            }
            misplaced = sum(
                used
                for pid, used in per_gpu.items()
                if pid in managed_pids and pid not in assigned_pids
            )
            accounted_used = external_used + committed + overage + misplaced
            reserve = self.reserve_for_total(snapshot.total_mib)
            allocatable_free = max(
                0, snapshot.total_mib - reserve - accounted_used
            )
            accounting.append(
                _GpuAccounting(
                    snapshot=snapshot,
                    reserve_mib=reserve,
                    external_used_mib=external_used,
                    committed_mib=committed,
                    managed_reported_mib=managed_reported,
                    managed_overage_mib=overage,
                    misplaced_managed_mib=misplaced,
                    accounted_used_mib=accounted_used,
                    allocatable_free_mib=allocatable_free,
                    lease_ids=tuple(record.lease_id for record in assigned),
                )
            )
        return accounting

    def _unavailable_reason(
        self,
        required_mib: int,
        allowed: set[str] | None,
        accounting: Sequence[_GpuAccounting],
    ) -> str:
        if not accounting:
            detail = self._last_snapshot_error or "no GPU snapshot available"
            return f"cannot reserve {required_mib} MiB: {detail}"
        candidates = [
            item for item in accounting if allowed is None or item.snapshot.index in allowed
        ]
        if not candidates:
            return (
                f"cannot reserve {required_mib} MiB: no allowed GPU is present "
                f"(allowed={sorted(allowed or set())})"
            )
        available = ", ".join(
            f"GPU {item.snapshot.index}={item.allocatable_free_mib} MiB"
            for item in candidates
        )
        return f"cannot reserve {required_mib} MiB; allocatable free: {available}"


def _lease_id(lease: str | GpuLease) -> str:
    if isinstance(lease, GpuLease):
        return lease.lease_id
    lease_id = str(lease).strip()
    if not lease_id:
        raise ValueError("lease id must not be empty")
    return lease_id


def _gpu_sort_key(index: str) -> tuple[int, int | str]:
    try:
        return (0, int(index))
    except (TypeError, ValueError):
        return (1, str(index))


def _normalize_snapshot(snapshot: object) -> GpuSnapshot:
    if isinstance(snapshot, GpuSnapshot):
        return snapshot
    # Accept simple injected/legacy objects with index/total_mib/used_mib fields.
    try:
        index = str(getattr(snapshot, "index"))
        total_mib = int(getattr(snapshot, "total_mib"))
        used_mib = int(getattr(snapshot, "used_mib"))
    except Exception as exc:
        raise TypeError(f"invalid GPU snapshot: {snapshot!r}") from exc
    raw_processes = getattr(snapshot, "processes", ())
    processes: list[GpuProcessUsage] = []
    if isinstance(raw_processes, Mapping):
        processes.extend(
            GpuProcessUsage(pid=int(pid), used_mib=int(process_mib))
            for pid, process_mib in raw_processes.items()
        )
    else:
        for process in raw_processes:
            if isinstance(process, GpuProcessUsage):
                processes.append(process)
            else:
                processes.append(
                    GpuProcessUsage(
                        pid=int(getattr(process, "pid")),
                        used_mib=int(getattr(process, "used_mib")),
                    )
                )
    return GpuSnapshot(
        index=index,
        uuid=getattr(snapshot, "uuid", None),
        total_mib=total_mib,
        used_mib=used_mib,
        processes=tuple(processes),
    )


_default_manager: GpuLeaseManager | None = None
_default_manager_lock = threading.Lock()


def _reset_default_manager_after_fork() -> None:
    global _default_manager_lock
    _default_manager_lock = threading.Lock()
    if _default_manager is not None:
        _default_manager._reset_for_current_process()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_default_manager_after_fork)


def get_default_gpu_lease_manager() -> GpuLeaseManager:
    """Return the process-wide manager used by launchers in this server."""

    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = GpuLeaseManager()
        return _default_manager


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_RESERVE_FLOOR_MIB",
    "DEFAULT_RESERVE_FRACTION",
    "GpuLease",
    "GpuLeaseError",
    "GpuLeaseManager",
    "GpuLeaseTimeout",
    "GpuProcessUsage",
    "GpuSnapshot",
    "GpuUnavailableError",
    "UnknownGpuLeaseError",
    "get_default_gpu_lease_manager",
    "pid_is_alive",
    "pid_start_identity",
    "query_gpu_snapshots",
]
