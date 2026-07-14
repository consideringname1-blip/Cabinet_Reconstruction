from __future__ import annotations

import heapq
import json
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PRIORITY_HOLOLENS = 0
PRIORITY_REALTIME_TRACKING = 10


def _read_request(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        return {}
    return json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))


def _send_response(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))


def request_socket(socket_path: str | Path, payload: dict[str, Any], *, timeout: float = 3600.0) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("FoundationPose backend returned no response")
    return json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))


@dataclass(order=True)
class _QueuedRequest:
    priority: int
    sequence: int
    conn: socket.socket = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    coalesce_key: str | None = field(compare=False, default=None)
    generation: int = field(compare=False, default=0)
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)


class FoundationPoseDispatcher:
    """Non-preemptive priority pool in front of one to four FP backends.

    Every backend consumes the same priority heap. Coalesced requests for the
    same object are serialized even when multiple backends are idle. Enqueuing
    a newer generation supersedes older queued work immediately; a generation
    already running may finish in the backend, but its result is discarded.
    """

    def __init__(self, *, socket_path: Path, backend_socket_paths: Sequence[Path]) -> None:
        if not isinstance(socket_path, Path):
            raise TypeError("socket_path must be pathlib.Path")
        if not isinstance(backend_socket_paths, Sequence):
            raise TypeError("backend_socket_paths must be a sequence of pathlib.Path")
        raw_backend_paths = tuple(backend_socket_paths)
        if not 1 <= len(raw_backend_paths) <= 4:
            raise ValueError("FoundationPose dispatcher requires between 1 and 4 backends")
        if any(not isinstance(path, Path) for path in raw_backend_paths):
            raise TypeError("every backend_socket_paths item must be pathlib.Path")

        self.socket_path = socket_path.expanduser().resolve()
        self.backend_socket_paths = tuple(path.expanduser().resolve() for path in raw_backend_paths)
        if len(set(self.backend_socket_paths)) != len(self.backend_socket_paths):
            raise ValueError("backend_socket_paths must be unique")
        if self.socket_path in self.backend_socket_paths:
            raise ValueError("dispatcher socket_path cannot also be a backend socket path")

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._heap: list[_QueuedRequest] = []
        self._sequence = 0
        self._coalesce_generation: dict[str, int] = {}
        self._busy_keys: set[str] = set()
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._worker_threads: list[threading.Thread] = []
        self._backend_states = [self._new_backend_state() for _ in self.backend_socket_paths]
        self._stopped = threading.Event()

    @staticmethod
    def _new_backend_state() -> dict[str, Any]:
        return {
            "busy": False,
            "priority": None,
            "sequence": None,
            "coalesce_key": None,
            "generation": None,
            "started_at_monotonic": None,
            "completed": 0,
            "last_error": None,
        }

    def start(self) -> None:
        with self._condition:
            if self._accept_thread is not None and self._accept_thread.is_alive():
                return
            if any(thread.is_alive() for thread in self._worker_threads):
                # Running backend calls are deliberately non-preemptive.
                raise RuntimeError("FoundationPose dispatcher workers are still stopping")
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
            self._heap.clear()
            self._sequence = 0
            self._busy_keys.clear()
            self._coalesce_generation.clear()
            self._backend_states = [self._new_backend_state() for _ in self.backend_socket_paths]
            self._stopped.clear()

            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(self.socket_path))
            server.listen(32)
            server.settimeout(0.5)
            self._server = server
            self._accept_thread = threading.Thread(
                target=self._accept_loop,
                daemon=True,
                name="foundationpose-dispatcher-accept",
            )
            self._worker_threads = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(backend_index,),
                    daemon=True,
                    name=f"foundationpose-dispatcher-worker-{backend_index}",
                )
                for backend_index in range(len(self.backend_socket_paths))
            ]
            self._accept_thread.start()
            for thread in self._worker_threads:
                thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopped.set()
            server = self._server
            self._server = None
            self._condition.notify_all()
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

        deadline = time.monotonic() + 2.0
        for thread in [self._accept_thread, *self._worker_threads]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._condition:
            pending = list(self._heap)
            self._heap.clear()
            self._condition.notify_all()
            if self._accept_thread is None or not self._accept_thread.is_alive():
                self._accept_thread = None
        for item in pending:
            self._reply_and_close(item, {"ok": False, "error": "dispatcher_stopped"})
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _accept_loop(self) -> None:
        while not self._stopped.is_set():
            server = self._server
            if server is None:
                break
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._receive_one, args=(conn,), daemon=True).start()

    def _receive_one(self, conn: socket.socket) -> None:
        try:
            payload = _read_request(conn)
            if payload.get("action") == "dispatcher_status":
                _send_response(conn, {"ok": True, "status": self.status()})
                conn.close()
                return

            priority = int(payload.pop("dispatcher_priority", PRIORITY_HOLOLENS))
            coalesce_key = str(payload.pop("dispatcher_coalesce_key", "") or "").strip() or None
            with self._condition:
                if self._stopped.is_set():
                    stopped = True
                    superseded: list[_QueuedRequest] = []
                else:
                    stopped = False
                    self._sequence += 1
                    generation = 0
                    if coalesce_key:
                        generation = int(self._coalesce_generation.get(coalesce_key, 0)) + 1
                        self._coalesce_generation[coalesce_key] = generation
                    item = _QueuedRequest(
                        priority=priority,
                        sequence=self._sequence,
                        conn=conn,
                        payload=payload,
                        coalesce_key=coalesce_key,
                        generation=generation,
                    )
                    superseded = self._remove_queued_key_locked(coalesce_key) if coalesce_key else []
                    heapq.heappush(self._heap, item)
                    self._condition.notify_all()

            for old_item in superseded:
                self._reply_and_close(old_item, self._superseded_response(old_item))
            if stopped:
                _send_response(conn, {"ok": False, "error": "dispatcher_stopped"})
                conn.close()
            # A selected worker owns the connection after enqueue.
        except Exception as exc:
            try:
                _send_response(conn, {"ok": False, "error": str(exc)})
            finally:
                conn.close()

    def _remove_queued_key_locked(self, coalesce_key: str) -> list[_QueuedRequest]:
        superseded = [item for item in self._heap if item.coalesce_key == coalesce_key]
        if superseded:
            self._heap = [item for item in self._heap if item.coalesce_key != coalesce_key]
            heapq.heapify(self._heap)
        return superseded

    def _is_superseded_locked(self, item: _QueuedRequest) -> bool:
        return bool(
            item.coalesce_key
            and self._coalesce_generation.get(item.coalesce_key) != item.generation
        )

    @staticmethod
    def _remove_heap_index(heap: list[_QueuedRequest], index: int) -> _QueuedRequest:
        item = heap[index]
        last = heap.pop()
        if index < len(heap):
            heap[index] = last
            heapq.heapify(heap)
        return item

    def _claim_next(self, backend_index: int) -> tuple[_QueuedRequest, float] | None:
        while True:
            stale: _QueuedRequest | None = None
            with self._condition:
                if self._stopped.is_set():
                    return None
                if not self.backend_socket_paths[backend_index].exists():
                    self._condition.wait(timeout=0.2)
                    continue

                stale_indexes = [
                    index for index, item in enumerate(self._heap) if self._is_superseded_locked(item)
                ]
                if stale_indexes:
                    stale_index = min(stale_indexes, key=lambda index: self._heap[index])
                    stale = self._remove_heap_index(self._heap, stale_index)
                else:
                    runnable_indexes = [
                        index
                        for index, item in enumerate(self._heap)
                        if item.coalesce_key is None or item.coalesce_key not in self._busy_keys
                    ]
                    if not runnable_indexes:
                        self._condition.wait(timeout=0.5)
                        continue
                    selected_index = min(runnable_indexes, key=lambda index: self._heap[index])
                    item = self._remove_heap_index(self._heap, selected_index)
                    if item.coalesce_key:
                        self._busy_keys.add(item.coalesce_key)
                    started_at = time.monotonic()
                    self._backend_states[backend_index].update(
                        {
                            "busy": True,
                            "priority": item.priority,
                            "sequence": item.sequence,
                            "coalesce_key": item.coalesce_key,
                            "generation": item.generation,
                            "started_at_monotonic": started_at,
                            "last_error": None,
                        }
                    )
                    return item, (started_at - item.enqueued_at) * 1000.0

            if stale is not None:
                self._reply_and_close(stale, self._superseded_response(stale))

    def _worker_loop(self, backend_index: int) -> None:
        backend_socket_path = self.backend_socket_paths[backend_index]
        while not self._stopped.is_set():
            # The configured pool size is an upper bound. Optional backends
            # may be started later when GPU capacity becomes available; a
            # worker without a live socket must not consume (and fail) queued
            # work while waiting for that backend.
            if not backend_socket_path.exists():
                self._stopped.wait(timeout=0.2)
                continue
            claimed = self._claim_next(backend_index)
            if claimed is None:
                return
            item, queue_wait_ms = claimed
            execution_started = time.monotonic()
            response: dict[str, Any] | None = None
            backend_error: Exception | None = None
            try:
                response = request_socket(backend_socket_path, item.payload)
                if not isinstance(response, dict):
                    raise TypeError("FoundationPose backend response must be a JSON object")
            except Exception as exc:  # Result policy is decided after releasing the key.
                backend_error = exc

            execution_ms = (time.monotonic() - execution_started) * 1000.0
            with self._condition:
                superseded = self._is_superseded_locked(item)
                stopped = self._stopped.is_set()
                if item.coalesce_key:
                    self._busy_keys.discard(item.coalesce_key)
                state = self._backend_states[backend_index]
                completed = int(state["completed"]) + 1
                state.update(self._new_backend_state())
                state["completed"] = completed
                state["last_error"] = str(backend_error) if backend_error is not None else None
                self._condition.notify_all()

            if stopped:
                result = {"ok": False, "error": "dispatcher_stopped"}
            elif superseded:
                result = self._superseded_response(item)
            elif backend_error is not None:
                result = {"ok": False, "error": str(backend_error)}
            else:
                assert response is not None
                result = response
                result["dispatcher"] = {
                    "backend_index": backend_index,
                    "backend_socket_path": str(backend_socket_path),
                    "priority": item.priority,
                    "queue_wait_ms": queue_wait_ms,
                    "execution_ms": execution_ms,
                }
            self._reply_and_close(item, result)

    @staticmethod
    def _superseded_response(item: _QueuedRequest) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "SUPERSEDED",
            "superseded": True,
            "dispatcher": {
                "priority": item.priority,
                "sequence": item.sequence,
                "coalesce_key": item.coalesce_key,
                "generation": item.generation,
            },
        }

    @staticmethod
    def _reply_and_close(item: _QueuedRequest, payload: dict[str, Any]) -> None:
        try:
            _send_response(item.conn, payload)
        except OSError:
            pass
        finally:
            try:
                item.conn.close()
            except OSError:
                pass

    def status(self) -> dict[str, Any]:
        with self._condition:
            worker_threads = list(self._worker_threads)
            backends: list[dict[str, Any]] = []
            for index, backend_socket_path in enumerate(self.backend_socket_paths):
                state = dict(self._backend_states[index])
                started_at = state.pop("started_at_monotonic")
                state["backend_index"] = index
                state["socket_path"] = str(backend_socket_path)
                state["socket_ready"] = backend_socket_path.exists()
                state["worker_alive"] = bool(
                    index < len(worker_threads) and worker_threads[index].is_alive()
                )
                state["running_ms"] = (
                    (time.monotonic() - float(started_at)) * 1000.0
                    if started_at is not None
                    else None
                )
                backends.append(state)
            return {
                "socket_path": str(self.socket_path),
                "backend_socket_paths": [str(path) for path in self.backend_socket_paths],
                "worker_count": len(self.backend_socket_paths),
                "queued": len(self._heap),
                "queued_by_priority": {
                    str(priority): sum(1 for item in self._heap if item.priority == priority)
                    for priority in sorted({item.priority for item in self._heap})
                },
                "busy_keys": sorted(self._busy_keys),
                "coalesce_keys": len(self._coalesce_generation),
                "accepting": bool(
                    not self._stopped.is_set()
                    and self._accept_thread is not None
                    and self._accept_thread.is_alive()
                ),
                "backends": backends,
            }
