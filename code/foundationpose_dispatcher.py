from __future__ import annotations

import json
import queue
import socket
import threading
import time
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
    """Priority proxy in front of the single non-preemptive FP worker."""

    def __init__(self, *, socket_path: Path, backend_socket_path: Path) -> None:
        self.socket_path = socket_path.expanduser().resolve()
        self.backend_socket_path = backend_socket_path.expanduser().resolve()
        self._queue: "queue.PriorityQueue[_QueuedRequest]" = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._sequence = 0
        self._coalesce_generation: dict[str, int] = {}
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._accept_thread is not None and self._accept_thread.is_alive():
                return
            if self._worker_thread is not None and self._worker_thread.is_alive():
                # A non-preemptive backend request cannot be killed safely.
                # Starting another worker here would violate the single-FP
                # serialization guarantee.
                raise RuntimeError("FoundationPose dispatcher worker is still stopping")
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
            self._stopped.clear()
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(self.socket_path))
            server.listen(32)
            server.settimeout(0.5)
            self._server = server
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="foundationpose-dispatcher-accept")
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="foundationpose-dispatcher-worker")
            self._accept_thread.start()
            self._worker_thread.start()

    def stop(self) -> None:
        self._stopped.set()
        server = self._server
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        self._server = None
        for thread in (self._accept_thread, self._worker_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        if self._accept_thread is None or not self._accept_thread.is_alive():
            self._accept_thread = None
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = None
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                _send_response(item.conn, {"ok": False, "error": "dispatcher_stopped"})
            except OSError:
                pass
            item.conn.close()
            self._queue.task_done()
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
            if self._stopped.is_set():
                _send_response(conn, {"ok": False, "error": "dispatcher_stopped"})
                conn.close()
                return
            if payload.get("action") == "dispatcher_status":
                _send_response(conn, {"ok": True, "status": self.status()})
                conn.close()
                return
            priority = int(payload.pop("dispatcher_priority", PRIORITY_HOLOLENS))
            coalesce_key = str(payload.pop("dispatcher_coalesce_key", "") or "").strip() or None
            with self._lock:
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
                self._queue.put(item)
        except Exception as exc:
            try:
                _send_response(conn, {"ok": False, "error": str(exc)})
            finally:
                conn.close()

    def _is_superseded(self, item: _QueuedRequest) -> bool:
        if not item.coalesce_key:
            return False
        with self._lock:
            return self._coalesce_generation.get(item.coalesce_key) != item.generation

    def _worker_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._stopped.is_set():
                    _send_response(item.conn, {"ok": False, "error": "dispatcher_stopped"})
                    continue
                if self._is_superseded(item):
                    _send_response(item.conn, {"ok": False, "error": "SUPERSEDED", "superseded": True})
                    continue
                response = request_socket(self.backend_socket_path, item.payload)
                if self._stopped.is_set():
                    _send_response(item.conn, {"ok": False, "error": "dispatcher_stopped"})
                elif self._is_superseded(item):
                    _send_response(item.conn, {"ok": False, "error": "SUPERSEDED", "superseded": True})
                else:
                    response["dispatcher"] = {
                        "priority": item.priority,
                        "queue_wait_ms": (time.monotonic() - item.enqueued_at) * 1000.0,
                    }
                    _send_response(item.conn, response)
            except Exception as exc:
                try:
                    _send_response(item.conn, {"ok": False, "error": str(exc)})
                except OSError:
                    pass
            finally:
                try:
                    item.conn.close()
                except OSError:
                    pass
                self._queue.task_done()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "socket_path": str(self.socket_path),
                "backend_socket_path": str(self.backend_socket_path),
                "queued": self._queue.qsize(),
                "coalesce_keys": len(self._coalesce_generation),
                "running": bool(self._worker_thread and self._worker_thread.is_alive()),
            }
