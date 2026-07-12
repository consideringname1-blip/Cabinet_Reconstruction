from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import foundationpose_dispatcher as dispatcher_module  # noqa: E402
from foundationpose_dispatcher import (  # noqa: E402
    PRIORITY_HOLOLENS,
    PRIORITY_REALTIME_TRACKING,
    FoundationPoseDispatcher,
    _QueuedRequest,
)


def _response(sock: socket.socket) -> dict:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))


class FoundationPoseDispatcherTests(unittest.TestCase):
    def _dispatcher(self) -> FoundationPoseDispatcher:
        root = Path(tempfile.gettempdir())
        return FoundationPoseDispatcher(socket_path=root / "fp-dispatch-test.sock", backend_socket_path=root / "fp-backend-test.sock")

    def _start_worker(self, dispatcher: FoundationPoseDispatcher) -> threading.Thread:
        dispatcher._stopped.clear()
        thread = threading.Thread(target=dispatcher._worker_loop, daemon=True)
        dispatcher._worker_thread = thread
        thread.start()
        return thread

    def _stop_worker(self, dispatcher: FoundationPoseDispatcher, thread: threading.Thread) -> None:
        dispatcher._stopped.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_hololens_priority_precedes_queued_realtime_request(self) -> None:
        dispatcher = self._dispatcher()
        realtime_server, realtime_client = socket.socketpair()
        hololens_server, hololens_client = socket.socketpair()
        self.addCleanup(realtime_client.close)
        self.addCleanup(hololens_client.close)
        dispatcher._queue.put(
            _QueuedRequest(PRIORITY_REALTIME_TRACKING, 1, realtime_server, {"name": "realtime"})
        )
        dispatcher._queue.put(_QueuedRequest(PRIORITY_HOLOLENS, 2, hololens_server, {"name": "hololens"}))
        calls: list[str] = []

        with patch.object(
            dispatcher_module,
            "request_socket",
            side_effect=lambda _path, payload: calls.append(payload["name"]) or {"ok": True},
        ):
            thread = self._start_worker(dispatcher)
            hololens_response = _response(hololens_client)
            realtime_response = _response(realtime_client)
            self._stop_worker(dispatcher, thread)

        self.assertEqual(["hololens", "realtime"], calls)
        self.assertTrue(hololens_response["ok"])
        self.assertTrue(realtime_response["ok"])

    def test_waiting_request_with_same_key_is_superseded(self) -> None:
        dispatcher = self._dispatcher()
        old_server, old_client = socket.socketpair()
        new_server, new_client = socket.socketpair()
        self.addCleanup(old_client.close)
        self.addCleanup(new_client.close)
        dispatcher._coalesce_generation["tracking:a"] = 2
        dispatcher._queue.put(
            _QueuedRequest(
                PRIORITY_REALTIME_TRACKING,
                1,
                old_server,
                {"name": "old"},
                coalesce_key="tracking:a",
                generation=1,
            )
        )
        dispatcher._queue.put(
            _QueuedRequest(
                PRIORITY_REALTIME_TRACKING,
                2,
                new_server,
                {"name": "new"},
                coalesce_key="tracking:a",
                generation=2,
            )
        )
        calls: list[str] = []

        with patch.object(
            dispatcher_module,
            "request_socket",
            side_effect=lambda _path, payload: calls.append(payload["name"]) or {"ok": True},
        ):
            thread = self._start_worker(dispatcher)
            old_response = _response(old_client)
            new_response = _response(new_client)
            self._stop_worker(dispatcher, thread)

        self.assertEqual({"ok": False, "error": "SUPERSEDED", "superseded": True}, old_response)
        self.assertEqual(["new"], calls)
        self.assertTrue(new_response["ok"])

    def test_running_backend_finishes_but_result_is_rejected_if_newer_request_arrived(self) -> None:
        dispatcher = self._dispatcher()
        server_sock, client_sock = socket.socketpair()
        self.addCleanup(client_sock.close)
        dispatcher._coalesce_generation["tracking:a"] = 1
        dispatcher._queue.put(
            _QueuedRequest(
                PRIORITY_REALTIME_TRACKING,
                1,
                server_sock,
                {"name": "running"},
                coalesce_key="tracking:a",
                generation=1,
            )
        )
        started = threading.Event()
        release = threading.Event()

        def backend(_path, _payload):
            started.set()
            self.assertTrue(release.wait(timeout=2.0))
            return {"ok": True, "result": {"pose": "old"}}

        with patch.object(dispatcher_module, "request_socket", side_effect=backend):
            thread = self._start_worker(dispatcher)
            self.assertTrue(started.wait(timeout=2.0))
            dispatcher._coalesce_generation["tracking:a"] = 2
            release.set()
            response = _response(client_sock)
            self._stop_worker(dispatcher, thread)

        self.assertEqual("SUPERSEDED", response["error"])
        self.assertTrue(response["superseded"])


if __name__ == "__main__":
    unittest.main()
