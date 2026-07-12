from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Importing the recorder normally re-execs into the ROS Python environment.
# Unit tests exercise its pure payload/join helpers without requiring ROS.
os.environ["SHIGURE_HISTORY_RECORDER_BOOTSTRAPPED"] = "1"

from stages.shigure_history.cache import RosStamp, ShigureMemoryStore, store_request  # noqa: E402
from stages.shigure_history import settings as shigure_settings  # noqa: E402
from stages.shigure_history.run_shigure_history_recorder import (  # noqa: E402
    TopicState,
    append_correlated_event,
)


def _header(sec: int, nanosec: int) -> SimpleNamespace:
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id="camera_color_optical_frame")


def _bbox(x: float, y: float, width: float, height: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, width=width, height=height)


def _detected_list(sec: int, nanosec: int, *, with_object: bool = True) -> SimpleNamespace:
    objects = []
    if with_object:
        objects.append(
            SimpleNamespace(
                action="obj_move",
                bounding_box=_bbox(10.0, 20.0, 30.0, 40.0),
                mask=SimpleNamespace(data=b"mask-png-bytes", format="png"),
            )
        )
    return SimpleNamespace(header=_header(sec, nanosec), object_list=objects)


def _contacted_list(sec: int, nanosec: int, *, with_contact: bool = True) -> SimpleNamespace:
    contacts = []
    if with_contact:
        contacts.append(
            SimpleNamespace(
                event_id="20260712010101_1",
                people_id="20260712010100_2",
                object_id="20260712010059_3",
                action="obj_move",
                people_bounding_box=_bbox(1.0, 2.0, 100.0, 200.0),
                object_bounding_box=_bbox(10.0, 20.0, 30.0, 40.0),
                object_cube=SimpleNamespace(x=0.1, y=0.2, z=0.3, width=0.4, height=0.5, depth=0.6),
            )
        )
    return SimpleNamespace(header=_header(sec, nanosec), contacted_list=contacts)


def _states() -> dict[str, TopicState]:
    return {
        "object_detection": TopicState(
            key="object_detection",
            topic="/shigure/object_detection",
            type_name="shigure_core_msgs/msg/DetectedObjectList",
            maxlen=32,
        ),
        "contacted": TopicState(
            key="contacted",
            topic="/shigure/contacted",
            type_name="shigure_core_msgs/msg/ContactedList",
            maxlen=32,
        ),
    }


class ShigureEventJoinTests(unittest.TestCase):
    def test_subscription_qos_contract_is_best_effort_for_remote_shigure_publishers(self) -> None:
        self.assertEqual(
            frozenset(shigure_settings.TOPIC_SPECS),
            shigure_settings.BEST_EFFORT_TOPIC_KEYS,
        )
        self.assertEqual(
            frozenset({"object_detection", "contacted"}),
            shigure_settings.CORRELATED_EVENT_TOPIC_KEYS,
        )
        self.assertFalse(hasattr(shigure_settings, "RELIABLE_TOPIC_KEYS"))

    def test_full_contact_payload_and_exact_frame_match(self) -> None:
        store = ShigureMemoryStore(max_seconds=60.0, max_samples=10, max_events=20)
        states = _states()
        stamp = RosStamp(100, 123)

        states["object_detection"].append(_detected_list(stamp.sec, stamp.nanosec))
        states["contacted"].append(_contacted_list(stamp.sec, stamp.nanosec))
        event = append_correlated_event(store, states, stamp)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.source_stamp, stamp)
        self.assertEqual(event.contacted_state, "present")
        self.assertEqual(event.object_detection_state, "present")
        self.assertGreater(event.received_monotonic, 0.0)
        self.assertTrue(event.received_utc)

        contact = (event.contacted or {})["contacts"][0]
        self.assertEqual(contact["event_id"], "20260712010101_1")
        self.assertEqual(contact["people_id"], "20260712010100_2")
        self.assertEqual(contact["object_id"], "20260712010059_3")
        self.assertEqual(contact["people_bounding_box"]["xyxy"], [1.0, 2.0, 101.0, 202.0])
        self.assertEqual(contact["object_cube"]["depth"], 0.6)

        match = (event.contact_object_matches or [])[0]
        self.assertEqual(match["status"], "matched_action_iou")
        self.assertEqual(match["best"]["object_detection_id"], "obj_move:0")
        self.assertAlmostEqual(match["best"]["bbox_iou"], 1.0)

    def test_explicit_empty_contact_is_not_missing_contact(self) -> None:
        store = ShigureMemoryStore(max_seconds=60.0, max_samples=10, max_events=20)
        states = _states()
        stamp = RosStamp(101, 0)

        states["object_detection"].append(_detected_list(stamp.sec, stamp.nanosec))
        missing = append_correlated_event(store, states, stamp)
        self.assertIsNotNone(missing)
        assert missing is not None
        self.assertEqual(missing.contacted_state, "missing")
        missing_sequence = missing.sequence

        states["contacted"].append(_contacted_list(stamp.sec, stamp.nanosec, with_contact=False))
        explicit_empty = append_correlated_event(store, states, stamp)
        self.assertIsNotNone(explicit_empty)
        assert explicit_empty is not None
        self.assertEqual(explicit_empty.contacted_state, "explicit_empty")
        self.assertEqual((explicit_empty.contacted or {})["contact_count"], 0)
        self.assertTrue((explicit_empty.contacted or {})["explicit_empty"])
        self.assertGreater(explicit_empty.sequence, missing_sequence)
        updates = store.iter_event_updates_after(missing_sequence)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].source_stamp, stamp)
        self.assertEqual(updates[0].contacted_state, "explicit_empty")

    def test_join_never_uses_nearest_timestamp(self) -> None:
        store = ShigureMemoryStore(max_seconds=60.0, max_samples=10, max_events=20)
        states = _states()
        object_stamp = RosStamp(102, 10)
        contact_stamp = RosStamp(102, 11)

        states["object_detection"].append(_detected_list(object_stamp.sec, object_stamp.nanosec))
        states["contacted"].append(_contacted_list(contact_stamp.sec, contact_stamp.nanosec))
        object_event = append_correlated_event(store, states, object_stamp)
        contact_event = append_correlated_event(store, states, contact_stamp)

        assert object_event is not None and contact_event is not None
        self.assertEqual(object_event.contacted_state, "missing")
        self.assertEqual(contact_event.object_detection_state, "missing")
        self.assertEqual(len(list(store.iter_events())), 2)

    def test_socket_event_payload_is_lightweight_by_default(self) -> None:
        store = ShigureMemoryStore(max_seconds=60.0, max_samples=10, max_events=20)
        states = _states()
        stamp = RosStamp(103, 0)
        states["object_detection"].append(_detected_list(stamp.sec, stamp.nanosec))
        states["contacted"].append(_contacted_list(stamp.sec, stamp.nanosec))
        append_correlated_event(store, states, stamp)

        lightweight = store_request(store, {"action": "latest_event"})
        full = store_request(store, {"action": "get_event", "stamp": stamp.to_dict(), "include_masks": True})
        light_object = lightweight["event"]["object_detection"]["objects"][0]
        full_object = full["event"]["object_detection"]["objects"][0]
        self.assertNotIn("mask_b64", light_object)
        self.assertIn("mask_b64", full_object)
        self.assertEqual(light_object["mask_bytes"], len(b"mask-png-bytes"))

    def test_empty_steady_state_frame_is_not_stored_as_event(self) -> None:
        store = ShigureMemoryStore(max_seconds=60.0, max_samples=10, max_events=20)
        states = _states()
        stamp = RosStamp(104, 0)
        states["object_detection"].append(_detected_list(stamp.sec, stamp.nanosec, with_object=False))
        states["contacted"].append(_contacted_list(stamp.sec, stamp.nanosec, with_contact=False))
        self.assertIsNone(append_correlated_event(store, states, stamp))
        self.assertEqual(store.status()["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
