from __future__ import annotations

from dataclasses import dataclass


DEPTH_SENSOR_AHAT = "AHAT"
DEPTH_SENSOR_LONGTHROW = "LONGTHROW"


@dataclass(frozen=True)
class DepthSensorLimits:
    sensor: str
    min_depth_mm: int
    max_reliable_depth_mm: int
    min_usable_depth_pixels: int
    max_upload_png_bytes: int
    upload_guard_enabled: bool


DEPTH_SENSOR_LIMITS = {
    DEPTH_SENSOR_AHAT: DepthSensorLimits(
        sensor=DEPTH_SENSOR_AHAT,
        min_depth_mm=200,
        max_reliable_depth_mm=1000,
        min_usable_depth_pixels=4096,
        max_upload_png_bytes=450000,
        upload_guard_enabled=False,
    ),
    DEPTH_SENSOR_LONGTHROW: DepthSensorLimits(
        sensor=DEPTH_SENSOR_LONGTHROW,
        min_depth_mm=200,
        max_reliable_depth_mm=7500,
        min_usable_depth_pixels=512,
        max_upload_png_bytes=450000,
        upload_guard_enabled=False,
    ),
}


def normalize_depth_sensor_name(value: str | None) -> str:
    sensor = str(value or "").strip()
    if sensor not in DEPTH_SENSOR_LIMITS:
        supported = ", ".join(sorted(DEPTH_SENSOR_LIMITS.keys()))
        raise ValueError(f"Unsupported depth sensor: {value!r}. Supported sensors: {supported}")
    return sensor


def get_depth_sensor_limits(value: str | None) -> DepthSensorLimits:
    return DEPTH_SENSOR_LIMITS[normalize_depth_sensor_name(value)]


def depth_sensor_limits_for_task(task: dict) -> DepthSensorLimits:
    return get_depth_sensor_limits((task.get("DepthCamera") or {}).get("sensor"))


def is_ahat_depth_sensor(value: str | None) -> bool:
    return normalize_depth_sensor_name(value) == DEPTH_SENSOR_AHAT


def is_longthrow_depth_sensor(value: str | None) -> bool:
    return normalize_depth_sensor_name(value) == DEPTH_SENSOR_LONGTHROW
