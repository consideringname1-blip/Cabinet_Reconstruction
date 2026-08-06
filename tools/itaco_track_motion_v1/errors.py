"""Structured failures for the stage-1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Failure:
    stage: str
    code: str
    message: str
    original_frame_id: int | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase1Error(RuntimeError):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


class ValidationErrors(Phase1Error):
    def __init__(self, failures: list[Failure]):
        self.failures = failures
        super().__init__(
            Failure(
                stage="frame_validation",
                code="manifest_validation_failed",
                message=f"Frame validation produced {len(failures)} error(s)",
                details={"errors": [item.to_dict() for item in failures]},
            )
        )


def require(condition: bool, *, stage: str, code: str, message: str,
            original_frame_id: int | None = None,
            details: dict[str, Any] | None = None) -> None:
    if not condition:
        raise Phase1Error(Failure(stage, code, message, original_frame_id, details))
