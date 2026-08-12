"""JSON sidecar metadata for IQ recordings, and the RecordingResult returned
to callers.

The metadata schema intentionally includes a few fields beyond the required
minimum (``simulated``, ``expected_file_size``, ``command``) because they
are cheap to record and materially help the future signal-processing/ML
stages tell a real recording from a simulated one, and help diagnose a
short/incomplete capture after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .states import RecordingStatus


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Render a datetime as an ISO-8601 UTC string; naive datetimes are assumed UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class RecordingMetadata:
    """JSON sidecar written alongside every IQ recording attempt."""

    satellite_name: str
    frequency_hz: int
    sample_rate: int
    gain: Optional[float]
    output_filename: Optional[str]
    recording_status: str

    norad_id: Optional[int] = None
    scheduled_aos: Optional[str] = None
    scheduled_los: Optional[str] = None
    actual_recording_start: Optional[str] = None
    actual_recording_stop: Optional[str] = None
    pre_buffer_seconds: float = 0.0
    post_buffer_seconds: float = 0.0
    recording_duration_seconds: Optional[float] = None
    output_file_size: Optional[int] = None
    expected_file_size: Optional[int] = None
    error_message: Optional[str] = None
    simulated: bool = False
    device_index: int = 0
    command: Optional[List[str]] = None
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path) -> "RecordingMetadata":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class RecordingResult:
    """What every public recording call returns - never raises for expected failures."""

    status: RecordingStatus
    output_file: Optional[str] = None
    metadata_file: Optional[str] = None
    error_message: Optional[str] = None
    actual_recording_start: Optional[str] = None
    actual_recording_stop: Optional[str] = None
    recording_duration_seconds: Optional[float] = None
    output_file_size: Optional[int] = None
    metadata: Optional[RecordingMetadata] = None

    @property
    def success(self) -> bool:
        return self.status == RecordingStatus.SUCCESS
