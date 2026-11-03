"""WEAVE Redis event protocol.

This module intentionally mirrors ``Weave/worker_scheduler/event.py``.  Keep
wire-format changes synchronized with that reference implementation.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Protocol

TENANT_EVENTS_CHANNEL = "tenant_events"


class RedisStorage(Protocol):
    def set(self, name: str, value: Any) -> Any: ...


class EventType(enum.Enum):
    CREATED = "created"
    RECOVERED = "recovered"
    COMPLETED = "completed"
    CRASHED = "crashed"
    DONE = "done"
    RELEASE_GPU = "release_gpu"
    READY = "ready"
    STATUS = "status"
    UNKNOWN = "unknown"

    @classmethod
    def match(cls, event_type: str) -> EventType:
        name_map = {key.lower(): value for key, value in cls.__members__.items()}
        return name_map.get(event_type, cls.UNKNOWN)


class EventLevel(enum.Enum):
    WORKER = "worker"
    PHASE = "phase"
    JOB = "job"


class Phase(enum.Enum):
    INIT = "init"
    GENERATE = "generate"
    TRAIN = "train"
    UPDATE = "update"

    @classmethod
    def match(cls, phase: str | None) -> Phase | None:
        name_map = {key.lower(): value for key, value in cls.__members__.items()}
        return name_map.get(phase) if phase is not None else None


@dataclass
class Event:
    job_name: str
    event_type: EventType
    phase: Phase | None = None
    worker_id: str | None = None
    value: Any | None = None

    @property
    def level(self) -> EventLevel:
        if self.worker_id is not None:
            return EventLevel.WORKER
        if self.phase is not None:
            return EventLevel.PHASE
        return EventLevel.JOB

    @property
    def storage_key(self) -> str:
        parts = [self.job_name]
        if self.phase is not None:
            parts.append(self.phase.value)
        if self.worker_id is not None:
            parts.append(str(self.worker_id))
        parts.append(self.event_type.value)
        return ":".join(parts)

    def execute(self, shared_storage: RedisStorage) -> None:
        shared_storage.set(self.storage_key, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "phase": self.phase,
            "worker_id": self.worker_id,
            "event_type": self.event_type,
            "value": self.value,
        }

    def __str__(self) -> str:
        parts = [self.job_name]
        if self.phase is not None:
            parts.append(self.phase.value)
        if self.worker_id is not None:
            parts.append(str(self.worker_id))
        suffix = self.event_type.value
        if self.value is not None:
            suffix += f"[{self.value}]"
        parts.append(suffix)
        return ":".join(parts)


class EventParser:
    """Parse ``job:(phase)?:(worker)?:event_type([value])?`` messages."""

    EVENT_PATTERNS = (
        re.compile(r"^(?P<job_name>[^:]+):(?P<phase>[^:]+):(?P<worker_id>[^:]+):(?P<event_type>[^:]+)$"),
        re.compile(r"^(?P<job_name>[^:]+):(?P<phase>[^:]+):(?P<event_type>[^:]+)$"),
        re.compile(r"^(?P<job_name>[^:]+):(?P<event_type>[^:]+)$"),
    )
    EVENT_INFO_PATTERN = re.compile(
        r"^(?P<event_type>[a-zA-Z_]+)(?:\[(?P<event_value>([a-zA-Z0-9_\s,\-]+)?[0-9,]+)\])?$"
    )

    @classmethod
    def parse(cls, event_key: str) -> Event | None:
        for pattern in cls.EVENT_PATTERNS:
            match = pattern.match(event_key)
            if match is None:
                continue
            groups = match.groupdict()
            value_match = cls.EVENT_INFO_PATTERN.match(groups["event_type"])
            if value_match is None:
                return None
            value_groups = value_match.groupdict()
            return Event(
                job_name=groups["job_name"],
                phase=Phase.match(groups.get("phase")),
                worker_id=groups.get("worker_id"),
                event_type=EventType.match(value_groups["event_type"]),
                value=value_groups["event_value"],
            )
        return None
