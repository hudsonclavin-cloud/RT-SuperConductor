"""Stable contracts for classical, quantum, and laboratory simulation stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class SimulationStage(str, Enum):
    """Progressive levels of virtual or physical investigation."""

    COMPOSITION_SCREEN = "composition_screen"
    STRUCTURE_SCREEN = "structure_screen"
    PHYSICS_CALCULATION = "physics_calculation"
    EXPERIMENT = "experiment"


class TaskStatus(str, Enum):
    """Lifecycle state shared by all simulation backends."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class SimulationTask:
    """A reproducible request sent to a backend."""

    task_id: str
    candidate_id: str
    stage: SimulationStage
    input_revision: str
    requested_outputs: tuple[str, ...] = field(default_factory=tuple)
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    status: TaskStatus = TaskStatus.QUEUED

    def __post_init__(self) -> None:
        for name in ("task_id", "candidate_id", "input_revision"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Backend output kept separate from observed evidence."""

    task_id: str
    status: TaskStatus
    outputs: dict[str, object] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    backend: str | None = None
    provenance: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if self.status is TaskStatus.FAILED and not self.error:
            raise ValueError("failed results must include an error")
        if any(value < 0 for value in self.uncertainty.values()):
            raise ValueError("uncertainties must be non-negative")


class SimulationBackend(Protocol):
    """Interface implemented by ML, DFT, quantum, or lab adapters."""

    name: str
    stage: SimulationStage

    def run(self, task: SimulationTask) -> SimulationResult:
        """Execute one task and return a provenance-bearing result."""


def make_task(
    *,
    candidate_id: str,
    stage: SimulationStage,
    input_revision: str,
    requested_outputs: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
) -> SimulationTask:
    """Create a stable task identifier from the candidate and input revision."""

    digest_input = "|".join((candidate_id, stage.value, input_revision, *depends_on))
    task_id = f"{stage.value}:{hashlib.sha256(digest_input.encode()).hexdigest()[:16]}"
    return SimulationTask(
        task_id=task_id,
        candidate_id=candidate_id,
        stage=stage,
        input_revision=input_revision,
        requested_outputs=requested_outputs,
        depends_on=depends_on,
    )
