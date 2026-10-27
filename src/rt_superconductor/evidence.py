"""Normalized evidence types that preserve unknown scientific values."""

from dataclasses import dataclass, field
from enum import Enum


class EvidenceKind(str, Enum):
    """How an assertion about a material was obtained."""

    EXPERIMENTAL = "experimental"
    COMPUTATIONAL = "computational"
    MODEL_PREDICTION = "model_prediction"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A traceable assertion about one material or sample.

    Optional numeric fields use ``None`` for unknown values. In particular, an
    unknown transition temperature must not be represented as zero.
    """

    record_id: str
    composition: str
    source_reference: str
    evidence_kind: EvidenceKind
    crystal_structure: str | None = None
    material_family: str | None = None
    reported_tc_k: float | None = None
    measurement_pressure_gpa: float | None = None
    synthesis_conditions: str | None = None
    measurement_method: str | None = None
    sample_identifier: str | None = None
    confidence: float | None = None
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("record_id", "composition", "source_reference"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.reported_tc_k is not None and self.reported_tc_k < 0:
            raise ValueError("reported_tc_k must be non-negative")
        if self.measurement_pressure_gpa is not None and self.measurement_pressure_gpa < 0:
            raise ValueError("measurement_pressure_gpa must be non-negative")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def has_ambient_pressure_evidence(self) -> bool:
        """Return whether a known pressure is close to atmospheric pressure."""

        return (
            self.measurement_pressure_gpa is not None
            and self.measurement_pressure_gpa <= 0.001
        )
