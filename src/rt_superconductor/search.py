"""Model-agnostic active-learning candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .simulation import SimulationStage, SimulationTask, make_task
from .targets import (
    ENGINEERING_PRESSURE_CEILING_GPA,
    ROOM_TEMPERATURE_MAX_K,
    ROOM_TEMPERATURE_MIN_K,
)

ENGINEERING_PRESSURE_GPA = ENGINEERING_PRESSURE_CEILING_GPA
ROOM_TEMP_MIN_K = ROOM_TEMPERATURE_MIN_K
ROOM_TEMP_MAX_K = ROOM_TEMPERATURE_MAX_K


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    """A proposed material state, distinct from an observed evidence record."""

    candidate_id: str
    composition: str
    predicted_tc_k: float
    tc_uncertainty_k: float
    pressure_gpa: float | None
    structural_plausibility: float = 0.0
    synthesis_feasibility: float = 0.0
    novelty: float = 0.0
    evidence_record_ids: tuple[str, ...] = field(default_factory=tuple)
    provenance: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.composition.strip():
            raise ValueError("candidate_id and composition must not be empty")
        if self.predicted_tc_k < 0 or self.tc_uncertainty_k < 0:
            raise ValueError("predicted temperature and uncertainty must be non-negative")
        if self.pressure_gpa is not None and self.pressure_gpa < 0:
            raise ValueError("pressure must be non-negative")
        for name in ("structural_plausibility", "synthesis_feasibility", "novelty"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AcquisitionWeights:
    """Weights for the first transparent acquisition function."""

    target_fit: float = 0.30
    uncertainty: float = 0.20
    structural_plausibility: float = 0.15
    synthesis_feasibility: float = 0.20
    novelty: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.target_fit,
            self.uncertainty,
            self.structural_plausibility,
            self.synthesis_feasibility,
            self.novelty,
        )
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("acquisition weights must be non-negative and sum to 1")


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """A ranked proposal with reasons that a reviewer can inspect."""

    candidate: CandidateHypothesis
    score: float
    reasons: tuple[str, ...]
    next_stage: SimulationStage = SimulationStage.COMPOSITION_SCREEN


def _target_fit(predicted_tc_k: float) -> float:
    if ROOM_TEMP_MIN_K <= predicted_tc_k <= ROOM_TEMP_MAX_K:
        return 1.0
    distance = (
        ROOM_TEMP_MIN_K - predicted_tc_k
        if predicted_tc_k < ROOM_TEMP_MIN_K
        else predicted_tc_k - ROOM_TEMP_MAX_K
    )
    return math.exp(-distance / 50.0)


def acquisition_score(
    candidate: CandidateHypothesis,
    *,
    pressure_ceiling_gpa: float = ENGINEERING_PRESSURE_GPA,
    allow_unknown_pressure: bool = False,
    weights: AcquisitionWeights = AcquisitionWeights(),
) -> tuple[float, tuple[str, ...]] | None:
    """Score one candidate, returning ``None`` for hard pressure exclusions."""

    if candidate.pressure_gpa is None and not allow_unknown_pressure:
        return None
    if candidate.pressure_gpa is not None and candidate.pressure_gpa > pressure_ceiling_gpa:
        return None
    reasons = []
    if candidate.pressure_gpa is None:
        pressure_fit = 0.0
        reasons.append("pressure is unknown and requires review")
    else:
        pressure_fit = 1.0 - min(candidate.pressure_gpa / pressure_ceiling_gpa, 1.0)
        reasons.append(f"known pressure={candidate.pressure_gpa:g} GPa")
    target_fit = _target_fit(candidate.predicted_tc_k)
    if target_fit == 1.0:
        reasons.append("predicted temperature is inside the target window")
    elif candidate.predicted_tc_k > ROOM_TEMP_MAX_K:
        reasons.append("predicted temperature is above the target window and needs review")
    else:
        reasons.append("predicted temperature is below the target window")
    uncertainty_score = min(candidate.tc_uncertainty_k / 100.0, 1.0)
    score = (
        weights.target_fit * target_fit
        + weights.uncertainty * uncertainty_score
        + weights.structural_plausibility * candidate.structural_plausibility
        + weights.synthesis_feasibility * candidate.synthesis_feasibility
        + weights.novelty * candidate.novelty
    )
    if candidate.structural_plausibility > 0:
        reasons.append("a structure is available or structurally plausible")
    if candidate.synthesis_feasibility > 0:
        reasons.append("synthesis feasibility has supporting evidence")
    if candidate.novelty > 0:
        reasons.append("the candidate adds novelty or information value")
    return score, tuple(reasons)


def propose_batch(
    candidates: list[CandidateHypothesis],
    *,
    batch_size: int = 10,
    pressure_ceiling_gpa: float = ENGINEERING_PRESSURE_GPA,
    allow_unknown_pressure: bool = False,
    weights: AcquisitionWeights = AcquisitionWeights(),
) -> list[CandidateProposal]:
    """Return a deterministic, reviewable next batch for virtual experiments."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    proposals = []
    for candidate in candidates:
        scored = acquisition_score(
            candidate,
            pressure_ceiling_gpa=pressure_ceiling_gpa,
            allow_unknown_pressure=allow_unknown_pressure,
            weights=weights,
        )
        if scored is None:
            continue
        score, reasons = scored
        proposals.append(CandidateProposal(candidate=candidate, score=score, reasons=reasons))
    proposals.sort(key=lambda item: (-item.score, item.candidate.candidate_id))
    return proposals[:batch_size]


def make_simulation_tasks(
    proposals: list[CandidateProposal], *, input_revision: str
) -> list[SimulationTask]:
    """Turn ranked proposals into stable composition-screen tasks."""

    return [
        make_task(
            candidate_id=proposal.candidate.candidate_id,
            stage=proposal.next_stage,
            input_revision=input_revision,
            requested_outputs=("predicted_tc_k", "tc_uncertainty_k", "pressure_feasibility"),
        )
        for proposal in proposals
    ]
