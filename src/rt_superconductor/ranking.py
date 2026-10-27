"""Transparent starter ranking for research candidates."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Normalized decision factors for one proposed candidate."""

    candidate_id: str
    predicted_value: float
    uncertainty: float
    structural_plausibility: float
    synthesis_feasibility: float
    novelty: float
    within_pressure_limit: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        for name in (
            "predicted_value",
            "uncertainty",
            "structural_plausibility",
            "synthesis_feasibility",
            "novelty",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RankingWeights:
    predicted_value: float = 0.35
    uncertainty: float = 0.20
    structural_plausibility: float = 0.15
    synthesis_feasibility: float = 0.20
    novelty: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.predicted_value,
            self.uncertainty,
            self.structural_plausibility,
            self.synthesis_feasibility,
            self.novelty,
        )
        if any(value < 0 for value in values):
            raise ValueError("ranking weights must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1")


def rank_candidates(
    candidates: list[CandidateAssessment],
    weights: RankingWeights = RankingWeights(),
) -> list[tuple[CandidateAssessment, float]]:
    """Rank feasible candidates by a documented weighted score.

    Uncertainty is rewarded because the initial objective includes information
    gain. Production ranking should replace this proxy with an experiment-aware
    acquisition function.
    """

    scored = []
    for candidate in candidates:
        if not candidate.within_pressure_limit:
            continue
        score = (
            weights.predicted_value * candidate.predicted_value
            + weights.uncertainty * candidate.uncertainty
            + weights.structural_plausibility * candidate.structural_plausibility
            + weights.synthesis_feasibility * candidate.synthesis_feasibility
            + weights.novelty * candidate.novelty
        )
        scored.append((candidate, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].candidate_id))
