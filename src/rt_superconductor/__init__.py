"""Core types for the RT-SuperConductor research pipeline."""

from .evidence import EvidenceKind, EvidenceRecord
from .datasets import iter_3dsc_mp_records, iter_uci_records
from .pipeline import PipelineSummary, composition_key, run_pipeline
from .ranking import CandidateAssessment, RankingWeights, rank_candidates

__all__ = [
    "CandidateAssessment",
    "EvidenceKind",
    "EvidenceRecord",
    "PipelineSummary",
    "RankingWeights",
    "composition_key",
    "iter_3dsc_mp_records",
    "iter_uci_records",
    "rank_candidates",
    "run_pipeline",
]
