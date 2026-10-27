"""Core types for the RT-SuperConductor research pipeline."""

from .evidence import EvidenceKind, EvidenceRecord
from .formula import FormulaResult, canonicalize_formula
from .datasets import iter_3dsc_mp_records, iter_supermat_records, iter_uci_records
from .ranking import CandidateAssessment, RankingWeights, rank_candidates
from .targets import AMBIENT_TARGET, ENGINEERING_TARGET, TargetWindow

__all__ = [
    "CandidateAssessment",
    "AMBIENT_TARGET",
    "EvidenceKind",
    "EvidenceRecord",
    "FormulaResult",
    "RankingWeights",
    "ENGINEERING_TARGET",
    "TargetWindow",
    "canonicalize_formula",
    "iter_3dsc_mp_records",
    "iter_supermat_records",
    "iter_uci_records",
    "rank_candidates",
]
