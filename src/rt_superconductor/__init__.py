"""Core types for the RT-SuperConductor research pipeline."""

from .evidence import EvidenceKind, EvidenceRecord
from .formula import FormulaResult, canonicalize_formula
from .datasets import iter_3dsc_mp_records, iter_supermat_records, iter_uci_records
from .ranking import CandidateAssessment, RankingWeights, rank_candidates
from .search import (
    AcquisitionWeights,
    CandidateHypothesis,
    CandidateProposal,
    make_simulation_tasks,
    propose_batch,
)
from .simulation import SimulationResult, SimulationStage, SimulationTask, make_task
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
    "AcquisitionWeights",
    "CandidateHypothesis",
    "CandidateProposal",
    "SimulationResult",
    "SimulationStage",
    "SimulationTask",
    "make_simulation_tasks",
    "make_task",
    "propose_batch",
    "iter_3dsc_mp_records",
    "iter_supermat_records",
    "iter_uci_records",
    "rank_candidates",
]
