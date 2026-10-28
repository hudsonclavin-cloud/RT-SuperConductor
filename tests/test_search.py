import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.search import CandidateHypothesis, make_simulation_tasks, propose_batch
from rt_superconductor.simulation import SimulationStage, TaskStatus
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA


class SearchTests(unittest.TestCase):
    def test_pressure_limit_excludes_known_high_pressure_candidates(self) -> None:
        candidates = [
            CandidateHypothesis(
                candidate_id="ambient",
                composition="X2Y",
                predicted_tc_k=290,
                tc_uncertainty_k=20,
                pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
            ),
            CandidateHypothesis(
                candidate_id="high-pressure",
                composition="H3S",
                predicted_tc_k=203,
                tc_uncertainty_k=10,
                pressure_gpa=150,
            ),
        ]

        proposals = propose_batch(candidates, batch_size=10)

        self.assertEqual([proposal.candidate.candidate_id for proposal in proposals], ["ambient"])
        self.assertTrue(any("inside the target window" in reason for reason in proposals[0].reasons))

    def test_unknown_pressure_requires_explicit_review_mode(self) -> None:
        candidate = CandidateHypothesis(
            candidate_id="unknown-pressure",
            composition="X2Y",
            predicted_tc_k=290,
            tc_uncertainty_k=20,
            pressure_gpa=None,
        )

        self.assertEqual(propose_batch([candidate]), [])
        proposals = propose_batch([candidate], allow_unknown_pressure=True)
        self.assertEqual(len(proposals), 1)
        self.assertTrue(any("pressure is unknown" in reason for reason in proposals[0].reasons))

    def test_simulation_tasks_are_stable_and_queued(self) -> None:
        candidate = CandidateHypothesis(
            candidate_id="ambient",
            composition="X2Y",
            predicted_tc_k=290,
            tc_uncertainty_k=20,
            pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
        )
        proposals = propose_batch([candidate])
        first = make_simulation_tasks(proposals, input_revision="dataset-v1")
        second = make_simulation_tasks(proposals, input_revision="dataset-v1")

        self.assertEqual(first[0].task_id, second[0].task_id)
        self.assertEqual(first[0].stage, SimulationStage.COMPOSITION_SCREEN)
        self.assertEqual(first[0].status, TaskStatus.QUEUED)


if __name__ == "__main__":
    unittest.main()
