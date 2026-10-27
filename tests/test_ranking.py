import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from rt_superconductor.ranking import CandidateAssessment, rank_candidates


class RankingTests(unittest.TestCase):
    def test_pressure_constraint_excludes_candidate(self) -> None:
        allowed = CandidateAssessment("allowed", 0.7, 0.5, 0.8, 0.8, 0.4)
        excluded = CandidateAssessment(
            "high-pressure", 1.0, 1.0, 1.0, 1.0, 1.0, within_pressure_limit=False
        )

        ranked = rank_candidates([excluded, allowed])

        self.assertEqual([item.candidate_id for item, _ in ranked], ["allowed"])

    def test_score_orders_candidates_deterministically(self) -> None:
        first = CandidateAssessment("first", 0.8, 0.8, 0.8, 0.8, 0.8)
        second = CandidateAssessment("second", 0.4, 0.4, 0.4, 0.4, 0.4)

        ranked = rank_candidates([second, first])

        self.assertEqual([item.candidate_id for item, _ in ranked], ["first", "second"])
        self.assertAlmostEqual(ranked[0][1], 0.8)


if __name__ == "__main__":
    unittest.main()
