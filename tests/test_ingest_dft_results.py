import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ingest_dft_results import completion_status, ingest_results, parse_epw_targets, parse_matdyn_frequencies


class DFTResultIngestionTests(unittest.TestCase):
    def test_nscf_completion_does_not_require_scf_convergence_marker(self):
        self.assertEqual(completion_status("qe_nscf", "bands written\nJOB DONE."), ("completed", "job_done"))
        self.assertEqual(completion_status("qe_nscf", "bands written"), ("not_converged", "missing_job_done"))

    def test_relax_requires_finished_optimization_and_job_done(self):
        self.assertEqual(completion_status("qe_relax", "End of BFGS Geometry Optimization\nJOB DONE."),
                         ("completed", "job_done_and_relaxation_end"))
        self.assertEqual(completion_status("qe_relax", "JOB DONE."),
                         ("not_converged", "missing_job_done_or_relaxation_end"))

    def test_parses_matdyn_thz_and_epw_targets(self):
        frequencies = parse_matdyn_frequencies(
            "freq ( 1) = -0.2500 [THz] = -8.33 [cm-1]\n"
            "omega( 2) = 4.125D+00 [THz] = 137.6 [cm-1]\n"
            "freq ( 3) = 33.3564095 [cm-1]"
        )
        self.assertAlmostEqual(frequencies[0], -0.25)
        self.assertAlmostEqual(frequencies[1], 4.125)
        self.assertAlmostEqual(frequencies[2], 1.0)
        self.assertEqual(parse_epw_targets("lambda = 0.7572\nomega_log = 164.99 K"), (0.7572, 164.99))
        lam, omega = parse_epw_targets("lambda = 0.2\nomega_log = 100 meV")
        self.assertEqual(lam, 0.2)
        self.assertAlmostEqual(omega, 1160.451812155)
        self.assertEqual(parse_epw_targets("lambda = 0.2\nomega_log = 100"), (0.2, None))

    def test_ingestion_emits_parsed_phonon_and_epw_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "ta"
            root.mkdir(parents=True)
            (root / "matdyn.out").write_text("freq (1) = 3.0 [THz]\nJOB DONE.")
            (root / "epw.out").write_text("lambda = 0.7\nomega_log = 100 K\n")
            out = Path(tmp) / "parsed.csv"
            report = ingest_results(Path(tmp) / "results", out)
            with out.open(newline="") as handle:
                rows = {row["stage"]: row for row in csv.DictReader(handle)}
        self.assertEqual(report["phonon_frequency_records"], 1)
        self.assertEqual(json.loads(rows["qe_matdyn"]["phonon_frequencies_thz_json"]), [3.0])
        self.assertEqual(rows["epw"]["epw_lambda"], "0.7")
        self.assertEqual(rows["epw"]["epw_omega_log_k"], "100.0")


if __name__ == "__main__":
    unittest.main()
