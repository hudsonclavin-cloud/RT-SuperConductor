import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_tantalum_epw_benchmark import prepare_tantalum_epw_benchmark


class TantalumEPWBenchmarkTests(unittest.TestCase):
    def test_jvasp_1014_targets_and_blocked_input_manifest(self):
        source = {
            "jid": "JVASP-1014", "lamb": 0.7572, "wlog": 164.9922,
            "Tc": 7.6395, "press": "0 GPa",
            "atoms": {
                "lattice_mat": [[-1.640392536, 1.640392536, 1.640392536],
                                [1.640392536, -1.640392536, 1.640392536],
                                [1.640392536, 1.640392536, -1.640392536]],
                "coords": [[0.0, 0.0, 0.0]], "elements": ["Ta"], "cartesian": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = prepare_tantalum_epw_benchmark(
                output_dir=root / "ta", source_row=source,
                source_sha256="pinned-source-checksum-for-test",
                pseudopotential_dir=root / "pseudo",
            )
            manifest = json.loads((root / "ta/benchmark-manifest.json").read_text())
            self.assertEqual(result["status"], "blocked_prerequisites")
            self.assertEqual(manifest["jid"], "JVASP-1014")
            self.assertEqual(manifest["target_lambda"], 0.7572)
            self.assertEqual(manifest["target_omega_log_k"], 164.99)
            self.assertEqual(manifest["dataset_doi"], "10.6084/m9.figshare.21370572.v4")
            self.assertFalse(manifest["dft_executed"])
            for name in manifest["input_files"]:
                self.assertTrue((root / "ta" / name).is_file())
            self.assertIn("HPC scheduler/access has not been confirmed", manifest["blocked_reasons"])

    def test_rejects_wrong_jarvis_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = prepare_tantalum_epw_benchmark(
                output_dir=Path(tmp) / "bad",
                source_row={"jid": "JVASP-1014", "lamb": 0.9, "wlog": 164.9922,
                            "atoms": {"lattice_mat": [[1,0,0],[0,1,0],[0,0,1]],
                                      "coords": [[0,0,0]], "elements": ["Ta"]}},
            )
        self.assertEqual(report["status"], "blocked_source_unavailable_or_invalid")


if __name__ == "__main__":
    unittest.main()
