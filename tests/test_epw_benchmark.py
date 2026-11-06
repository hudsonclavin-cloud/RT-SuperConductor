import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from prepare_epw_benchmark import prepare_epw_benchmark


class _FakeMPRester:
    def __init__(self, api_key, structure):
        self.api_key = api_key
        self.structure = structure
        self.materials = SimpleNamespace(
            summary=SimpleNamespace(search=self._search),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _search(self, *, formula, fields):
        assert formula == "Nb3Sn"
        assert "energy_above_hull" in fields
        return [SimpleNamespace(material_id="mp-nb3sn-test", formula_pretty="Nb3Sn", energy_above_hull=0.01)]

    def get_structure_by_material_id(self, material_id):
        assert material_id == "mp-nb3sn-test"
        return self.structure


class EPWBenchmarkTests(unittest.TestCase):
    def setUp(self):
        # Conventional 8-atom cubic A15 (Cr3Si-type, Pm-3n) Nb3Sn cell.
        # Nb occupies the six 6c sites; Sn occupies the two 2a sites.
        self.structure = Structure(
            Lattice.cubic(5.29),
            ["Nb"] * 6 + ["Sn"] * 2,
            [
                [0, 0.25, 0.5], [0, 0.75, 0.5],
                [0.5, 0, 0.25], [0.5, 0, 0.75],
                [0.25, 0.5, 0], [0.75, 0.5, 0],
                [0, 0, 0], [0.5, 0.5, 0.5],
            ],
        )
        self.assertEqual(len(self.structure), 8)
        self.assertEqual(self.structure.composition.reduced_formula, "Nb3Sn")
        self.assertEqual(SpacegroupAnalyzer(self.structure, symprec=0.01).get_space_group_number(), 223)

    def test_writes_mp_structure_based_inputs_and_marks_missing_prerequisites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = prepare_epw_benchmark(
                output_dir=root / "benchmark",
                api_key="test-key",
                pseudopotential_dir=root / "pseudos",
                rester_factory=lambda key: _FakeMPRester(key, self.structure),
            )
            output = root / "benchmark"
            self.assertEqual(report["status"], "blocked_prerequisites")
            self.assertFalse(report["dft_executed"])
            for name in ("relax.in", "vc-relax.in", "scf.in", "nscf.in", "ph.in", "q2r.in", "matdyn.in", "epw.in", "nb3sn-mp-reference.cif"):
                self.assertTrue((output / name).is_file())
            relax = (output / "relax.in").read_text(encoding="utf-8")
            self.assertIn("calculation = 'relax'", relax)
            self.assertIn("nat = 8", relax)
            self.assertIn("&IONS", relax)
            self.assertNotIn("ion_dynamics", relax.split("&CONTROL", 1)[1].split("/", 1)[0])
            self.assertIn("ATOMIC_POSITIONS angstrom", relax)
            self.assertIn("Nb3Sn", relax)
            self.assertIn("../pseudos", relax)
            self.assertIn("Hazel compute access/allocation has not been confirmed", report["blocked_reasons"])
            manifest = json.loads((output / "benchmark-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["materials_project_id"], "mp-nb3sn-test")
            self.assertEqual(manifest["literature_target_tc_k"], 18.0)
            self.assertEqual(len(self.structure), 8)

    def test_missing_api_key_writes_blocked_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MP_API_KEY": ""}):
                report = prepare_epw_benchmark(output_dir=Path(tmp) / "benchmark", api_key=None)
            self.assertEqual(report["status"], "blocked_missing_materials_project_key")
            self.assertFalse(report["dft_executed"])
            self.assertTrue((Path(tmp) / "benchmark/generation-summary.json").is_file())

    def test_prerequisites_only_remove_blocked_status_and_never_run_dft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pseudos = root / "upf"
            pseudos.mkdir()
            (pseudos / "Nb.UPF").write_text("test pseudo", encoding="utf-8")
            (pseudos / "Sn.UPF").write_text("test pseudo", encoding="utf-8")
            report = prepare_epw_benchmark(
                output_dir=root / "benchmark",
                api_key="test-key",
                pseudopotential_dir=pseudos,
                hazel_access_confirmed=True,
                rester_factory=lambda key: _FakeMPRester(key, self.structure),
            )
            self.assertEqual(report["status"], "prepared_for_manual_review")
            self.assertFalse(report["dft_executed"])
            self.assertTrue((root / "benchmark/nb3sn-mp-reference.cif").is_file())


if __name__ == "__main__":
    unittest.main()
