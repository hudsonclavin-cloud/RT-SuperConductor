import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_untested_pool import (  # noqa: E402
    chemistry_policy_flags,
    composition_structure_distance,
    eligible_family_metrics,
    _fetch_mp_with_client,
    read_formula_source,
    qualify_by_method,
    reduced_formula,
    score_family_candidates,
    select_top_per_family,
    summary_filter_candidates,
)
from rt_superconductor.harmonized import HarmonizedRecord


def _structure(species="Nb", *, disordered=False):
    from pymatgen.core import Lattice, Structure

    site_species = {species: 0.5, "Sn": 0.5} if disordered else species
    return Structure(Lattice.cubic(3.3), [site_species], [[0, 0, 0]])


def _summary(mp_id, formula, *, hull=0.01, sites=2, is_metal=True, gap=0.0, ordered=True, sg=225):
    return SimpleNamespace(
        material_id=mp_id,
        formula_pretty=formula,
        composition_reduced=None,
        energy_above_hull=hull,
        nsites=sites,
        is_metal=is_metal,
        band_gap=gap,
        structure=_structure("Nb" if ordered else "Fe", disordered=not ordered),
        symmetry=SimpleNamespace(number=sg),
        task_ids=[f"{mp_id}-task"],
        es_source_calc_id=None,
    )


class UntestedPoolTests(unittest.TestCase):
    def test_formula_normalization_uses_reduced_composition(self):
        self.assertEqual(reduced_formula("Fe2Se2"), "FeSe")
        self.assertEqual(reduced_formula("Nb3Sn"), "Nb3Sn")
        self.assertIsNone(reduced_formula("not a formula"))

    def test_summary_filters_apply_stability_size_ordering_metallicity_and_source_match(self):
        docs = [
            _summary("mp-new", "Nb3Sn"),
            _summary("mp-unstable", "FeSe", hull=0.03),
            _summary("mp-large", "Al", sites=21),
            _summary("mp-disordered", "NbSn", ordered=False),
            _summary("mp-insulator", "MgO", is_metal=False, gap=1.8),
            _summary("mp-known", "CuO"),
        ]

        kept, counts = summary_filter_candidates(docs, {"CuO"})

        self.assertEqual([row.material_id for row in kept], ["mp-new"])
        self.assertEqual(counts["mp_summary_returned"], 6)
        self.assertEqual(counts["energy_above_hull_le_0p025"], 5)
        self.assertEqual(counts["site_count_le_20"], 4)
        self.assertEqual(counts["ordered_structures"], 3)
        self.assertEqual(counts["summary_metallic_prefilter"], 2)
        self.assertEqual(counts["formula_matches_known_superconductor_source"], 1)
        self.assertEqual(counts["unmatched_reduced_formula"], 1)

    def test_unparseable_source_system_overlap_is_reported_without_claiming_a_formula_match(self):
        kept, counts = summary_filter_candidates(
            [_summary("mp-new", "Nb3Sn")], set(), {("Nb", "Sn")}
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(counts["formula_matches_known_superconductor_source"], 0)
        self.assertEqual(counts["ambiguous_formula_source_system_overlap"], 1)

    def test_supported_metallicity_requires_method_level_zero_gap_or_metal_flag(self):
        docs = [
            _summary("mp-a", "Nb3Sn"),
            _summary("mp-b", "AlCu", gap=0.0),
        ]
        docs[1].task_ids = ["mp-b-task"]
        tasks = [
            SimpleNamespace(task_id="mp-a-task", run_type="GGA+U", output=SimpleNamespace(is_metal=True, bandgap=0.0)),
            SimpleNamespace(task_id="mp-b-task", run_type="GGA+U", output=SimpleNamespace(is_metal=True, bandgap=0.1)),
            SimpleNamespace(task_id="gga-task", run_type="GGA", output=SimpleNamespace(is_metal=True, bandgap=0.0)),
            SimpleNamespace(task_id="scan-insulator", run_type="r2SCAN", output=SimpleNamespace(is_metal=False, bandgap=0.2)),
        ]

        qualified, counts = qualify_by_method(docs, tasks)

        self.assertEqual(qualified[0][1], (("GGA+U", 0.0),))
        self.assertEqual(counts["method_qualified_metallic"], 1)
        self.assertEqual(counts["no_supported_metallic_method"], 1)

    def test_mp_fetch_uses_current_supported_summary_fields_and_task_ids(self):
        class SummaryRoute:
            def search(self, **kwargs):
                self.kwargs = kwargs
                return [_summary("mp-new", "Nb3Sn")]

        class TaskRoute:
            def search(self, **kwargs):
                self.kwargs = kwargs
                return [SimpleNamespace(
                    task_id="mp-new-task", run_type="GGA+U",
                    output=SimpleNamespace(bandgap=0.0),
                )]

        summary = SummaryRoute()
        tasks = TaskRoute()
        client = SimpleNamespace(materials=SimpleNamespace(
            summary=summary, tasks=tasks,
        ))
        source_index = SimpleNamespace(formulas=set(), unresolved_element_systems=set())

        qualified, counts = _fetch_mp_with_client(client, source_index)

        self.assertNotIn("es_source_calc_id", summary.kwargs["fields"])
        self.assertEqual(tasks.kwargs["task_ids"], ["mp-new-task"])
        self.assertEqual(len(qualified), 1)
        self.assertEqual(counts["task_ids_queried"], 1)

    def test_family_gate_requires_explicit_embargo_and_strictly_more_than_half(self):
        report = {"within_family_evaluation": {"families": {
            "Pass": {"status": "ready", "shared_mp_ids_embargoed": True, "spearman_rho": 0.51},
            "At threshold": {"status": "ready", "shared_mp_ids_embargoed": True, "spearman_rho": 0.5},
            "Not embargoed": {"status": "ready", "spearman_rho": 0.9},
        }}}

        self.assertEqual(eligible_family_metrics(report), {"Pass": 0.51})

    def test_structure_distance_uses_space_group_in_addition_to_composition(self):
        cubic_match = composition_structure_distance("Nb3Sn", 225, "Nb3Sn", 225)
        tetragonal = composition_structure_distance("Nb3Sn", 225, "Nb3Sn", 123)
        self.assertLess(cubic_match, tetragonal)

    def test_only_top_50_per_family_are_written_in_conservative_score_order(self):
        rows = [
            {"chemical_family": "Other", "mp_id": f"mp-{index}",
             "conservative_score_tc_minus_sd_k": float(index), "predicted_tc_kelvin": float(index)}
            for index in range(60)
        ]

        top = select_top_per_family(rows, limit=50)

        self.assertEqual(len(top), 50)
        self.assertEqual(top[0]["mp_id"], "mp-59")
        self.assertEqual(top[-1]["rank_within_family"], 50)

    def test_chemistry_policy_flags_are_review_flags_for_existing_entries(self):
        self.assertIn("radioactive_or_actinide_lab_access:Th", chemistry_policy_flags("ThFeAsO"))
        self.assertIn("binary_transition_metal_oxide_or_halide_review", chemistry_policy_flags("NiO"))
        self.assertEqual(chemistry_policy_flags("Nb3Sn"), [])

    def test_source_formula_reader_normalizes_reduced_formulas(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("material,Tc\nFe2Se2,8\n", encoding="utf-8")

            formulas, unresolved, stats = read_formula_source(path, ("material",), "test")

        self.assertEqual(formulas, {"FeSe"})
        self.assertEqual(unresolved, set())
        self.assertEqual(stats["rows"], 1)

    def test_only_candidates_assigned_to_an_eligible_family_are_scored(self):
        training = [
            HarmonizedRecord("a" * 24, "3dsc", "FeSe", 20.0, 0.000101325, "unknown", "unknown",
                             "10.0000/test-a", "curated_database", "curated", 0,
                             chemical_family="Ferrites", mp_id="mp-1", energy_above_hull=0.01,
                             spacegroup_number=129),
            HarmonizedRecord("b" * 24, "3dsc", "Nb3Sn", 18.0, 0.000101325, "unknown", "unknown",
                             "10.0000/test-b", "curated_database", "curated", 0,
                             chemical_family="Other", mp_id="mp-2", energy_above_hull=0.01,
                             spacegroup_number=223),
        ]
        qualified = [
            (_summary("mp-fe-s", "FeS", sg=129), (("GGA+U", 0.0),)),
            (_summary("mp-other", "Nb3Sn", sg=223), (("r2SCAN", 0.0),)),
        ]

        scored, counts = score_family_candidates(qualified, training, {"Ferrites": 0.7})

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["chemical_family"], "Ferrites")
        self.assertEqual(scored[0]["dft_verification"], "pending")
        self.assertEqual(counts["family_gate_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
