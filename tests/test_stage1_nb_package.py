"""The niobium benchmark package: files as recorded, inputs that agree, and meshes the chain accepts."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/benchmarks/epw-nb-bcc"
WRAPPER = ROOT / "scripts/hpc/submit_stage1_nb.sh"


def _value(text, key):
    return re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\s,!]+)", text, re.MULTILINE).group(1)


def _run(tmp_path, *args, **extra_env):
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(tmp_path / "home"), **extra_env}
    for name in ("STAGE0_MODE", "CHAIN_MODE", "LAUNCHER", "LOCAL", "PH_TIME", "EPW_TIME", "START_AT"):
        if name not in extra_env:
            env.pop(name, None)
    return subprocess.run(["bash", str(WRAPPER), *args], capture_output=True, text=True, env=env)


def test_files_match_provenance_checksums():
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", (PACKAGE / "PROVENANCE.md").read_text())
    assert len(rows) == 7
    for relative, digest in rows:
        assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == digest, relative


def test_the_three_pw_inputs_describe_the_same_crystal():
    texts = {name: (PACKAGE / name).read_text() for name in ("phonons/scf.in", "epw/scf.in", "epw/nscf.in")}
    for key in ("celldm(1)", "ecutwfc", "ibrav", "degauss", "smearing"):
        assert len({_value(text, key) for text in texts.values()}) == 1, key
    for text in texts.values():
        assert "Nb_ONCV_PBE_sr.upf" in text and "prefix      = 'nb'" in text
    assert _value(texts["phonons/scf.in"], "celldm(1)") == "6.2568" and _value(texts["epw/scf.in"], "ecutwfc") == "90"


def test_band_step_lists_the_full_coarse_grid_and_enough_bands():
    nscf = (PACKAGE / "epw/nscf.in").read_text()
    epw = (PACKAGE / "epw/epw.in").read_text()
    assert "nosym" not in nscf
    count = int(nscf.split("K_POINTS crystal\n")[1].splitlines()[0])
    assert count == int(_value(epw, "nk1")) * int(_value(epw, "nk2")) * int(_value(epw, "nk3")) == 1728
    skipped = int(re.search(r"exclude_bands = 1-(\d+)", epw).group(1))
    assert int(_value(nscf, "nbnd")) >= skipped + int(_value(epw, "nbndsub")) and _value(epw, "nbndsub") == "9"


def test_epw_input_uses_the_double_delta_formula_and_one_point_per_symmetry_set():
    lines = [line.split("!")[0].replace(" ", "") for line in (PACKAGE / "epw/epw.in").read_text().splitlines()]
    assert "delta_approx=.true." in lines and "mp_mesh_q=.true." in lines and "a2f=.true." in lines
    assert "amass(1)=92.90637" in lines


def test_meshes_are_commensurate_in_both_modes(tmp_path):
    for mode in ("full", "smoke"):
        run = tmp_path / mode
        result = _run(tmp_path, str(run), "compute", "normal", "16", CHAIN_MODE=mode)
        assert result.returncode == 0, result.stderr
        epw = (run / "epw/epw.in").read_text()
        ph = (run / "phonons/ph.in").read_text()
        for n in (1, 2, 3):
            nq, nk = int(_value(epw, f"nq{n}")), int(_value(epw, f"nk{n}"))
            assert nk % nq == 0 and nq == int(_value(ph, f"nq{n}"))
            assert int(_value(epw, f"nkf{n}")) >= 20
        assert int(_value(ph, "nq1")) == (6 if mode == "full" else 2)
    assert (tmp_path / "smoke/CHAIN_MODE.txt").read_text().count("->") == 10
    commands = [line for line in _run(tmp_path, str(tmp_path / "again"), "compute", "normal", "16",
                                      CHAIN_MODE="full").stderr.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 6 and all("--job-name=rtsc-nb-" in c for c in commands)


def test_review_template_carries_the_reference_the_pass_rule_names():
    review = json.loads((PACKAGE / "review.template.json").read_text())
    assert review["role"] == "benchmark" and review["converged"] is False and review["formula"] == "Nb"
    assert review["reference"] == {"lambda": 1.248, "omega_log_k": 147.1, "source": review["reference"]["source"]}
    rule = (ROOT / "docs/hazel-runbook.md").read_text()
    assert "**Niobium: λ\n   1.248, ωlog 147.1 K.**" in rule
