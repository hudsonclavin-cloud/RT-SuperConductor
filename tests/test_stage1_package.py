import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/benchmarks/epw-ta-jvasp-1014"
CHAIN = ROOT / "scripts/hpc/submit_epw_chain.sh"


def _run(tmp_path, *args, script=ROOT / "scripts/hpc/submit_stage1_ta.sh", **extra_env):
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(tmp_path / "home"), **extra_env}
    for name in ("STAGE0_MODE", "CHAIN_MODE", "LAUNCHER", "LOCAL", "PH_TIME", "EPW_TIME", "START_AT"):
        if name not in extra_env:
            env.pop(name, None)
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env)


def _value(text, key):
    return re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\s,!]+)", text, re.MULTILINE).group(1)


def test_files_match_provenance_checksums():
    table = (PACKAGE / "PROVENANCE.md").read_text()
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", table)
    assert len(rows) == 7
    for relative, digest in rows:
        assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == digest, relative


def test_the_three_pw_inputs_describe_the_same_crystal():
    texts = {name: (PACKAGE / name).read_text() for name in ("phonons/scf.in", "epw/scf.in", "epw/nscf.in")}
    for key in ("celldm(1)", "ecutwfc", "ibrav", "degauss", "smearing"):
        assert len({_value(text, key) for text in texts.values()}) == 1, key
    for text in texts.values():
        assert "Ta_ONCV_PBE_sr.upf" in text and "pseudo_dir  = '../pp/'" in text
    assert (PACKAGE / "pp/Ta_ONCV_PBE_sr.upf").is_file()


def test_band_step_lists_the_full_coarse_grid_with_symmetry_left_on():
    nscf = (PACKAGE / "epw/nscf.in").read_text()
    epw = (PACKAGE / "epw/epw.in").read_text()
    assert "nosym" not in nscf                     # EPW crashes on a band step run with nosym = .true.
    count = int(nscf.split("K_POINTS crystal\n")[1].splitlines()[0])
    nk = [int(_value(epw, f"nk{n}")) for n in (1, 2, 3)]
    assert count == nk[0] * nk[1] * nk[2] == 512
    # Enough bands for the skipped semicore states plus the Wannier functions.
    skipped = int(re.search(r"exclude_bands = 1-(\d+)", epw).group(1))
    assert int(_value(nscf, "nbnd")) >= skipped + int(_value(epw, "nbndsub"))


def test_meshes_are_commensurate_in_both_modes(tmp_path):
    for mode in ("full", "smoke"):
        run = tmp_path / mode
        result = _run(tmp_path, str(run), "compute", "normal", "16", CHAIN_MODE=mode)
        assert result.returncode == 0, result.stderr
        epw = (run / "epw/epw.in").read_text()
        ph = (run / "phonons/ph.in").read_text()
        for n in (1, 2, 3):
            nq, nk = int(_value(epw, f"nq{n}")), int(_value(epw, f"nk{n}"))
            assert nk % nq == 0                                   # EPW needs the k-grid to contain the q-grid
            assert nq == int(_value(ph, f"nq{n}"))                # and the same q-grid as the phonons
            assert int(_value(epw, f"nkf{n}")) >= 20              # dense enough to find states at the Fermi level
        assert "dvscf_dir   = '../phonons/save'" in epw
    assert (tmp_path / "full/CHAIN_MODE.txt").read_text().strip() == "mode=full"
    assert (tmp_path / "smoke/CHAIN_MODE.txt").read_text().count("->") == 10
    assert (tmp_path / "full/phonons/scf.in").read_bytes() == (PACKAGE / "phonons/scf.in").read_bytes()


def test_full_mode_gets_the_long_time_limits(tmp_path):
    result = _run(tmp_path, str(tmp_path / "run"), "compute", "normal", "16", CHAIN_MODE="full")
    commands = [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 6 and all("--job-name=rtsc-ta-" in c for c in commands)
    assert "--time=12:00:00" in next(c for c in commands if "ph.x" in c)
    assert "--time=04:00:00" in next(c for c in commands if "epw.x" in c)
    assert "printf 'ta\\n%s\\n'" in next(c for c in commands if "pp.py" in c)


def test_chain_refuses_a_band_step_with_symmetry_switched_off(tmp_path):
    package = tmp_path / "package"
    shutil.copytree(PACKAGE, package)
    nscf = package / "epw/nscf.in"
    nscf.write_text(nscf.read_text().replace("    nbnd        = 14\n", "    nbnd        = 14\n    nosym       = .true.\n"))
    result = _run(tmp_path, str(package), str(tmp_path / "run"), "compute", "normal", script=CHAIN)
    assert result.returncode == 6 and "nosym" in result.stderr


def test_chain_refuses_a_directory_that_is_not_a_package(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _run(tmp_path, str(tmp_path / "empty"), str(tmp_path / "run"), "compute", "normal", script=CHAIN)
    assert result.returncode == 2 and "chain.conf" in result.stderr


def test_review_template_carries_the_reference_the_project_already_uses():
    review = json.loads((PACKAGE / "review.template.json").read_text())
    assert review["role"] == "benchmark" and review["converged"] is False
    source = (ROOT / "scripts/prepare_tantalum_epw_benchmark.py").read_text()
    assert f"TARGET_LAMBDA = {review['reference']['lambda']}" in source
    assert f"TARGET_OMEGA_LOG_K = {review['reference']['omega_log_k']}" in source
