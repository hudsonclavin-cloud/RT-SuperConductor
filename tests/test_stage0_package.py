import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/benchmarks/epw-pb-tutorial"


def test_vendored_files_match_provenance_checksums():
    table = (PACKAGE / "PROVENANCE.md").read_text()
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", table)
    assert len(rows) == 7
    for relative, digest in rows:
        assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == digest, relative


def test_inputs_point_at_the_shipped_pseudopotential_and_phonon_save():
    for name in ("phonons/scf.in", "epw/scf.in", "epw/nscf.in"):
        text = (PACKAGE / "woSOC" / name).read_text()
        assert re.search(r"pseudo_dir\s*=\s*'\.\./\.\./pp/'", text), name
        assert "pb_s.UPF" in text
    assert (PACKAGE / "pp/pb_s.UPF").is_file()
    assert "dvscf_dir   = '../phonons/save'" in (PACKAGE / "woSOC/epw/epw.in").read_text()
    assert "os.system('mkdir save')" in (PACKAGE / "woSOC/phonons/pp.py").read_text()


def test_submit_script_dry_run_chains_six_jobs(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(tmp_path / "home")}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hpc/submit_stage0_pb.sh"), str(tmp_path / "run"), "part", "qos", "4"],
        capture_output=True, text=True, env=env, check=True)
    commands = [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 6
    assert all("--partition=part" in c and "--qos=qos" in c for c in commands)
    assert all("--dependency=afterok:" in c for c in commands[1:])
    assert "epw.x -npool 4" in commands[-1]
    assert (tmp_path / "run/woSOC/phonons/logs/staging-manifest.txt").is_file()


def test_submit_script_refuses_home(tmp_path):
    home = tmp_path / "home"
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(home)}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hpc/submit_stage0_pb.sh"), str(home / "run"), "part", "qos"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 3


def _dry_run(tmp_path, *args, **extra_env):
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(tmp_path / "home"), **extra_env}
    for name in ("STAGE0_MODE", "LAUNCHER", "LOCAL", "PH_TIME"):
        if name not in extra_env:
            env.pop(name, None)
    return subprocess.run(
        ["bash", str(ROOT / "scripts/hpc/submit_stage0_pb.sh"), str(tmp_path / "run"), *args],
        capture_output=True, text=True, env=env)


def test_smoke_mode_is_the_default_and_edits_only_the_copies(tmp_path):
    result = _dry_run(tmp_path, "compute", "normal", "8")
    assert result.returncode == 0, result.stderr
    run = tmp_path / "run"
    assert "8 8 8 1 1 1" in (run / "woSOC/phonons/scf.in").read_text()
    ph = (run / "woSOC/phonons/ph.in").read_text()
    assert all(f"nq{n}=3," in ph for n in (1, 2, 3))
    epw = (run / "woSOC/epw/epw.in").read_text()
    assert all(re.search(rf"nq{n}\s+= 3", epw) for n in (1, 2, 3))
    # The coarse k-mesh in epw.in must stay a multiple of the q-mesh.
    assert all(re.search(rf"nk{n}\s+= 6", epw) for n in (1, 2, 3))
    assert all(re.search(rf"nqf{n}\s+= 8\n", epw) for n in (1, 2, 3))
    # The fine k-mesh must stay dense: at 8x8x8 EPW finds no states at the Fermi
    # level and lambda divides by zero (tests/fixtures/epw_smoke_zero_dos_excerpt.out).
    assert all(re.search(rf"nkf{n}\s+= 20\n", epw) for n in (1, 2, 3))
    record = (run / "STAGE0_MODE.txt").read_text()
    assert record.startswith("mode=smoke") and record.count("->") == 10
    assert "14 14 14 1 1 1" in (PACKAGE / "woSOC/phonons/scf.in").read_text()
    assert "--time=01:50:00" in [line for line in result.stderr.splitlines() if "ph.x" in line][0]


def test_full_mode_leaves_the_official_inputs_unchanged_and_allows_a_long_phonon_job(tmp_path):
    result = _dry_run(tmp_path, "compute", "normal", "32", STAGE0_MODE="full")
    assert result.returncode == 0, result.stderr
    run = tmp_path / "run"
    for name in ("phonons/scf.in", "phonons/ph.in", "epw/epw.in"):
        assert (run / "woSOC" / name).read_bytes() == (PACKAGE / "woSOC" / name).read_bytes(), name
    assert (run / "STAGE0_MODE.txt").read_text().strip() == "mode=full"
    assert "--time=24:00:00" in [line for line in result.stderr.splitlines() if "ph.x" in line][0]


def test_unknown_mode_is_refused(tmp_path):
    assert _dry_run(tmp_path, "compute", "normal", STAGE0_MODE="quick").returncode == 2


def test_launcher_replaces_srun_inside_slurm_jobs(tmp_path):
    result = _dry_run(tmp_path, "compute", "normal", "8", LAUNCHER="mpirun -np 8")
    commands = [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 6
    # (pytest's temporary directory is named after this test, so match the command, not the word)
    assert not any(re.search(r"srun (pw|ph|epw)\.x", c) for c in commands)
    assert sum("mpirun -np 8" in c for c in commands) == 5   # every step but the serial pp.py


def test_local_mode_runs_the_same_steps_without_sbatch(tmp_path):
    result = _dry_run(tmp_path, "-", "-", "4", LOCAL="1")
    lines = result.stderr.splitlines()
    assert not any(line.startswith("sbatch ") for line in lines)
    steps = [line for line in lines if line.startswith("(cd ")]
    assert len(steps) == 6
    assert "mpirun -np 4 epw.x -npool 4" in steps[-1]
    assert "python3 pp.py" in steps[2] and "mpirun" not in steps[2]


def test_qe_setup_script_refuses_home_and_pins_the_checked_version(tmp_path):
    script = ROOT / "scripts/hpc/setup_qe_env.sh"
    home = tmp_path / "home"
    env = {**os.environ, "DRY_RUN": "1", "HOME": str(home)}
    assert subprocess.run(["bash", str(script), str(home / "apps")], capture_output=True, env=env).returncode == 3
    result = subprocess.run(["bash", str(script), str(tmp_path / "apps")], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "qe=7.6=mpi_openmpi_*" in result.stdout
    assert f"-p {tmp_path / 'apps'}/envs/rtsc-qe" in result.stdout
    for exe in ("epw.x", "wannier90.x", "ph.x"):
        assert exe in result.stdout


def test_mpi_check_job_is_a_small_two_task_job():
    text = (ROOT / "scripts/hpc/check_qe_mpi.sbatch").read_text()
    for line in ("#SBATCH --partition=compute", "#SBATCH --qos=normal", "#SBATCH --ntasks=2", "#SBATCH --time=00:05:00"):
        assert line in text
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/hpc/check_qe_mpi.sbatch")]).returncode == 0


def test_resume_skips_earlier_steps_and_keeps_the_inputs(tmp_path):
    assert _dry_run(tmp_path, "compute", "normal", "8").returncode == 0
    marker = tmp_path / "run/woSOC/phonons/ph.in"
    marker.write_text(marker.read_text() + "! edited by hand\n")
    result = _dry_run(tmp_path, "compute", "normal", "8", START_AT="pp")
    assert result.returncode == 0, result.stderr
    assert "! edited by hand" in marker.read_text()          # inputs were not recopied
    commands = [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]
    assert [re.search(r"--job-name=rtsc-pb-(\S+)", c).group(1) for c in commands] == ["pp", "epw-scf", "nscf", "epw"]
    assert "--dependency" not in commands[0]                  # the first resumed job waits on nothing
    assert "ph skipped" in result.stdout


def test_resume_needs_an_existing_run_and_a_known_step(tmp_path):
    assert _dry_run(tmp_path, "compute", "normal", START_AT="epw").returncode == 2
    assert _dry_run(tmp_path, "compute", "normal", START_AT="wannier").returncode == 2
