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
