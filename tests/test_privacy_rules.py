"""Hazel results for candidate compounds must not be committable (owner's decision of 2026-11-19, AGENTS.md)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
@pytest.mark.parametrize("path", ["results/namgpb2/report.json", "results/csir2p2/lambda", "results/sbte2/sbte2.dyn1",
                                  "results/limgpb2/report.json", "data/raw/hazel/namgpb2-full-results.tgz"])
def test_candidate_results_are_ignored_by_git(path):
    assert ignored(path)


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_the_benchmark_and_the_readme_can_still_be_committed():
    assert not ignored("results/zr6ga2ni/report.json")
    assert not ignored("results/README.md")
