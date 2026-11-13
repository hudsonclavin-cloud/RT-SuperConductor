"""One command for the whole loop: ``python -m rtsc2 run``.

Stages write JSON/CSV under ``pipeline_v2/reports``. A stage is skipped when
its inputs, settings and the package source are unchanged since the report
on disk was written, so a second ``run`` finishes in seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import agent, data, evaluate
from .models import registry

REPORTS = data.ROOT / "pipeline_v2/reports"
DEFAULT_MODELS = ["legacy_boosting", "tree_bag", "deep_ensemble", "blend"]


# A stage is recomputed only when a module it actually depends on changes.
STAGE_MODULES = {"benchmark": ("data", "models", "evaluate"), "agent-replay": ("data", "models", "agent"),
                 "propose": ("data", "models", "agent", "cli")}


def _source_hash(modules: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if modules is None or path.stem in modules:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _provenance(candidates: Path) -> dict[str, Any]:
    import sklearn

    try:
        commit = subprocess.run(["git", "-C", str(data.ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        commit = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"git_commit": commit, "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "torch": torch_version, "rtsc2_source": _source_hash(),
            "candidates_sha256": data.sha256(candidates),
            "seeds": {"outer_split": "1000 + repeat", "models": "7 + repeat", "replay": "0..seeds-1"}}


def stage(name: str, settings: dict[str, Any], candidates: Path, compute: Callable[[], dict[str, Any]],
          reports: Path, force: bool = False) -> dict[str, Any]:
    key = hashlib.sha256(json.dumps([name, settings, data.sha256(candidates), _source_hash(STAGE_MODULES.get(name))],
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = reports / f"{name}.json"
    shown = path.relative_to(data.ROOT) if path.is_relative_to(data.ROOT) else path
    if path.is_file() and not force:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("cache_key") == key:
            print(f"[{name}] unchanged, reusing {shown}")
            return stored
    start = time.perf_counter()
    result = compute()
    result.update({"cache_key": key, "settings": settings, "seconds": round(time.perf_counter() - start, 1),
                   "provenance": _provenance(candidates)})
    reports.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{name}] done in {result['seconds']} s -> {shown}")
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        import pandas as pd

        labelled = data.load(args.candidates, labelled_only=True)
        report = evaluate.benchmark(labelled, args.models, repeats=args.repeats)
        scores = report.pop("scores")
        table = pd.concat([pd.DataFrame({"mp_id": labelled.ids, "model": name, "repeat": repeat,
                                         "label": labelled.label.astype(int), "score": np.round(p, 6)})
                           for name, per in scores.items() if not name.startswith("legacy:")
                           for repeat, p in enumerate(per)])
        args.reports.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.reports / "benchmark-oof.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
        return report

    return stage("benchmark", {"models": args.models, "repeats": args.repeats}, args.candidates, compute,
                 args.reports, args.force)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    settings = {"seeds": args.seeds, "rounds": args.rounds, "batch": args.batch, "initial": args.initial,
                "high_tc_k": args.high_tc}

    def compute() -> dict[str, Any]:
        labelled = data.load(args.candidates, labelled_only=True)
        kwargs = dict(seeds=args.seeds, rounds=args.rounds, batch=args.batch, initial=args.initial)
        high = (np.nan_to_num(labelled.tc, nan=0.0) >= args.high_tc).astype(int)
        targets = {"any_superconductor": labelled.label.astype(int), f"tc_at_least_{args.high_tc:g}_k": high}
        report: dict[str, Any] = {name: agent.compare_policies(labelled, y, **kwargs) for name, y in targets.items()}
        # Strict replay: one query per unseen element system. "diverse" equals "greedy" there.
        strict = tuple(p for p in agent.POLICIES if p != "diverse")
        report["strict"] = {name: agent.compare_policies(labelled, y, policies=strict, strict=True, **kwargs)
                            for name, y in targets.items()}
        report["caveat"] = ("Replay over materials whose outcome is already known. It ranks policies on this "
                            "labelled set; it does not measure the hit rate to expect on untested materials.")
        return report

    return stage("agent-replay", settings, args.candidates, compute, args.reports, args.force)


def eligible_mask(dataset: data.Dataset, max_sites: int = 20) -> np.ndarray:
    """The legacy EPW screen (Other family, metallic, non-magnetic, small cell), plus no
    element without a stable isotope: Tc, Pm, and everything from Po upward."""
    frame = dataset.frame
    sites = np.asarray([float(v) if v else np.inf for v in frame["num_sites"]])
    z = dataset.el_index
    radioactive = ((z == 43) | (z == 61) | (z >= 84)).any(axis=1)
    return ((frame["chemical_family"] == "Other") & (frame["is_metal"] == "True")
            & (frame["is_magnetic"] == "False") & (sites <= max_sites)).to_numpy() & ~radioactive


def run_propose(args: argparse.Namespace) -> dict[str, Any]:
    settings = {"model": args.model, "policy": args.policy, "batch": args.batch}

    def compute() -> dict[str, Any]:
        full = data.load(args.candidates)
        known = full.subset(full.labelled)
        model = registry(seed=0)[args.model]().fit(
            known.X, known.label.astype(int), groups=known.groups, tc=known.tc,
            tokens=(known.el_index, known.el_fraction))
        eligible = eligible_mask(full)
        table = agent.propose(full, model, policy=args.policy, batch=args.batch, eligible=eligible)
        args.reports.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.reports / "next-batch.csv", index=False, lineterminator="\n")
        return {"trained_on": int(full.labelled.sum()), "untested": int((~full.labelled).sum()),
                "eligible_untested": int((~full.labelled & eligible).sum()), "proposed": int(len(table)),
                "output": "pipeline_v2/reports/next-batch.csv",
                "interpretation": "Model hypotheses for review. Not predictions of superconductivity."}

    return stage("propose", settings, args.candidates, compute, args.reports, args.force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rtsc2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "benchmark", "replay", "propose"])
    parser.add_argument("--candidates", type=Path, default=data.CANDIDATES)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--force", action="store_true", help="recompute even when a stage is unchanged")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="benchmark: models to compare")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=10, help="replay: campaigns per policy")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--initial", type=int, default=100, help="replay: size of the starting labelled set")
    parser.add_argument("--high-tc", type=float, default=10.0, help="replay: kelvin threshold for the rare target")
    parser.add_argument("--model", default="tree_bag", choices=sorted(registry()), help="propose: model to fit")
    parser.add_argument("--policy", default="greedy", choices=agent.POLICIES,
                        help="propose: selection rule; greedy is the default because no policy beat it in replay")
    args = parser.parse_args(argv)
    todo = {"run": (run_benchmark, run_replay, run_propose), "benchmark": (run_benchmark,),
            "replay": (run_replay,), "propose": (run_propose,)}[args.command]
    for step in todo:
        step(args)
    return 0
