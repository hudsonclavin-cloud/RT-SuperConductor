#!/usr/bin/env python3
"""The overnight GPU run: every job in a fixed order, inside a time budget, resumable (``docs/overnight-gpu.md``).

    python scripts/overnight_gpu.py --out runs/rehearsal --profile smoke --device cuda      # about 5 minutes
    python scripts/overnight_gpu.py --out runs/overnight-20261122 --hours 8 --device cuda  # the real run

Run the same command again to continue after a crash, a reboot or a STOP: finished jobs are skipped, and the
time budget starts again from then (give it the hours that are left). Create a file named STOP in the output
folder to stop after the current job. On Windows the run keeps the machine awake while it works.

Order:
1. a 2-epoch structure check (the run stops if it fails);
2. the tabular network on the formula bundle, 20 networks per fold, in two widths;
3. track A, every development row, as cycle 6: formula_all (the control), hybrid_all, pretraining on the Materials
   Project structures, hybrid_all fine-tuned from it;
4. track S, the rows with a 3DSC structure: formula (the control), cgcnn, hybrid, and both fine-tuned;
5. a sweep around the better track A model, one setting changed at a time, while time allows;
6. ``final-A`` and ``final-S``: the configuration with the lowest mean validation loss in each track, retrained
   with new seeds.

Selection uses only each fold's early-stopping rows (``mean_validation_bce`` in ``run.json``). No test row is scored
on this machine. ``share/`` holds everything to send back (predictions and logs, no model weights).

Standalone: the Python standard library here, numpy and PyTorch in the two trainers it calls.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRUCTURE_TRAINER = HERE / "structure_gpu_train.py"
TABULAR_TRAINER = HERE / "local_gpu_train.py"
DEFAULT_STRUCTURES = ROOT / "data/processed/local-gpu/structures.npz"
DEFAULT_FORMULA = ROOT / "data/processed/local-gpu/bundle.npz"

# Seconds per network before anything has been measured (laptop GPU guesses; replaced by measurements as jobs finish).
GUESS_SECONDS = {"tabular": 6.0, "formula": 15.0, "S": 120.0, "A": 180.0, "pretrain": 900.0}
FINAL_SEEDS = {"A": (3, 10), "S": (2, 6)}          # (fewest, most) networks per fold in the final runs


def now() -> float:
    return time.time()


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.share = self.out / "share"
        self.out.mkdir(parents=True, exist_ok=True)
        self.share.mkdir(exist_ok=True)
        self.started = now()
        self.deadline = self.started + 3600.0 * args.hours
        self.jobs: list[dict] = []
        self.smoke = args.profile == "smoke"

    # -- logging and state -------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{stamp()} {message}"
        print(line, flush=True)
        with open(self.out / "driver.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def left(self) -> float:
        return self.deadline - now()

    def stop_requested(self) -> bool:
        return (self.out / "STOP").exists()

    def run_json(self, name: str) -> dict | None:
        path = self.out / name / "run.json"
        if path.is_file():
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                return run if run.get("complete") else None
            except json.JSONDecodeError:
                return None
        return None

    def seconds_per_network(self, family: str) -> float:
        measured = [j["seconds_per_network"] for j in self.jobs if j.get("family") == family and j.get("seconds_per_network")]
        return float(sum(measured) / len(measured)) if measured else GUESS_SECONDS[family]

    def estimate(self, job: dict) -> float:
        if job["kind"] == "pretrain":
            done = [j for j in self.jobs if j["kind"] == "pretrain" and j.get("minutes")]
            return 60.0 * done[0]["minutes"] if done else GUESS_SECONDS["pretrain"] * (0.02 if self.smoke else 1.0)
        return job["networks"] * self.seconds_per_network(job["family"]) * 1.15 + 20.0

    # -- one job ------------------------------------------------------------
    def command(self, job: dict, batch: int | None = None) -> list[str]:
        a = self.args
        out = self.out / job["name"]
        if job["kind"] == "tabular":
            return [a.python, str(TABULAR_TRAINER), str(a.formula), "--out", str(out), "--device", a.device, *job["options"]]
        if job["kind"] == "pretrain":
            return [a.python, str(STRUCTURE_TRAINER), "pretrain", str(a.structures), "--out", str(out), "--device", a.device,
                    *job["options"]]
        cmd = [a.python, str(STRUCTURE_TRAINER), "train", str(a.structures), "--out", str(out), "--device", a.device,
               "--model", job["model"], *job["options"]]
        if job["model"].endswith("_all"):
            cmd += ["--formula-bundle", str(a.formula)]
        if job.get("pretrained"):
            cmd += ["--pretrained", str(self.out / "pretrain" / "encoder.pt")]
        if batch:
            cmd += ["--batch", str(batch)]
        return cmd

    def execute(self, job: dict) -> dict:
        name = job["name"]
        done = self.run_json(name)
        if done is not None:
            job.update(status="done", minutes=done.get("minutes"), mean_validation_bce=done.get("mean_validation_bce"),
                       seconds_per_network=done.get("seconds_per_network"), resumed=True)
            self.log(f"[skip] {name}: finished in an earlier session")
            self.publish(job)
            return job
        if job.get("pretrained") and self.run_json("pretrain") is None:
            job.update(status="skipped", reason="no pretrained encoder")
            self.log(f"[skip] {name}: the pretraining job did not finish")
            return job
        folder = self.out / name
        if folder.exists():
            shutil.rmtree(folder)                          # an unfinished attempt; start it again
        folder.mkdir(parents=True)
        batch = None
        for attempt in (1, 2):
            cmd = self.command(job, batch)
            job["command"] = cmd
            self.log(f"[start] {name} (attempt {attempt}; about {self.estimate(job) / 60:.0f} min; "
                     f"{self.left() / 3600:.2f} h left)")
            began = now()
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with open(folder / "train.log", "a", encoding="utf-8") as handle:
                code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            text = (folder / "train.log").read_text(encoding="utf-8", errors="replace")
            run = self.run_json(name)
            if code == 0 and run is not None:
                job.update(status="done", minutes=round((now() - began) / 60.0, 2),
                           mean_validation_bce=run.get("mean_validation_bce"),
                           seconds_per_network=run.get("seconds_per_network"))
                self.log(f"[done] {name}: {job['minutes']} min, mean validation BCE {job['mean_validation_bce']}")
                self.publish(job)
                return job
            if "out of memory" in text.lower() and job["kind"] == "structure" and attempt == 1:
                batch = max(16, int(self.batch_of(job) / 2))
                self.log(f"[retry] {name}: GPU out of memory; trying again with batch {batch}")
                continue
            break
        tail = "\n".join(text.splitlines()[-15:])
        job.update(status="failed", exit_code=code, log_tail=tail)
        self.log(f"[FAILED] {name} (exit {code}). Last lines of {folder / 'train.log'}:\n{tail}")
        self.publish(job)
        return job

    def batch_of(self, job: dict) -> int:
        options = job["options"]
        return int(options[options.index("--batch") + 1]) if "--batch" in options else (256 if job["kind"] == "pretrain" else 128)

    def publish(self, job: dict) -> None:
        """Copy the small outputs of a job to share/ (never model weights), and rewrite the summary."""
        source, target = self.out / job["name"], self.share / job["name"]
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if path.suffix in (".json", ".npz") or path.name == "train.log":
                    shutil.copy2(path, target / path.name)
        self.write_summary()

    # -- planning ------------------------------------------------------------
    def common(self) -> list[str]:
        if self.smoke:
            return ["--folds", "0", "--epochs", "2", "--patience", "1", "--candidates", "shortlist_pool"]
        return []

    def seeds(self, n: int) -> list[str]:
        return ["--seeds", str(1 if self.smoke else n)]

    def core_jobs(self) -> list[dict]:
        c = self.common()
        S = lambda name, model, n, family, **kw: {"name": name, "kind": "structure", "model": model, "family": family,  # noqa: E731
                                                   "options": [*c, *self.seeds(n), *kw.pop("options", [])],
                                                   "networks": (1 if self.smoke else 5) * (1 if self.smoke else n), **kw}
        tab_common = ["--folds", "0", "--epochs", "3", "--patience", "1"] if self.smoke else []
        pre_options = ["--epochs", "2", "--limit", "2000"] if self.smoke else ["--epochs", "60"]
        return [
            {"name": "check", "kind": "structure", "model": "cgcnn", "family": "S", "networks": 1, "critical": True,
             "options": ["--folds", "0", "--seeds", "1", "--epochs", "2", "--patience", "1", "--candidates", "shortlist_pool"]},
            {"name": "tab-default-s20", "kind": "tabular", "family": "tabular", "networks": 5 if self.smoke else 100,
             "options": [*tab_common, "--seeds", "1" if self.smoke else "20"]},
            {"name": "tab-wide-s20", "kind": "tabular", "family": "tabular", "networks": 5 if self.smoke else 100,
             "options": [*tab_common, "--seeds", "1" if self.smoke else "20", "--hidden", "512,256,128", "--dropout", "0.2"]},
            S("A-formula_all", "formula_all", 3, "formula", track="A", control=True),
            S("A-hybrid_all", "hybrid_all", 3, "A", track="A"),
            {"name": "pretrain", "kind": "pretrain", "family": "pretrain", "networks": 1, "options": pre_options},
            S("A-hybrid_all-pre", "hybrid_all", 3, "A", track="A", pretrained=True),
            S("S-formula", "formula", 3, "formula", track="S", control=True),
            S("S-cgcnn", "cgcnn", 3, "S", track="S"),
            S("S-hybrid", "hybrid", 3, "S", track="S"),
            S("S-cgcnn-pre", "cgcnn", 3, "S", track="S", pretrained=True),
            S("S-hybrid-pre", "hybrid", 3, "S", track="S", pretrained=True),
        ]

    def best(self, track: str) -> dict | None:
        done = [j for j in self.jobs if j.get("track") == track and not j.get("control") and j.get("status") == "done"
                and j.get("mean_validation_bce") is not None and not j["name"].startswith("final")]
        return min(done, key=lambda j: j["mean_validation_bce"]) if done else None

    def sweep_jobs(self, base: dict) -> list[dict]:
        """One setting changed at a time from the better track A model."""
        if base.get("pretrained"):
            changes = [("lr5e-4", ["--lr", "5e-4"]), ("lr2e-3", ["--lr", "2e-3"]), ("drop0", ["--dropout", "0.0"]),
                       ("drop0.25", ["--dropout", "0.25"]), ("enc-lr1", ["--encoder-lr-scale", "1.0"]),
                       ("enc-lr0.1", ["--encoder-lr-scale", "0.1"]), ("aux0", ["--aux-weight", "0.0"]),
                       ("aux1", ["--aux-weight", "1.0"]), ("xh256", ["--x-hidden", "256"]), ("head256", ["--head", "256"]),
                       ("batch64", ["--batch", "64"]), ("wd1e-3", ["--weight-decay", "1e-3"])]
        else:
            changes = [("conv3", ["--conv", "3"]), ("conv6", ["--conv", "6"]), ("h96", ["--hidden", "96"]),
                       ("h192", ["--hidden", "192"]), ("lr5e-4", ["--lr", "5e-4"]), ("lr2e-3", ["--lr", "2e-3"]),
                       ("drop0", ["--dropout", "0.0"]), ("drop0.25", ["--dropout", "0.25"]), ("aux0", ["--aux-weight", "0.0"]),
                       ("xh256", ["--x-hidden", "256"]), ("batch64", ["--batch", "64"]), ("wd1e-3", ["--weight-decay", "1e-3"])]
        if self.smoke:
            changes = changes[:2]
        base_options = [o for o in base["options"] if True]
        out = []
        for tag, change in changes:
            options = strip_seeds(base_options) + self.seeds(2) + change
            out.append({"name": f"sweep-{tag}", "kind": "structure", "model": base["model"], "family": base["family"],
                        "track": "A", "pretrained": base.get("pretrained", False), "options": options,
                        "networks": 1 if self.smoke else 10, "change": change})
        return out

    def final_job(self, track: str, base: dict, seeds: int) -> dict:
        options = strip_seeds(base["options"]) + ["--seeds", str(seeds), "--seed-base", "100"]
        return {"name": f"final-{track}", "kind": "structure", "model": base["model"], "family": base["family"],
                "track": track, "pretrained": base.get("pretrained", False), "options": options,
                "networks": (1 if self.smoke else 5) * seeds, "chosen_from": base["name"],
                "chosen_validation_bce": base.get("mean_validation_bce")}

    def final_seeds(self, track: str, reserve: float) -> int:
        low, high = FINAL_SEEDS[track]
        if self.smoke:
            return 1
        per_seed = 5 * self.seconds_per_network(track) * 1.15
        fit = int((self.left() - reserve - 300.0) // per_seed)
        return max(low, min(high, fit))

    # -- the whole run ---------------------------------------------------------
    def main(self) -> int:
        self.log(f"=== overnight run, profile {self.args.profile}, {self.args.hours} h, device {self.args.device}, out {self.out}")
        self.write_env()
        keep_awake(True)
        try:
            for job in self.core_jobs():
                if self.stop_requested():
                    self.log("STOP file found: stopping before " + job["name"])
                    return self.finish(2)
                reserve = self.final_reserve(minimum=True)
                if not job.get("critical") and self.run_json(job["name"]) is None and now() + self.estimate(job) > self.deadline - reserve:
                    self.jobs.append({**job, "status": "skipped", "reason": "time budget"})
                    self.log(f"[skip] {job['name']}: would not leave time for the final runs")
                    continue
                self.jobs.append(self.execute(job))
                if job.get("critical") and self.jobs[-1]["status"] != "done":
                    self.log("The structure check failed, so nothing else is run. Fix the error above, then run the same command again.")
                    return self.finish(4)
            base = self.best("A")
            if base is not None:
                for job in self.sweep_jobs(base):
                    if self.stop_requested():
                        self.log("STOP file found: stopping the sweep")
                        return self.finish(2)
                    if self.run_json(job["name"]) is None and now() + self.estimate(job) > self.deadline - self.final_reserve(minimum=False):
                        self.jobs.append({**job, "status": "skipped", "reason": "time budget"})
                        self.log(f"[skip] {job['name']}: time budget")
                        continue
                    self.jobs.append(self.execute(job))
            for track in ("A", "S"):
                if self.stop_requested():
                    self.log("STOP file found: stopping before the final runs")
                    return self.finish(2)
                base = self.best(track)
                if base is None:
                    self.log(f"[skip] final-{track}: no finished {track} model to choose from")
                    continue
                reserve = self.final_reserve(minimum=True, only="S") if track == "A" else 0.0
                existing = self.run_json(f"final-{track}")
                seeds = existing["settings"]["seeds"] if existing else self.final_seeds(track, reserve)
                job = self.final_job(track, base, seeds)
                if existing is None and now() + self.estimate(job) > self.deadline + (900.0 if track == "A" else 0.0):
                    self.jobs.append({**job, "status": "skipped", "reason": "time budget"})
                    self.log(f"[skip] final-{track}: not enough time left")
                    continue
                self.log(f"final-{track}: {base['name']} has the lowest mean validation BCE ({base['mean_validation_bce']}); "
                         f"retraining it with {seeds} new networks per fold")
                self.jobs.append(self.execute(job))
            failed = any(j.get("status") == "failed" for j in self.jobs)
            return self.finish(1 if failed else 0)
        except KeyboardInterrupt:
            self.log("Interrupted (Ctrl+C). Run the same command again to continue.")
            return self.finish(2)
        finally:
            keep_awake(False)

    def final_reserve(self, minimum: bool, only: str | None = None) -> float:
        total = 0.0
        for track in ("A", "S"):
            if only and track != only:
                continue
            low, high = FINAL_SEEDS[track]
            seeds = 1 if self.smoke else (low if minimum else max(low, high // 2 + 1))
            total += (1 if self.smoke else 5) * seeds * self.seconds_per_network(track) * 1.15 + 30.0
        return total

    def finish(self, code: int) -> int:
        self.write_summary(code)
        self.log(f"=== finished with code {code}: {self.share / 'summary.json'}")
        return code

    def write_env(self) -> None:
        env = {"python": sys.version.split()[0], "platform": platform.platform(), "profile": self.args.profile}
        try:
            import torch

            env.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda)
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                env.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / 2**30, 2),
                           capability=f"{props.major}.{props.minor}")
        except Exception as error:  # noqa: BLE001
            env["torch_error"] = str(error)
        self.env = env
        (self.share / "env.json").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")
        self.log("environment: " + json.dumps(env))

    def write_summary(self, code: int | None = None) -> None:
        keep = ("name", "kind", "model", "track", "family", "pretrained", "control", "status", "reason", "minutes",
                "mean_validation_bce", "seconds_per_network", "change", "chosen_from", "chosen_validation_bce", "exit_code",
                "resumed")
        summary = {"profile": self.args.profile, "hours": self.args.hours, "device": self.args.device,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)), "updated": stamp(),
                   "exit_code": code, "env": getattr(self, "env", None),
                   "final": "final-A", "final_secondary": "final-S",
                   "formula_control": "A-formula_all", "secondary_control": "S-formula",
                   "selection": "lowest mean validation BCE (early-stopping rows only) within each track; controls excluded",
                   "jobs": [{k: j[k] for k in keep if k in j} for j in self.jobs]}
        (self.share / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        lines = [f"{'job':<24} {'status':<8} {'minutes':>8} {'val BCE':>8}"]
        for j in summary["jobs"]:
            bce = j.get("mean_validation_bce")
            lines.append(f"{j['name']:<24} {j.get('status', ''):<8} {j.get('minutes') or '':>8} "
                         f"{'' if bce is None else f'{bce:.4f}':>8}")
        (self.share / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if (self.out / "driver.log").is_file():
            shutil.copy2(self.out / "driver.log", self.share / "driver.log")


def strip_seeds(options: list[str]) -> list[str]:
    out, skip = [], False
    for o in options:
        if skip:
            skip = False
            continue
        if o in ("--seeds", "--seed-base"):
            skip = True
            continue
        out.append(o)
    return out


def keep_awake(on: bool) -> None:
    """Windows only: ask the system not to sleep while the run works (the display may still turn off)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:  # noqa: BLE001
        pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the run folder (created if missing)")
    parser.add_argument("--hours", type=float, default=8.0, help="time budget from now")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--formula", type=Path, default=DEFAULT_FORMULA)
    parser.add_argument("--python", default=sys.executable, help="the Python that runs the trainers")
    parser.add_argument("--dry-run", action="store_true", help="print the core jobs and exit")
    args = parser.parse_args(argv)
    for path in (args.structures, args.formula):
        if not path.is_file():
            parser.error(f"{path} not found (see docs/overnight-gpu.md, step 2)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    run = Run(args)
    if args.dry_run:
        for job in run.core_jobs():
            print(job["name"], " ".join(run.command(job)))
        return 0
    return run.main()


if __name__ == "__main__":
    raise SystemExit(main())
