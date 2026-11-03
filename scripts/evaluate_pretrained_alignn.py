#!/usr/bin/env python3
"""Evaluate frozen pretrained ALIGNN structure embeddings for EPC targets.

The ALIGNN checkpoint is pretrained on JARVIS formation energy, not lambda or
omega_log. We freeze the graph encoder, capture its pooled graph embedding,
then fit only random-forest probes inside the same elemental-system GroupKFold
used by the composition models. This avoids using the EPC-specific ALIGNN
checkpoint that was trained on this evaluation dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from train_physics_targets import (
    DATASET_FILE, DATASET_SHA256, DATASET_URL, allen_dynes_tc_k, _element_system,
)

ALIGNN_WEIGHTS_URL = "https://ndownloader.figshare.com/files/31458679"
ALIGNN_WEIGHTS_SHA256 = "d80f5e067c0b0b0f569b8158ecfeeeb68472bebe2ceb27ff787dcfcc68948e20"
ALIGNN_WEIGHT_FILE = "jv_formation_energy_peratom_alignn.zip"


def _load_rows(dataset_zip: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(dataset_zip) as archive:
        rows = json.load(archive.open("jarvis_epc_data_figshare_1058.json"))
    return [row for row in rows if float(row.get("lamb") or 0) > 0 and float(row.get("wlog") or 0) > 0]


def _load_frozen_alignn(weight_path: Path):
    import torch
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    with zipfile.ZipFile(weight_path) as archive:
        names = archive.namelist()
        config_name = next(name for name in names if name.endswith("config.json"))
        checkpoint_name = next((name for name in names if "best_model.pt" in name), None)
        if checkpoint_name is None:
            checkpoints = [name for name in names if "checkpoint_" in name and name.endswith(".pt")]
            if not checkpoints:
                raise ValueError("ALIGNN archive has no checkpoint")
            checkpoint_name = sorted(checkpoints)[-1]
        config = json.loads(archive.read(config_name))
        checkpoint = torch.load(io.BytesIO(archive.read(checkpoint_name)), map_location="cpu", weights_only=False)
    model = ALIGNN(ALIGNNConfig(**config["model"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _extract_embeddings(rows: list[dict[str, Any]], model) -> tuple[np.ndarray, list[int]]:
    import torch
    from alignn.graphs import Graph
    from jarvis.core.atoms import Atoms

    device = torch.device("cpu")
    model.to(device)
    captured: list[np.ndarray] = []
    handle = model.fc.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().cpu().numpy().reshape(-1))
    )
    embeddings: list[np.ndarray] = []
    valid_indices: list[int] = []
    for index, row in enumerate(rows):
        try:
            atoms = Atoms.from_dict(row["atoms"])
            graph, line_graph = Graph.atom_dgl_multigraph(
                atoms, cutoff=8.0, max_neighbors=12,
                atom_features="cgcnn", compute_line_graph=True,
            )
            lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
            captured.clear()
            with torch.no_grad():
                model([graph.to(device), line_graph.to(device), lattice.to(device)])
            if not captured or not np.isfinite(captured[-1]).all():
                continue
            embeddings.append(captured[-1].astype(np.float32, copy=False))
            valid_indices.append(index)
        except Exception as exc:
            # A malformed/unsupported graph is omitted and reported, never
            # replaced with a composition-only vector.
            print(f"ALIGNN structure failed for {row.get('jid')}: {type(exc).__name__}", file=sys.stderr)
        if (index + 1) % 100 == 0:
            print(f"ALIGNN embedding progress: {index + 1}/{len(rows)} structures; {len(embeddings)} usable", flush=True)
    handle.remove()
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), valid_indices
    return np.vstack(embeddings), valid_indices


def _metric(actual: np.ndarray, predicted: np.ndarray, unit: str) -> dict[str, float | str]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "unit": unit,
    }


def evaluate_embeddings(rows: list[dict[str, Any]], embeddings: np.ndarray, indices: list[int]) -> dict[str, Any]:
    selected = [rows[index] for index in indices]
    lam = np.asarray([float(row["lamb"]) for row in selected])
    wlog = np.asarray([float(row["wlog"]) for row in selected])
    target_tc = np.asarray([allen_dynes_tc_k(a, b) for a, b in zip(lam, wlog)])
    groups = np.asarray([_element_system(_reduced_formula(row)) for row in selected])
    # _element_system handles normalized formula strings. JARVIS grouped
    # formulas are flattened using their site list to avoid parser ambiguity.
    splitter = GroupKFold(n_splits=5)
    pred_lambda = np.full(len(selected), np.nan)
    pred_log_wlog = np.full(len(selected), np.nan)
    pred_direct_tc = np.full(len(selected), np.nan)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(embeddings, target_tc, groups), 1):
        direct = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=2026 + fold, n_jobs=-1)
        lambda_model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=3026 + fold, n_jobs=-1)
        wlog_model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=4026 + fold, n_jobs=-1)
        direct.fit(embeddings[train], target_tc[train])
        lambda_model.fit(embeddings[train], lam[train])
        wlog_model.fit(embeddings[train], np.log(wlog[train]))
        pred_direct_tc[test] = direct.predict(embeddings[test])
        pred_lambda[test] = lambda_model.predict(embeddings[test])
        pred_log_wlog[test] = np.exp(wlog_model.predict(embeddings[test]))
        folds.append({
            "fold": fold, "train_rows": int(len(train)), "test_rows": int(len(test)),
            "held_out_elemental_systems": sorted(set(groups[test])),
        })
    pred_physics_tc = np.asarray([allen_dynes_tc_k(a, b) for a, b in zip(pred_lambda, pred_log_wlog)])
    return {
        "status": "ready", "rows_used": len(selected), "rows_dropped_graph_failure": len(rows) - len(selected),
        "elemental_system_groups": len(set(groups)), "split": "5-fold GroupKFold by elemental-system signature",
        "feature_count": int(embeddings.shape[1]),
        "metrics": {
            "direct_tc": _metric(target_tc, pred_direct_tc, "K"),
            "physics_tc_from_lambda_omega_log": _metric(target_tc, pred_physics_tc, "K"),
            "lambda": _metric(lam, pred_lambda, "dimensionless"),
            "omega_log": _metric(wlog, pred_log_wlog, "K"),
        },
        "folds": folds,
        "pretraining": {
            "model": "jv_formation_energy_peratom_alignn",
            "weights_url": ALIGNN_WEIGHTS_URL,
            "weights_sha256": ALIGNN_WEIGHTS_SHA256,
            "use": "frozen ALIGNN graph representation; random-forest EPC probes trained fold-wise",
            "warning": "The generic JARVIS formation-energy pretraining corpus may contain test structures; the EPC targets are not used by this checkpoint, but this remains a transductive pretraining caveat.",
        },
    }


def _reduced_formula(row: dict[str, Any]) -> str:
    from pymatgen.core import Composition

    amounts: dict[str, int] = {}
    for element in row["atoms"]["elements"]:
        amounts[element] = amounts.get(element, 0) + 1
    return Composition(amounts).reduced_formula


def run(data_dir: Path, output_dir: Path, weight_path: Path | None = None) -> dict[str, Any]:
    dataset_path = data_dir / DATASET_FILE
    if not dataset_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, dataset_path)
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_sha != DATASET_SHA256:
        raise ValueError("JARVIS-SuperconDB checksum mismatch before graph evaluation")
    if weight_path is None:
        weight_path = data_dir / ALIGNN_WEIGHT_FILE
    if not weight_path.exists():
        urllib.request.urlretrieve(ALIGNN_WEIGHTS_URL, weight_path)
    weights_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if weights_sha != ALIGNN_WEIGHTS_SHA256:
        raise ValueError(f"ALIGNN pretrained weight checksum mismatch: {weights_sha}")
    rows = _load_rows(dataset_path)
    model = _load_frozen_alignn(weight_path)
    embeddings, indices = _extract_embeddings(rows, model)
    report = evaluate_embeddings(rows, embeddings, indices)
    report["dataset"] = {"doi": "10.6084/m9.figshare.21370572.v4", "sha256": dataset_sha}
    report["artifacts"] = {"weights_sha256": weights_sha, "embedding_rows": len(indices)}
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pretrained-alignn-physics-target-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/jarvis-supercond"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/physics-target"))
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    report = run(args.data_dir, args.output_dir, args.weights)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
