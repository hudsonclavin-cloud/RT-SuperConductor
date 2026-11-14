"""Load the labelled table once and cache every derived array.

The legacy scripts recompute Magpie features (about a minute for the full
snapshot) on every run. Here the feature matrix is built once per input file
and stored as ``.npz`` keyed by the input's SHA-256, so repeat runs start in
well under a second.

The feature definition itself is imported from the legacy trainer on purpose:
an identical matrix is what makes the benchmark numbers comparable.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"
STRUCTURE = ROOT / "data/candidates/mp-structure-features.csv.gz"
CACHE = ROOT / "pipeline_v2/.cache"
FEATURE_VERSION = "1"  # bump when the feature definition changes
MAX_ELEMENTS = 8
N_ELEMENT_SLOTS = 119  # index 0 is padding, 1..118 are atomic numbers


@dataclass(frozen=True)
class Dataset:
    """Every row of the snapshot; ``label`` is NaN where no usable label exists."""

    ids: np.ndarray          # mp_id
    formulas: np.ndarray
    X: np.ndarray            # tabular features, NaN = missing
    feature_names: tuple[str, ...]
    el_index: np.ndarray     # (n, MAX_ELEMENTS) atomic numbers, 0 = padding
    el_fraction: np.ndarray  # (n, MAX_ELEMENTS) atomic fractions
    label: np.ndarray        # 1 superconductor, 0 usable negative, NaN otherwise
    tc: np.ndarray           # reported maximum Tc in kelvin, NaN if none
    groups: np.ndarray       # element system, e.g. "Nb-Sn"
    frame: pd.DataFrame      # the original columns, for filters and reports

    @property
    def labelled(self) -> np.ndarray:
        return np.isfinite(self.label)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.ids[mask], self.formulas[mask], self.X[mask], self.feature_names,
                       self.el_index[mask], self.el_fraction[mask], self.label[mask], self.tc[mask],
                       self.groups[mask], self.frame.loc[mask].reset_index(drop=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_labels(frame: pd.DataFrame) -> np.ndarray:
    """Same policy as ``build_mp_candidates.training_label``, vectorised.

    Superconductor -> 1. Tested non-superconductor -> 0 only when its evidence
    tier allows training. Untested materials are never negatives.
    """
    label = np.full(len(frame), np.nan)
    status = frame["known_status"].to_numpy()
    use = frame["negative_training_use"].fillna("").to_numpy() if "negative_training_use" in frame else None
    label[status == "superconductor"] = 1.0
    negative = status == "tested_non_superconductor"
    if use is not None:
        negative &= (use == "use") | (use == "")
    label[negative] = 0.0
    return label


def element_tokens(formulas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from pymatgen.core import Composition

    index = np.zeros((len(formulas), MAX_ELEMENTS), dtype=np.int64)
    fraction = np.zeros((len(formulas), MAX_ELEMENTS), dtype=np.float32)
    for row, formula in enumerate(formulas):
        parts = sorted(Composition(formula).fractional_composition.items(), key=lambda kv: -kv[1])
        for slot, (element, share) in enumerate(parts[:MAX_ELEMENTS]):
            index[row, slot], fraction[row, slot] = element.Z, share
    return index, fraction


def _legacy_features(rows: list[dict[str, str]], systems: list[str]) -> tuple[np.ndarray, list[str]]:
    for extra in (ROOT / "scripts", ROOT / "src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    from train_mp_sc_classifier import feature_matrix

    return feature_matrix(rows, systems)


def load(candidates: Path = CANDIDATES, *, structure: Path | None = None, cache_dir: Path = CACHE,
         labelled_only: bool = False) -> Dataset:
    """Build or reload the dataset. ``structure`` appends the per-material structure table."""
    frame = pd.read_csv(candidates, dtype=str, keep_default_na=False, low_memory=False)
    key = hashlib.sha256(f"{sha256(candidates)}:{FEATURE_VERSION}".encode()).hexdigest()[:16]
    cached = cache_dir / f"features-{key}.npz"
    if cached.is_file():
        stored = np.load(cached, allow_pickle=False)
        X, names = stored["X"], [str(n) for n in stored["names"]]
        el_index, el_fraction = stored["el_index"], stored["el_fraction"]
    else:
        rows = frame.to_dict("records")
        X, names = _legacy_features(rows, sorted(set(frame["crystal_system"])))
        el_index, el_fraction = element_tokens(frame["formula"].tolist())
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, X=X, names=np.asarray(names), el_index=el_index, el_fraction=el_fraction)
    if structure is not None:
        extra = pd.read_csv(structure).set_index("mp_id").reindex(frame["mp_id"])
        X = np.hstack([X, extra.to_numpy(dtype=float)])
        names = names + [f"structure__{c}" for c in extra.columns]
    blank = frame.replace("", np.nan)
    dataset = Dataset(
        ids=frame["mp_id"].to_numpy(), formulas=frame["formula"].to_numpy(), X=X, feature_names=tuple(names),
        el_index=el_index, el_fraction=el_fraction, label=training_labels(blank),
        tc=pd.to_numeric(blank["known_tc_max_k"], errors="coerce").to_numpy(dtype=float),
        groups=frame["elements"].map(lambda text: "-".join(sorted(text.split()))).to_numpy(), frame=frame)
    return dataset.subset(dataset.labelled) if labelled_only else dataset
