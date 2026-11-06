#!/usr/bin/env python3
"""Per-material feature tables keyed by Materials Project ID."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np


def load_feature_table(path: Path, mp_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return a matrix aligned to ``mp_ids``; missing materials or values become NaN."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = [n for n in (reader.fieldnames or []) if n != "mp_id"]
        table = {row["mp_id"]: row for row in reader}

    def value(row: dict[str, str] | None, name: str) -> float:
        try:
            return float(row[name]) if row is not None and row.get(name) not in (None, "") else float("nan")
        except ValueError:
            return float("nan")

    matrix = np.asarray([[value(table.get(mid), n) for n in names] for mid in mp_ids], dtype=float)
    return matrix, names
