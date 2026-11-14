"""Turning chemical compositions into arrays for the from-scratch network.

A composition is a set of elements with atomic fractions that sum to one.
It becomes three aligned arrays padded to a common length N:

* ``index``     (B, N) int   row of each element in the element table, 0 for padding
* ``fraction``  (B, N) float atomic fraction, 0 for padding
* ``mask``      (B, N) bool  True where an element is real

Only numpy and the standard library are used here, so a trained model can be
applied anywhere numpy runs.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]+)?)")


def parse_formula(formula: str) -> dict[str, float]:
    """Atomic fractions from a formula such as ``La1.85Sr0.15CuO4`` or ``Ba(RhPb2)3``.

    Handles nested brackets and decimal amounts. Raises ValueError on anything
    it cannot read, so a bad formula is never silently turned into numbers.
    """
    text = formula.replace(" ", "").replace("[", "(").replace("]", ")")
    position = 0

    def group(depth: int) -> dict[str, float]:
        nonlocal position
        amounts: dict[str, float] = {}
        while position < len(text):
            char = text[position]
            if char == "(":
                position += 1
                inner = group(depth + 1)
                number = re.match(r"[0-9]*\.?[0-9]+", text[position:])
                factor = float(number.group(0)) if number else 1.0
                position += len(number.group(0)) if number else 0
                for element, amount in inner.items():
                    amounts[element] = amounts.get(element, 0.0) + amount * factor
            elif char == ")":
                if depth == 0:
                    raise ValueError(f"unbalanced bracket in {formula!r}")
                position += 1
                return amounts
            else:
                match = _TOKEN.match(text, position)
                if not match or not match.group(1):
                    raise ValueError(f"cannot read {formula!r} at position {position}")
                element, number = match.group(1), match.group(2)
                amounts[element] = amounts.get(element, 0.0) + (float(number) if number else 1.0)
                position = match.end()
        if depth:
            raise ValueError(f"unbalanced bracket in {formula!r}")
        return amounts

    amounts = {element: amount for element, amount in group(0).items() if amount > 0}
    total = sum(amounts.values())
    if not amounts or total <= 0:
        raise ValueError(f"no elements in {formula!r}")
    return {element: amount / total for element, amount in amounts.items()}


def composition_key(fractions: dict[str, float], digits: int = 3) -> str:
    """A canonical text form: elements in alphabetical order with rounded fractions."""
    return ";".join(f"{element}:{round(fraction, digits):.{digits}f}"
                    for element, fraction in sorted(fractions.items()))


def parse_key(key: str) -> dict[str, float]:
    out = {}
    for part in key.split(";"):
        element, fraction = part.split(":")
        out[element] = float(fraction)
    return out


def stable_hash(text: str) -> int:
    """An integer that depends only on the text, the same on every machine and run."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class ElementTable:
    """Fixed physical properties per element, standardized, with row 0 reserved for padding."""

    symbols: list[str]                 # row i + 1 belongs to symbols[i]
    feature_names: list[str]
    features: np.ndarray               # (len(symbols) + 1, F), row 0 all zero

    @property
    def index(self) -> dict[str, int]:
        return {symbol: i + 1 for i, symbol in enumerate(self.symbols)}

    @classmethod
    def load(cls, path: Path, dtype=np.float32) -> "ElementTable":
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        names = [name for name in rows[0] if name != "symbol"]
        raw = np.array([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
        mean, std = raw.mean(axis=0), raw.std(axis=0)
        scaled = (raw - mean) / np.where(std > 0, std, 1.0)
        features = np.vstack([np.zeros((1, len(names))), scaled]).astype(dtype)
        return cls([row["symbol"] for row in rows], names, features)


def encode(compositions: list[dict[str, float]], table: ElementTable, max_elements: int | None = None,
           dtype=np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compositions -> (index, fraction, mask), elements ordered by decreasing fraction."""
    lookup = table.index
    width = max_elements or max(len(c) for c in compositions)
    index = np.zeros((len(compositions), width), np.int64)
    fraction = np.zeros((len(compositions), width), dtype)
    for row, composition in enumerate(compositions):
        if len(composition) > width:
            raise ValueError(f"composition with {len(composition)} elements exceeds max_elements={width}")
        ordered = sorted(composition.items(), key=lambda item: (-item[1], item[0]))
        for column, (element, amount) in enumerate(ordered):
            if element not in lookup:
                raise KeyError(f"element {element!r} is not in the element table")
            index[row, column] = lookup[element]
            fraction[row, column] = amount
    return index, fraction, index > 0


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a CSV that may be gzip-compressed."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write a CSV, gzip-compressed without a timestamp when the name ends in .gz (so reruns are byte-identical)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle:
            handle.write(data)
    else:
        path.write_bytes(data)
