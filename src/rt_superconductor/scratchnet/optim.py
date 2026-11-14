"""Optimizer, learning-rate schedule and gradient utilities, in plain numpy."""
from __future__ import annotations

import math

import numpy as np


class AdamW:
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019).

    ``parameters`` is a list of (name, value, grad) with ``value`` updated in
    place. Weight decay is applied only to names in ``decay_names``; biases,
    LayerNorm scales and embeddings are normally left out.
    """

    def __init__(self, names: list[str], *, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, decay_names: set[str] | None = None):
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.decay_names = set(decay_names or ())
        self.step_count = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self._names = list(names)

    def step(self, parameters: list[tuple[str, np.ndarray, np.ndarray]], lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.step_count += 1
        b1, b2 = self.betas
        correction1 = 1.0 - b1 ** self.step_count
        correction2 = 1.0 - b2 ** self.step_count
        for name, value, grad in parameters:
            if name not in self.m:
                self.m[name] = np.zeros_like(value)
                self.v[name] = np.zeros_like(value)
            m, v = self.m[name], self.v[name]
            m *= b1
            m += (1.0 - b1) * grad
            v *= b2
            v += (1.0 - b2) * grad * grad
            if self.weight_decay and name in self.decay_names:
                value *= 1.0 - lr * self.weight_decay
            value -= lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def state(self) -> dict[str, np.ndarray]:
        out = {"step_count": np.asarray(self.step_count)}
        for name in self.m:
            out["m/" + name] = self.m[name]
            out["v/" + name] = self.v[name]
        return out

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.step_count = int(state["step_count"])
        self.m = {key[2:]: np.array(value) for key, value in state.items() if key.startswith("m/")}
        self.v = {key[2:]: np.array(value) for key, value in state.items() if key.startswith("v/")}


def warmup_cosine(step: int, total_steps: int, peak: float, warmup_steps: int, floor: float = 0.0) -> float:
    """Linear warm-up to ``peak`` then a cosine decay to ``floor``."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return peak
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> float:
    """Scale all gradients together so their combined norm is at most ``max_norm``. Returns the norm before clipping."""
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if max_norm and total > max_norm:
        factor = max_norm / (total + 1e-12)
        for g in grads:
            g *= factor
    return total


def numerical_gradient(loss_fn, array: np.ndarray, entries: list[tuple[int, ...]], eps: float = 1e-5) -> np.ndarray:
    """Central-difference derivative of ``loss_fn()`` with respect to chosen entries of ``array`` (edited in place)."""
    out = np.zeros(len(entries))
    for i, index in enumerate(entries):
        original = array[index]
        array[index] = original + eps
        up = loss_fn()
        array[index] = original - eps
        down = loss_fn()
        array[index] = original
        out[i] = (up - down) / (2.0 * eps)
    return out


def relative_error(a: np.ndarray, b: np.ndarray, floor: float = 1e-3) -> float:
    """max |a - b| / max(|a|, |b|, floor): the usual gradient-check score.

    The floor matters when the true gradient is zero: a central difference
    then returns rounding noise of about 1e-10, and dividing two tiny numbers
    would report a large "relative" error for a correct result. Below the
    floor the comparison is effectively absolute.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)))
