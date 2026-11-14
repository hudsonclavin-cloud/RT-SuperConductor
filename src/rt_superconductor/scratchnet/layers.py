"""Neural-network layers written from scratch in numpy.

Every layer has a ``forward`` that remembers what it needs and a ``backward``
that takes the gradient of the loss with respect to the layer's output,
stores the gradients of its own parameters in ``self.grads`` and returns the
gradient with respect to its input. There is no automatic differentiation:
each backward pass is the chain rule written out by hand, and
``tests/test_scratchnet_layers.py`` checks every one against finite
differences.

Shapes use B for the batch, N for the (padded) number of elements in a
composition and D for a feature width. ``mask`` is a boolean array of shape
(B, N) that is True where an element is real and False where it is padding.
"""
from __future__ import annotations

import numpy as np

NEG_INF = -1e30   # added to attention scores of padded positions before the softmax


class Layer:
    """Base class: parameters and their gradients live in two dictionaries."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training = True

    def zero_grads(self) -> None:
        for name, value in self.params.items():
            self.grads[name] = np.zeros_like(value)


def glorot(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=np.float32) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(dtype)


def masked_softmax(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Softmax over the last axis that gives exactly zero weight where ``valid`` is False.

    Rows with no valid entry return all zeros instead of NaN.
    """
    scores = np.where(valid, scores, NEG_INF)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores) * valid
    total = weights.sum(axis=-1, keepdims=True)
    return weights / np.where(total > 0, total, 1.0)


class Linear(Layer):
    """y = x W + b on the last axis."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32, bias: bool = True):
        super().__init__()
        self.params["W"] = glorot(rng, fan_in, fan_out, dtype)
        if bias:
            self.params["b"] = np.zeros(fan_out, dtype)
        self.zero_grads()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.params["W"]
        if "b" in self.params:
            y = y + self.params["b"]
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x2 = self._x.reshape(-1, self._x.shape[-1])
        d2 = dout.reshape(-1, dout.shape[-1])
        self.grads["W"] = x2.T @ d2
        if "b" in self.params:
            self.grads["b"] = d2.sum(axis=0)
        return dout @ self.params["W"].T


class Embedding(Layer):
    """A table of vectors looked up by integer index. Row 0 is the padding row and stays zero."""

    def __init__(self, count: int, width: int, rng: np.random.Generator, dtype=np.float32, scale: float = 0.1):
        super().__init__()
        table = rng.normal(0.0, scale, size=(count, width)).astype(dtype)
        table[0] = 0.0
        self.params["E"] = table
        self.zero_grads()

    def forward(self, index: np.ndarray) -> np.ndarray:
        self._index = index
        return self.params["E"][index]

    def backward(self, dout: np.ndarray) -> None:
        grad = np.zeros_like(self.params["E"])
        np.add.at(grad, self._index.reshape(-1), dout.reshape(-1, dout.shape[-1]))
        grad[0] = 0.0
        self.grads["E"] = grad


class LayerNorm(Layer):
    """Normalize the last axis to zero mean and unit variance, then scale and shift."""

    def __init__(self, width: int, dtype=np.float32, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.params["gamma"] = np.ones(width, dtype)
        self.params["beta"] = np.zeros(width, dtype)
        self.zero_grads()

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        self._inv_std = 1.0 / np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + self.eps)
        self._xhat = centred * self._inv_std
        return self._xhat * self.params["gamma"] + self.params["beta"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat = self._xhat
        flat = dout.reshape(-1, dout.shape[-1])
        self.grads["gamma"] = (flat * xhat.reshape(flat.shape)).sum(axis=0)
        self.grads["beta"] = flat.sum(axis=0)
        dxhat = dout * self.params["gamma"]
        return self._inv_std * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                                - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


class SiLU(Layer):
    """x * sigmoid(x): a smooth relative of ReLU."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._s = sigmoid(x)
        return x * self._s

    def backward(self, dout: np.ndarray) -> np.ndarray:
        s = self._s
        return dout * (s + self._x * s * (1.0 - s))


class Dropout(Layer):
    """Zero a random share of the inputs during training and rescale the rest."""

    def __init__(self, rate: float, rng: np.random.Generator):
        super().__init__()
        self.rate = float(rate)
        self.rng = rng

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate <= 0.0:
            self._keep = None
            return x
        self._keep = (self.rng.random(x.shape) >= self.rate).astype(x.dtype) / (1.0 - self.rate)
        return x * self._keep

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout if self._keep is None else dout * self._keep


class SelfAttention(Layer):
    """Masked multi-head scaled dot-product self-attention.

    Each element of a composition looks at every other element and takes a
    weighted average of their vectors. ``key_bias`` (B, N) is added to the
    score of each element being looked at; passing log(fraction) there makes
    abundant elements count for more, as in Roost and CrabNet.

    The key projection has no bias: adding the same vector to every key shifts
    all scores in a row equally, the softmax cancels it, and the parameter
    would never receive a gradient.
    """

    def __init__(self, width: int, heads: int, rng: np.random.Generator, dtype=np.float32):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width, self.heads, self.head_width = width, heads, width // heads
        for name in ("Wq", "Wk", "Wv", "Wo"):
            self.params[name] = glorot(rng, width, width, dtype)
        for name in ("bq", "bv", "bo"):
            self.params[name] = np.zeros(width, dtype)
        self.zero_grads()

    def _split(self, x: np.ndarray) -> np.ndarray:        # (B, N, D) -> (B, H, N, D/H)
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.head_width).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:        # (B, H, N, D/H) -> (B, N, D)
        b, _, n, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, self.width)

    def forward(self, x: np.ndarray, mask: np.ndarray, key_bias: np.ndarray | None = None) -> np.ndarray:
        p = self.params
        self._x, self._mask = x, mask
        q = self._split(x @ p["Wq"] + p["bq"])
        k = self._split(x @ p["Wk"])
        v = self._split(x @ p["Wv"] + p["bv"])
        scale = float(1.0 / np.sqrt(self.head_width))     # a Python float: a numpy float64 here would promote float32 arrays
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale                      # (B, H, N, N)
        if key_bias is not None:
            scores = scores + key_bias[:, None, None, :]
        attention = masked_softmax(scores, mask[:, None, None, :])
        merged = self._merge(attention @ v)
        self._q, self._k, self._v, self._attention, self._merged, self._scale = q, k, v, attention, merged, scale
        return (merged @ p["Wo"] + p["bo"]) * mask[..., None]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        p = self.params
        x = self._x
        dout = dout * self._mask[..., None]
        flat = lambda a: a.reshape(-1, a.shape[-1])           # noqa: E731
        self.grads["Wo"] = flat(self._merged).T @ flat(dout)
        self.grads["bo"] = flat(dout).sum(axis=0)
        d_heads = self._split(dout @ p["Wo"].T)                              # (B, H, N, D/H)
        a = self._attention
        d_attention = d_heads @ self._v.transpose(0, 1, 3, 2)                # (B, H, N, N)
        d_v = a.transpose(0, 1, 3, 2) @ d_heads
        d_scores = a * (d_attention - (d_attention * a).sum(axis=-1, keepdims=True))
        d_q = (d_scores @ self._k) * self._scale
        d_k = (d_scores.transpose(0, 1, 3, 2) @ self._q) * self._scale
        dx = np.zeros_like(x)
        for name, grad in (("q", d_q), ("k", d_k), ("v", d_v)):
            merged = self._merge(grad)
            self.grads["W" + name] = flat(x).T @ flat(merged)
            if "b" + name in p:
                self.grads["b" + name] = flat(merged).sum(axis=0)
            dx = dx + merged @ p["W" + name].T
        return dx


class AttentionPool(Layer):
    """Collapse the elements of a composition into one vector per pooling head.

    Each head scores every element with a learned vector, adds ``bias``
    (log fraction) and takes the softmax-weighted average of the element
    vectors. Output shape (B, heads * D).
    """

    def __init__(self, width: int, heads: int, rng: np.random.Generator, dtype=np.float32):
        super().__init__()
        self.width, self.heads = width, heads
        self.params["W"] = glorot(rng, width, heads, dtype)
        self.params["b"] = np.zeros(heads, dtype)
        self.zero_grads()

    def forward(self, h: np.ndarray, mask: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        self._h, self._mask = h, mask
        scores = h @ self.params["W"] + self.params["b"]                     # (B, N, heads)
        if bias is not None:
            scores = scores + bias[..., None]
        weights = masked_softmax(scores.transpose(0, 2, 1), mask[:, None, :])   # (B, heads, N)
        self._weights = weights
        pooled = weights @ h                                                 # (B, heads, D)
        return pooled.reshape(h.shape[0], self.heads * self.width)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h, weights = self._h, self._weights
        d_pooled = dout.reshape(h.shape[0], self.heads, self.width)          # (B, heads, D)
        dh = weights.transpose(0, 2, 1) @ d_pooled                           # direct path, (B, N, D)
        d_weights = d_pooled @ h.transpose(0, 2, 1)                          # (B, heads, N)
        d_scores = weights * (d_weights - (d_weights * weights).sum(axis=-1, keepdims=True))
        d_scores = d_scores.transpose(0, 2, 1)                               # (B, N, heads)
        flat_h = h.reshape(-1, self.width)
        flat_s = d_scores.reshape(-1, self.heads)
        self.grads["W"] = flat_h.T @ flat_s
        self.grads["b"] = flat_s.sum(axis=0)
        return dh + d_scores @ self.params["W"].T


def weighted_mean_forward(h: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average of element vectors weighted by fraction: (B, N, D), (B, N) -> (B, D)."""
    return (h * weights[..., None]).sum(axis=1)


def weighted_mean_backward(dout: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return dout[:, None, :] * weights[..., None]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-x)) without overflow for large |x|."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def bce_with_logits(logits: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted mean binary cross-entropy from raw scores, and its gradient.

    ``weight`` is 0 for rows that have no class label; the mean is over the
    total weight, so unlabelled rows neither add loss nor dilute it.
    """
    total = float(weight.sum())
    if total <= 0:
        return 0.0, np.zeros_like(logits)
    per_row = np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    loss = float((per_row * weight).sum() / total)
    return loss, (sigmoid(logits) - target) * weight / total


def huber(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray, delta: float = 1.0) -> tuple[float, np.ndarray]:
    """Weighted mean Huber loss (squared near zero, linear beyond ``delta``) and its gradient."""
    total = float(weight.sum())
    if total <= 0:
        return 0.0, np.zeros_like(prediction)
    residual = prediction - target
    small = np.abs(residual) <= delta
    per_row = np.where(small, 0.5 * residual ** 2, delta * (np.abs(residual) - 0.5 * delta))
    loss = float((per_row * weight).sum() / total)
    return loss, np.where(small, residual, delta * np.sign(residual)) * weight / total
