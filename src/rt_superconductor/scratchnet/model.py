"""The composition network: a chemical formula in, two predictions out.

The network reads a composition as a *set* of elements. Each element starts as
a vector built from three things: a learned embedding for that element, its
fixed physical properties (electronegativity, valence electrons and so on),
and how much of it there is. Self-attention blocks then let every element
adjust its vector in the light of the others (copper next to oxygen is not
the same as copper next to zinc). A pooling step collapses the set into one
vector, and two small heads read from it:

* ``logit``: a score whose sigmoid is the probability the material superconducts;
* ``tc``: the predicted critical temperature, in standardized log(1 + Tc) units.

With ``blocks = 0`` there is no attention and the model is a plain "deep sets"
network, which serves as the simpler comparison.

Two optional extras, both off by default:

* ``n_global``: whole-formula features (the same statistics gradient boosting
  uses) are passed through one small layer and joined to the pooled vector;
* ``n_aux``: extra outputs for auxiliary targets (for example the band gap and
  formation energy from the Materials Project). Learning to predict them from
  composition alone teaches the shared layers some chemistry that the
  superconductor labels do not cover.

Forward and backward passes are written out by hand from the layers in
``layers.py``. ``tests/test_scratchnet_model.py`` checks the whole backward
pass against finite differences and, when PyTorch is installed, against
PyTorch's automatic differentiation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import layers as L

LOG_FLOOR = 1e-6          # fractions are clipped here before taking a logarithm


@dataclass
class Config:
    n_elements: int = 96              # rows in the element table, not counting the padding row
    n_features: int = 28              # physical properties per element
    width: int = 64                   # size of each element's vector
    blocks: int = 2                   # self-attention blocks; 0 gives a deep-sets model
    heads: int = 4                    # attention heads per block
    pool_heads: int = 2               # attention-pooling heads
    trunk: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    use_embedding: bool = True        # learned vector per element
    use_features: bool = True         # physical properties per element
    fraction_bias: bool = True        # add log(fraction) to attention scores
    n_global: int = 0                 # whole-formula features joined to the pooled vector; 0 turns this off
    n_aux: int = 0                    # auxiliary regression outputs; 0 turns this off

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Config":
        raw = json.loads(text)
        raw["trunk"] = tuple(raw["trunk"])
        return cls(**raw)


@dataclass
class _Block:
    attention: L.SelfAttention
    norm1: L.LayerNorm
    drop1: L.Dropout
    ff1: L.Linear
    act: L.SiLU
    ff2: L.Linear
    norm2: L.LayerNorm
    drop2: L.Dropout
    cache: dict = field(default_factory=dict)


class CompositionNet:
    def __init__(self, config: Config, element_features: np.ndarray, seed: int = 0, dtype=np.float32):
        if element_features.shape != (config.n_elements + 1, config.n_features):
            raise ValueError(f"element_features must have shape {(config.n_elements + 1, config.n_features)}, "
                             f"got {element_features.shape}")
        if not (config.use_embedding or config.use_features):
            raise ValueError("at least one of use_embedding and use_features must be on")
        self.config, self.dtype = config, dtype
        self.element_features = element_features.astype(dtype)
        rng = np.random.default_rng(seed)
        self.dropout_rng = np.random.default_rng(seed + 1)
        w = config.width
        self.layers: dict[str, L.Layer] = {}
        if config.use_embedding:
            self.layers["embedding"] = L.Embedding(config.n_elements + 1, w, rng, dtype)
        if config.use_features:
            self.layers["feature_proj"] = L.Linear(config.n_features, w, rng, dtype, bias=False)
        self.layers["fraction_proj"] = L.Linear(3, w, rng, dtype)
        self.layers["input_norm"] = L.LayerNorm(w, dtype)
        self.layers["input_drop"] = L.Dropout(config.dropout, self.dropout_rng)
        self.blocks: list[_Block] = []
        for i in range(config.blocks):
            block = _Block(L.SelfAttention(w, config.heads, rng, dtype), L.LayerNorm(w, dtype),
                           L.Dropout(config.dropout, self.dropout_rng), L.Linear(w, 2 * w, rng, dtype), L.SiLU(),
                           L.Linear(2 * w, w, rng, dtype), L.LayerNorm(w, dtype),
                           L.Dropout(config.dropout, self.dropout_rng))
            self.blocks.append(block)
            for name in ("attention", "norm1", "drop1", "ff1", "act", "ff2", "norm2", "drop2"):
                self.layers[f"block{i}.{name}"] = getattr(block, name)
        self.layers["pool"] = L.AttentionPool(w, config.pool_heads, rng, dtype)
        previous = w * (config.pool_heads + 1)
        self.trunk: list[tuple[L.Linear, L.SiLU, L.Dropout]] = []
        for i, size in enumerate(config.trunk):
            triple = (L.Linear(previous, size, rng, dtype), L.SiLU(), L.Dropout(config.dropout, self.dropout_rng))
            self.trunk.append(triple)
            self.layers[f"trunk{i}.linear"], self.layers[f"trunk{i}.act"], self.layers[f"trunk{i}.drop"] = triple
            previous = size
        self.layers["head_class"] = L.Linear(previous, 1, rng, dtype)
        self.layers["head_tc"] = L.Linear(previous, 1, rng, dtype)
        # The optional parts are created last so that switching them on does not change how the rest is initialized.
        if config.n_aux:
            self.layers["head_aux"] = L.Linear(previous, config.n_aux, rng, dtype)
        if config.n_global:
            self.layers["global_proj"] = L.Linear(config.n_global, w, rng, dtype)
            self.layers["global_act"] = L.SiLU()
            self.layers["global_drop"] = L.Dropout(config.dropout, self.dropout_rng)
            first = self.trunk[0][0] if self.trunk else None
            if first is None:
                raise ValueError("n_global needs at least one trunk layer")
            # Widen the first trunk layer to take the extra vector, leaving its existing weights as they were.
            first.params["W"] = np.concatenate([first.params["W"], L.glorot(rng, w, first.params["W"].shape[1], dtype)])
            first.grads["W"] = np.zeros_like(first.params["W"])
        self.aux: np.ndarray | None = None

    # ------------------------------------------------------------------ parameters

    def parameters(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(name, value, gradient) for every trainable array, in a fixed order."""
        return [(f"{layer_name}.{name}", layer.params[name], layer.grads[name])
                for layer_name, layer in self.layers.items() for name in layer.params]

    def parameter_count(self) -> int:
        return sum(value.size for _, value, _ in self.parameters())

    def decay_names(self) -> set[str]:
        """Weight matrices get weight decay; biases, LayerNorm scales and the embedding table do not."""
        return {name for name, value, _ in self.parameters()
                if value.ndim == 2 and not name.startswith("embedding.")}

    def state(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value, _ in self.parameters()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        for name, value, _ in self.parameters():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"saved state does not match the model at {name}")
            value[...] = state[name]

    def train(self, flag: bool = True) -> None:
        for layer in self.layers.values():
            layer.training = flag

    # ------------------------------------------------------------------ forward

    def forward(self, index: np.ndarray, fraction: np.ndarray, mask: np.ndarray,
                global_features: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(logit, standardized Tc). Auxiliary outputs, when the model has them, are left in ``self.aux``."""
        config, layers = self.config, self.layers
        fraction = fraction.astype(self.dtype)
        m = mask[..., None].astype(self.dtype)
        log_fraction = np.log(np.maximum(fraction, LOG_FLOOR)) * mask
        amount = np.stack([fraction, log_fraction / 4.0, np.sqrt(fraction)], axis=-1).astype(self.dtype)
        x = layers["fraction_proj"].forward(amount)
        if config.use_embedding:
            x = x + layers["embedding"].forward(index)
        if config.use_features:
            x = x + layers["feature_proj"].forward(self.element_features[index])
        x = layers["input_norm"].forward(x * m) * m
        x = layers["input_drop"].forward(x)
        bias = log_fraction if config.fraction_bias else None
        for block in self.blocks:
            attended = block.drop1.forward(block.attention.forward(x, mask, bias))
            x = block.norm1.forward(x + attended) * m
            hidden = block.drop2.forward(block.ff2.forward(block.act.forward(block.ff1.forward(x))))
            x = block.norm2.forward(x + hidden) * m
        self._m, self._fraction = m, fraction
        parts = [layers["pool"].forward(x, mask, bias), L.weighted_mean_forward(x, fraction)]
        if config.n_global:
            if global_features is None or global_features.shape != (len(index), config.n_global):
                raise ValueError(f"this model needs global_features of shape ({len(index)}, {config.n_global})")
            projected = layers["global_proj"].forward(global_features.astype(self.dtype))
            parts.append(layers["global_drop"].forward(layers["global_act"].forward(projected)))
        t = np.concatenate(parts, axis=1)
        for linear, act, drop in self.trunk:
            t = drop.forward(act.forward(linear.forward(t)))
        self.aux = layers["head_aux"].forward(t) if config.n_aux else None
        return layers["head_class"].forward(t)[:, 0], layers["head_tc"].forward(t)[:, 0]

    # ------------------------------------------------------------------ backward

    def backward(self, d_logit: np.ndarray, d_tc: np.ndarray, d_aux: np.ndarray | None = None) -> None:
        """Push the loss gradients for the outputs back through every layer."""
        config, layers, m = self.config, self.layers, self._m
        dt = (layers["head_class"].backward(d_logit[:, None].astype(self.dtype))
              + layers["head_tc"].backward(d_tc[:, None].astype(self.dtype)))
        if config.n_aux:
            if d_aux is None:
                d_aux = np.zeros((len(d_logit), config.n_aux))
            dt = dt + layers["head_aux"].backward(d_aux.astype(self.dtype))
        for linear, act, drop in reversed(self.trunk):
            dt = linear.backward(act.backward(drop.backward(dt)))
        split, set_end = config.width * config.pool_heads, config.width * (config.pool_heads + 1)
        dx = layers["pool"].backward(dt[:, :split]) + L.weighted_mean_backward(dt[:, split:set_end], self._fraction)
        if config.n_global:
            layers["global_proj"].backward(layers["global_act"].backward(layers["global_drop"].backward(dt[:, set_end:])))
        for block in reversed(self.blocks):
            d_sum = block.norm2.backward(dx * m)
            d_hidden = block.ff1.backward(block.act.backward(block.ff2.backward(block.drop2.backward(d_sum))))
            d_sum = block.norm1.backward((d_sum + d_hidden) * m)
            dx = d_sum + block.attention.backward(block.drop1.backward(d_sum))
        dx = layers["input_norm"].backward(layers["input_drop"].backward(dx) * m) * m
        layers["fraction_proj"].backward(dx)
        if config.use_embedding:
            layers["embedding"].backward(dx)
        if config.use_features:
            layers["feature_proj"].backward(dx)

    # ------------------------------------------------------------------ convenience

    def predict(self, index: np.ndarray, fraction: np.ndarray, mask: np.ndarray,
                global_features: np.ndarray | None = None, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation-mode outputs (dropout off) for many compositions."""
        was_training = self.layers["input_drop"].training
        self.train(False)
        logits, tcs = [], []
        for start in range(0, len(index), batch_size):
            stop = start + batch_size
            extra = None if global_features is None else global_features[start:stop]
            logit, tc = self.forward(index[start:stop], fraction[start:stop], mask[start:stop], extra)
            logits.append(logit)
            tcs.append(tc)
        self.train(was_training)
        return np.concatenate(logits), np.concatenate(tcs)

    def save(self, path: Path, extra: dict | None = None) -> None:
        arrays = {"param/" + name: value for name, value in self.state().items()}
        arrays["config_json"] = np.asarray(self.config.to_json())
        arrays["extra_json"] = np.asarray(json.dumps(extra or {}, sort_keys=True))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path, element_features: np.ndarray, dtype=np.float32) -> tuple["CompositionNet", dict]:
        with np.load(Path(path), allow_pickle=False) as data:
            config = Config.from_json(str(data["config_json"]))
            model = cls(config, element_features, seed=0, dtype=dtype)
            model.load_state({key[6:]: data[key] for key in data.files if key.startswith("param/")})
            extra = json.loads(str(data["extra_json"]))
        return model, extra
