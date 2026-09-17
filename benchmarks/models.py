"""
benchmarks.models
-----------------
Model factory used by the benchmark harness.

Two families of baseline:
- Matched-bottleneck (Linear -> hidden=6 -> activation -> Linear).
  Every model in this family sees the same compression before the
  activation runs, so accuracy comparisons isolate the effect of the
  activation itself. Used for the "scientific" comparison.
- Natural-width MLP (Linear(in -> H) -> activation -> Linear(H -> out)).
  Standard 256-wide baseline that does NOT bottleneck through 6.
  Used for the "engineering" comparison: is the geometric model
  competitive in practice given its hard 6-dim constraint?

The factory takes a string name + dimensions and returns an ``nn.Module``.
Every model exposes ``regularization_loss()``; pointwise activations get a
no-op via BottleneckClassifier or NaturalWidthMLP.

``eta_invariants`` is kept here as the SO(3,3)-invariant readout baseline
that every canonical top-tagging table is anchored on. It needs only the
metric eta and the deterministic lift below, not the SO(3,3) activation,
which lives in the separate so33 repository (see REMOVED.md).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

# eta = diag(+1,+1,+1,-1,-1,-1) on R^{3,3}. so3c.algebra.ETA is the same
# tensor -- Re(z . z) = v^T eta v -- and tests/test_so3c_algebra.py pins its
# value, so the baseline below reproduces its recorded numbers exactly.
from so3c.algebra import ETA

DIM = 6                                  # dimension of R^{3,3}


def _lift_4to6(p4: torch.Tensor) -> torch.Tensor:
    """Equivariant lift of a (..., 4) Minkowski 4-vector into R^{3,3}.

    Layout matches ETA = diag(+1,+1,+1,-1,-1,-1):
        out[..., 0:3] = (px, py, pz)        spacelike (+)
        out[...,   3] = E                    timelike  (-)
        out[..., 4:6] = 0                    unused timelike axes

    This is deterministic (no learned parameters), so SO(3,3) boosts
    acting on the lifted vector correspond exactly to the standard
    Lorentz action on the original (E, p). A learned Linear(4->6) would
    not commute with SO(3,3), which is why the earlier OOD experiment
    showed no advantage for models wrapped in one.
    """
    shape = p4.shape[:-1] + (DIM,)
    out = p4.new_zeros(shape)
    out[..., 0] = p4[..., 1]
    out[..., 1] = p4[..., 2]
    out[..., 2] = p4[..., 3]
    out[..., 3] = p4[..., 0]
    return out


# Names the factory understands.
MATCHED_MODELS = (
    "relu_bottleneck",
    "tanh_bottleneck",
    "gelu_bottleneck",
    "so3c",                # complexified-SO(3) bottleneck, dynamic metric
    "so3c_static",         # complexified-SO(3) bottleneck, 6 static coeffs
)
NATURAL_MODELS = (
    "relu_mlp",
    "tanh_mlp",
    "gelu_mlp",
    "so3c_multi",          # multi-channel so3c: capacity comparison, exact flow
)
# so3c names support representation="flat" only (no Deep-Sets path yet).
SO3C_MODELS = ("so3c", "so3c_static", "so3c_multi")
SET_MODELS = (
    # Set-based classifiers — only valid for representation="constituents".
    "eta_invariants",        # SO(3,3)-invariant features: the anchor baseline
    "so3c_invariant_set",    # bivector-lift complex invariants (so3c Arch A)
    "so3c_equivariant_set",  # + channel lift & geodesic flow (so3c Arch B)
    "so3c_interaction_set",  # + SO3CInteraction multi-particle ODE flow
    "so3c_covariant_set",    # flow with a COVARIANT connection (the fix)
    "so3c_message_set",      # + multi-round covariant message passing
)
# The so3c set family (subset of SET_MODELS, dispatched in _build_deepsets).
SO3C_SET_MODELS = (
    "so3c_invariant_set",
    "so3c_equivariant_set",
    "so3c_interaction_set",
    "so3c_covariant_set",
    "so3c_message_set",
)
ALL_MODELS = MATCHED_MODELS + NATURAL_MODELS + SET_MODELS

# Default number of parallel blocks for the "*_multi" variants.
MULTI_CHANNELS = 4


# ─────────────────────────────────────────────────────────────────────────
# Matched bottleneck
# ─────────────────────────────────────────────────────────────────────────

class BottleneckClassifier(nn.Module):
    """Generic Linear -> activation -> Linear pipeline through a bottleneck.

    Every matched baseline uses the same Linear(in -> hidden) -> activation
    -> Linear(hidden -> out) shape, so "matched bottleneck" comparisons are
    honest: every model sees the same compression before its activation runs.

    Parameters
    ----------
    in_features  : int        input dimension
    out_features : int        output dimension
    activation   : nn.Module  activation module operating on `hidden`-dim vectors
    hidden       : int        bottleneck dimension (default 6)
    dtype        : torch.dtype  parameter dtype (default float64)
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   nn.Module,
        hidden:       int = DIM,
        dtype:        torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.input_proj  = nn.Linear(in_features, hidden).to(dtype)
        self.activation  = activation
        self.output_proj = nn.Linear(hidden, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        h = self.input_proj(x)
        h = self.activation(h)
        return self.output_proj(h)

    def regularization_loss(self) -> torch.Tensor:
        """Forward to activation.regularization_loss if present, else 0.

        Lets training code call ``model.regularization_loss()`` uniformly
        regardless of whether the activation has a regulariser.
        """
        reg = getattr(self.activation, "regularization_loss", None)
        if callable(reg):
            return reg()
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


# ─────────────────────────────────────────────────────────────────────────
# Natural-width MLP
# ─────────────────────────────────────────────────────────────────────────

class NaturalWidthMLP(nn.Module):
    """Linear(in -> H) -> activation -> Linear(H -> out), no bottleneck.

    Used as the engineering-fairness baseline: MLPs use a wide hidden
    layer (default 256) while the geometric models stay at 6. Tests whether
    the geometric model is competitive *in practice* despite its compression.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        activation:   nn.Module,
        hidden:       int = 256,
        dtype:        torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.l1 = nn.Linear(in_features, hidden).to(dtype)
        self.activation = activation
        self.l2 = nn.Linear(hidden, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        return self.l2(self.activation(self.l1(x)))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


# ─────────────────────────────────────────────────────────────────────────
# Per-constituent Deep Sets (Top Tagging headline)
# ─────────────────────────────────────────────────────────────────────────

class DeepSetsClassifier(nn.Module):
    """Per-particle phi -> masked mean pool -> rho head.

    The generic per-constituent baseline. Each jet arrives as a
    (B, K, 5) tensor where ``[..., :4]`` is the standardised
    (E, px, py, pz) of each of K constituents and ``[..., 4]`` is a
    mask (1.0 real, 0.0 padding). The same per-particle function phi =
    activation(Linear(4 -> hidden)) is applied to every constituent, then a
    masked mean pool over constituents feeds the linear classifier head.

    Unlike the aggregated jet-level loader, this keeps the per-particle
    substructure, so a geometric prior has something to exploit. The
    ReLU/Tanh/GELU baselines share this exact skeleton with their
    activation swapped in, keeping the comparison apples-to-apples.
    """

    def __init__(
        self,
        activation:   nn.Module,
        out_features: int,
        in_features:  int = 4,
        hidden:       int = DIM,
        dtype:        torch.dtype = torch.float64,
        channels:     list[nn.Module] | None = None,
        pool:         str = "mean",
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.pool = pool
        # Single-channel (default) uses ``activation``; multi-channel passes a
        # list of parallel activations, each with its own embedding, whose
        # pooled outputs are concatenated before the head.
        if channels is None:
            self.embeds = nn.ModuleList([nn.Linear(in_features, hidden).to(dtype)])
            self.acts   = nn.ModuleList([activation])
            head_in = hidden
        else:
            self.embeds = nn.ModuleList(
                [nn.Linear(in_features, hidden).to(dtype) for _ in channels]
            )
            self.acts = nn.ModuleList(channels)
            head_in = hidden * len(channels)
        self.head = nn.Linear(head_in, out_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        feat = x[..., :4]                       # (B, K, 4)
        mask = x[..., 4:5]                       # (B, K, 1)
        B, K, _ = feat.shape

        pooled_channels = []
        for embed, act in zip(self.embeds, self.acts):
            h = embed(feat)                                   # (B, K, hidden)
            h = act(h.reshape(B * K, -1)).reshape(B, K, -1)
            h = h * mask                                      # zero out padding
            if self.pool == "sum":
                pooled = h.sum(dim=1)
            else:                                             # masked mean
                pooled = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled_channels.append(pooled)
        return self.head(torch.cat(pooled_channels, dim=-1))

    def regularization_loss(self) -> torch.Tensor:
        total = torch.zeros((), dtype=self.dtype)
        for act in self.acts:
            reg = getattr(act, "regularization_loss", None)
            if callable(reg):
                total = total + reg()
        return total


# ─────────────────────────────────────────────────────────────────────────
# The invariant-readout anchor
# ─────────────────────────────────────────────────────────────────────────

class EtaInvariantsClassifier(nn.Module):
    """SO(3,3)-INVARIANT-by-construction classifier.

    Per-particle features are reduced to the only quantities that survive
    an SO(3,3) transformation: the η-norm  m_i² = <p_i, η p_i>  of each
    constituent and the pairwise η-inner products  s_ij = <p_i, η p_j>.
    Those scalars are pooled into permutation-invariant statistics and
    fed to a plain MLP. The whole network commutes with any SO(3,3) boost
    BY CONSTRUCTION (no approximation), so the model is structurally
    immune to the boost-OOD failure mode that hits Linear-wrapped designs.

    No learned flow here on purpose: this is the maximally-conservative
    "use only the η metric, nothing else" baseline, and the anchor row of
    the canonical top-tagging table.
    """

    def __init__(
        self,
        out_features: int,
        hidden: int = 64,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("eta", ETA.to(dtype))

        # 7 pooled invariant statistics (see forward) -> small MLP -> logits.
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        p4   = x[..., :4]                                  # (B, K, 4)
        mask = x[..., 4]                                    # (B, K)
        p    = _lift_4to6(p4)                               # (B, K, 6)

        eta = self.eta                                      # (6,)
        # Per-particle η-norm  m_i^2 = <p_i, η p_i>
        m2 = (p * eta * p).sum(dim=-1)                      # (B, K)
        m2 = m2 * mask
        # Pairwise η-inner products  s_ij = <p_i, η p_j>
        s  = torch.einsum("bki,i,bli->bkl", p, eta, p)      # (B, K, K)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)   # (B, K, K)
        s = s * pair_mask
        eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
        s_off = s * (1.0 - eye)                             # off-diagonal pairs

        n_real      = mask.sum(dim=-1).clamp_min(1.0)
        n_real_pair = pair_mask.sum(dim=(-2, -1)).clamp_min(1.0)

        feats = torch.stack([
            m2.sum(dim=-1) / n_real,                        # mean per-particle mass²
            m2.pow(2).sum(dim=-1) / n_real,                 # mean (mass²)²
            m2.abs().sum(dim=-1) / n_real,                  # mean |mass²|
            s_off.sum(dim=(-2, -1)) / n_real_pair,          # mean pairwise inner
            s_off.pow(2).sum(dim=(-2, -1)) / n_real_pair,   # mean squared pairwise
            s_off.abs().max(dim=-1).values.max(dim=-1).values,    # max |pair|
            (p.sum(dim=1) * eta * p.sum(dim=1)).sum(dim=-1),       # jet-total <P,ηP>
        ], dim=-1)                                          # (B, 7)
        return self.mlp(feats)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.dtype, device=next(self.parameters()).device)


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────

def _build_deepsets(
    name: str,
    in_features: int,
    out_features: int,
    *,
    dtype: torch.dtype,
    natural_hidden: int,
    pool: str,
    so3c_kwargs: dict | None = None,
) -> nn.Module:
    """Construct the per-constituent Deep Sets variant for a model name."""
    if name == "eta_invariants":
        return EtaInvariantsClassifier(out_features=out_features, dtype=dtype)

    pointwise: dict[str, Callable[[], nn.Module]] = {
        "relu_bottleneck": nn.ReLU, "tanh_bottleneck": nn.Tanh,
        "gelu_bottleneck": nn.GELU,
        "relu_mlp": nn.ReLU, "tanh_mlp": nn.Tanh, "gelu_mlp": nn.GELU,
    }
    if name in pointwise:
        hidden = DIM if name.endswith("_bottleneck") else natural_hidden
        return DeepSetsClassifier(
            activation=pointwise[name](), out_features=out_features,
            in_features=in_features, hidden=hidden, dtype=dtype, pool=pool,
        )

    if name in SO3C_SET_MODELS:
        # Complexified-SO(3) set models on the bivector lift. They consume
        # the raw (B, K, 5) constituents directly (their lift is internal).
        from benchmarks.so3c_models import (
            SO3CCovariantSetClassifier,
            SO3CEquivariantSetClassifier,
            SO3CInteractionSetClassifier,
            SO3CInvariantSetClassifier,
            SO3CMessageSetClassifier,
        )
        # Capacity knobs (channels / hidden / act_hidden / T) come through
        # so3c_kwargs so a scaling study is a CLI matter, not a code edit.
        extra = dict(so3c_kwargs or {})
        if name != "so3c_message_set":
            # Message-passing-only knobs; harmless to drop for the others so
            # a sweep can pass one kwargs dict across the whole family.
            for k in ("rounds", "scalar_dim", "msg_dim", "channel_mixing",
                      "neighbors", "beams", "beam_energy", "dropout",
                      "mass_input", "self_edges", "relnorm_edge", "falpha",
                      "vector_channel", "pair_latent"):
                extra.pop(k, None)
        if name == "so3c_invariant_set":
            extra.pop("channels", None)      # no channel axis in the no-flow model
            extra.pop("act_hidden", None)
            extra.pop("T", None)
            return SO3CInvariantSetClassifier(out_features=out_features,
                                              dtype=dtype, **extra)
        if name == "so3c_equivariant_set":
            return SO3CEquivariantSetClassifier(out_features=out_features,
                                                dtype=dtype, **extra)
        if name == "so3c_covariant_set":
            return SO3CCovariantSetClassifier(out_features=out_features,
                                              dtype=dtype, **extra)
        if name == "so3c_message_set":
            return SO3CMessageSetClassifier(out_features=out_features,
                                            dtype=dtype, **extra)
        extra.pop("channels", None)          # interaction model is single-channel
        extra.pop("act_hidden", None)
        return SO3CInteractionSetClassifier(out_features=out_features,
                                            dtype=dtype, **extra)

    raise ValueError(f"Unknown model name: {name!r}. Known: {ALL_MODELS}")


def build_model(
    name: str,
    in_features: int,
    out_features: int,
    *,
    natural_hidden: int = 256,
    dtype: torch.dtype = torch.float64,
    representation: str = "flat",
    pool: str = "mean",
    so3c_kwargs: dict | None = None,
) -> nn.Module:
    """Construct a model by string name.

    Matched-bottleneck names use hidden=6 throughout. Natural-width
    names use ``natural_hidden`` (default 256).

    Parameters
    ----------
    name        : one of ALL_MODELS.
    in_features : input dimensionality. For ``representation="constituents"``
                  this is the per-particle feature count (4 = E,px,py,pz).
    out_features: number of classes (or regression outputs).
    natural_hidden : hidden size for natural-width MLPs.
    dtype       : parameter dtype.
    representation : "flat" (default; input is (B, in_features)) or
                  "constituents" (input is (B, K, 5) — a Deep Sets model
                  applies the activation per particle then masked-pools).
    pool        : "mean" (default) or "sum" pooling over constituents.
    so3c_kwargs : forwarded to the so3c model constructors (see
                  benchmarks.so3c_models); filtered per model name.

    Returns
    -------
    nn.Module with ``forward(x)`` and ``regularization_loss()``.
    """
    if representation == "constituents":
        if name in SO3C_MODELS:
            raise ValueError(
                f"{name!r} supports representation='flat' only; the so3c "
                f"Deep-Sets path is not implemented yet."
            )
        return _build_deepsets(
            name, in_features, out_features,
            dtype=dtype, natural_hidden=natural_hidden,
            pool=pool, so3c_kwargs=so3c_kwargs,
        )

    # so3c variants: the closed-form exact flow is a bounded group element
    # for any input, so the flat path needs neither input bounding nor a
    # norm cap. T is degenerate with the learnable connection norm; the
    # activation's native T=1.0 gives the full soft-normalised range.
    if name in ("so3c", "so3c_static"):
        from benchmarks.so3c_models import SO3CBottleneck
        return SO3CBottleneck(
            in_features=in_features, out_features=out_features,
            mode="dynamic" if name == "so3c" else "static", dtype=dtype,
            **(so3c_kwargs or {}),
        )
    if name == "so3c_multi":
        from benchmarks.so3c_models import MultiChannelSO3C
        return MultiChannelSO3C(
            in_features=in_features, out_features=out_features,
            channels=MULTI_CHANNELS, dtype=dtype,
        )

    pointwise: dict[str, Callable[[], nn.Module]] = {
        "relu_bottleneck": nn.ReLU,
        "tanh_bottleneck": nn.Tanh,
        "gelu_bottleneck": nn.GELU,
        "relu_mlp":        nn.ReLU,
        "tanh_mlp":        nn.Tanh,
        "gelu_mlp":        nn.GELU,
    }
    if name in pointwise:
        act = pointwise[name]()
        if name.endswith("_bottleneck"):
            return BottleneckClassifier(
                in_features=in_features, out_features=out_features,
                activation=act, hidden=DIM, dtype=dtype,
            )
        return NaturalWidthMLP(
            in_features=in_features, out_features=out_features,
            activation=act, hidden=natural_hidden, dtype=dtype,
        )

    raise ValueError(
        f"Unknown model name: {name!r}. "
        f"Known: {ALL_MODELS}"
    )


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
