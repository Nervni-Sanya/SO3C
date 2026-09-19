"""Fixed so3c_message_set cases for the golden-output regression test.

``tests/data/so3c_message_golden.npz`` holds the logits and parameter
gradients these cases produced with the implementation every published
number was measured with. Any rewrite of the forward pass for speed must
reproduce them in float64; see test_message_set_matches_golden.

Every zero-initialised head is excited, so the flow, the vector update and
the pair latent all act: with the heads at zero the flow is the identity
and a rewrite of it would be tested vacuously.

Every parameter and input is drawn from numpy's legacy RandomState, whose
stream is guaranteed stable across versions and platforms, straight into
float64. Letting torch initialise the layers made the cases differ at the
float32 ulp level (~4e-8) between torch 2.11 on Windows and torch 2.10 on
Kaggle, because nn.Linear initialises in float32 before .to(float64); the
on-card guard then failed on the platform, not on the code
(benchmarks/diag_cuda_golden.py).
"""

from __future__ import annotations

import numpy as np
import torch

from benchmarks.so3c_models import SO3CMessageSetClassifier

CONFIGS = {
    "base": dict(channels=8, rounds=3, beams=True),
    "nobeams": dict(channels=4, rounds=3),
    "fixes": dict(channels=8, rounds=3, beams=True, mass_input=False,
                  self_edges=False),
    "vector": dict(channels=8, rounds=3, beams=True, vector_channel=True),
    "pair8": dict(channels=8, rounds=3, beams=True, pair_latent=8),
    "relnorm": dict(channels=8, rounds=3, beams=True, relnorm_edge=True),
    "falpha3": dict(channels=8, rounds=3, beams=True, falpha=3),
    "combo": dict(channels=8, rounds=3, beams=True, vector_channel=True,
                  pair_latent=8, mass_input=False, self_edges=False,
                  relnorm_edge=True, falpha=2),
    "knn": dict(channels=4, rounds=2, neighbors=4),
    "noscalar": dict(channels=4, rounds=2, scalar_dim=0),
    "rounds1": dict(channels=4, rounds=1, beams=True),
    "nomix": dict(channels=4, rounds=2, beams=True, channel_mixing=False),
    "vector_nobeams": dict(channels=4, rounds=2, vector_channel=True),
    "pair_noscalar": dict(channels=4, rounds=2, pair_latent=4, scalar_dim=0),
    "combo_nobeams": dict(channels=4, rounds=3, vector_channel=True,
                          pair_latent=4, relnorm_edge=True, falpha=2,
                          self_edges=False, mass_input=False),
    "wide": dict(channels=3, rounds=2, beams=True, act_hidden=7, msg_dim=5,
                 scalar_dim=6, hidden=11, T=0.7),
}


def _stable_parameters(model, rs: np.random.RandomState) -> None:
    """Overwrite every parameter with platform-stable float64 values at
    initialisation-like scales; zero-initialised heads get non-zero values."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            n = rs.standard_normal(tuple(p.shape))
            if name == "channel_weights":                    # (C, 2)
                n[:, 0] = 1.0 + 0.1 * n[:, 0]
                n[:, 1] = 0.1 * n[:, 1]
            elif name == "mix":                              # (R, C, C, 2)
                n = 0.1 * n
                n[..., 0] += np.eye(p.shape[1])
            elif name.startswith("falpha_embed"):            # exponents
                n = 0.1 + 0.3 * np.abs(n)
            elif p.dim() >= 2:
                n = n / np.sqrt(p.shape[-1])
                if name.startswith(("w_head", "v_head")):
                    n = n * 1.6                              # excite the flow
            else:
                n = 0.1 * n
            p.copy_(torch.from_numpy(np.ascontiguousarray(n)).to(p.dtype))


def build_case(name: str, B: int = 6, K: int = 10):
    """(model, x, y) for one configuration, float64, fully deterministic and
    identical on every platform."""
    model = SO3CMessageSetClassifier(dtype=torch.float64, **CONFIGS[name])
    rs = np.random.RandomState(1234)
    _stable_parameters(model, rs)
    n = np.array([K, K - 3, 4, K - 1, 7, K][:B])
    mask = (np.arange(K)[None, :] < n[:, None]).astype(np.float64)
    pv = rs.standard_normal((B, K, 3)) * 0.4 + np.array([2.0, 0.5, 1.0])
    E = np.sqrt((pv * pv).sum(-1, keepdims=True))
    x = np.concatenate([np.concatenate([E, pv], -1) * mask[..., None],
                        mask[..., None]], -1)
    y = torch.tensor([0, 1, 1, 0, 1, 0][:B])
    return model, torch.from_numpy(x), y


def gradient_fingerprint(grad: torch.Tensor, k: int = 32) -> torch.Tensor:
    """(n,) float64 gradient -> (k,) fixed random projections.

    Keeps the fixture small enough to ship inside a notebook, while any
    change to the gradient moves the projections: they are k independent
    Gaussian directions, drawn from a fixed seed for each gradient length.
    """
    g = torch.Generator().manual_seed(7 + grad.numel())
    R = torch.randn(grad.numel(), k, dtype=torch.float64, generator=g)
    return grad.to(torch.float64) @ R


def load_reference_module():
    """The pre-rewrite implementation (16302b1), shipped as
    tests/data/ref_so3c_models_16302b1.py, for same-device comparisons."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parent / "data" / "ref_so3c_models_16302b1.py"
    spec = importlib.util.spec_from_file_location("ref_so3c_models_16302b1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

