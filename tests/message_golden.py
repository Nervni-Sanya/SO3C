"""Fixed so3c_message_set cases for the golden-output regression test.

``tests/data/so3c_message_golden.npz`` holds the logits and parameter
gradients these cases produced with the implementation every published
number was measured with. Any rewrite of the forward pass for speed must
reproduce them in float64; see test_message_set_matches_golden.

Every zero-initialised head is excited, so the flow, the vector update and
the pair latent all act: with the heads at zero the flow is the identity
and a rewrite of it would be tested vacuously.
"""

from __future__ import annotations

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
}


def build_case(name: str, B: int = 6, K: int = 10):
    """(model, x, y) for one configuration, float64, fully deterministic."""
    torch.manual_seed(1234)
    model = SO3CMessageSetClassifier(dtype=torch.float64, **CONFIGS[name])
    g = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for head in list(model.w_head) + list(model.v_head):
            head.weight.copy_(0.4 * torch.randn(head.weight.shape,
                                                dtype=torch.float64, generator=g))
            head.bias.copy_(0.1 * torch.randn(head.bias.shape,
                                              dtype=torch.float64, generator=g))
    n = torch.tensor([K, K - 3, 4, K - 1, 7, K][:B])
    mask = (torch.arange(K)[None, :] < n[:, None]).to(torch.float64)
    pv = (torch.randn(B, K, 3, dtype=torch.float64, generator=g) * 0.4
          + torch.tensor([2.0, 0.5, 1.0], dtype=torch.float64))
    E = pv.norm(dim=-1, keepdim=True)
    x = torch.cat([torch.cat([E, pv], -1) * mask[..., None], mask[..., None]], -1)
    y = torch.tensor([0, 1, 1, 0, 1, 0][:B])
    return model, x, y


def gradient_fingerprint(grad: torch.Tensor, k: int = 32) -> torch.Tensor:
    """(n,) float64 gradient -> (k,) fixed random projections.

    Keeps the fixture small enough to ship inside a notebook, while any
    change to the gradient moves the projections: they are k independent
    Gaussian directions, drawn from a fixed seed for each gradient length.
    """
    g = torch.Generator().manual_seed(7 + grad.numel())
    R = torch.randn(grad.numel(), k, dtype=torch.float64, generator=g)
    return grad.to(torch.float64) @ R

