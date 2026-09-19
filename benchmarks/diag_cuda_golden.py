"""
benchmarks.diag_cuda_golden
---------------------------
Why does so3c_message_set in float64 on CUDA differ from the CPU golden
fixture by ~4e-8 when the CPU matches it to 1e-14?

The golden cases are well conditioned (a 1e-15 relative perturbation of the
inputs or weights moves the logits by ~1e-14), so a 4e-8 difference is not
amplified rounding: something is computed differently on the card. This
script runs inside one Kaggle container and separates the possibilities:

  1. platform: the fixture against this container's CPU (other torch
     version, other BLAS);
  2. device: this container's CPU against its GPU, per golden case;
  3. old vs new: the pre-rewrite implementation (16302b1, shipped as
     tests/data/ref_so3c_models_16302b1.py) against the new one, on the
     same device -- if they agree on CUDA, the rewrite is not the cause;
  4. ops: float64 CPU-vs-CUDA differences of the primitives the model uses;
  5. modules: CPU-vs-CUDA outputs of every submodule for one case, in call
     order, to find where the difference first appears.

Run:
    python -m benchmarks.diag_cuda_golden
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import torch

from so3c.algebra import _rodrigues_coefficients, expm_so3c, expm_so3c_apply
from tests.message_golden import CONFIGS, build_case, gradient_fingerprint

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_reference():
    path = ROOT / "tests" / "data" / "ref_so3c_models_16302b1.py"
    spec = importlib.util.spec_from_file_location("ref_so3c_models_16302b1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(name, device, cls=None):
    model, x, y = build_case(name)
    if cls is not None:                      # same weights, other implementation
        ref = cls(dtype=torch.float64, **CONFIGS[name])
        ref.load_state_dict(model.state_dict())
        model = ref
    model, x, y = model.to(device), x.to(device), y.to(device)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    g = gradient_fingerprint(torch.cat([q.reshape(-1) for q in grads]).cpu())
    return logits.detach().cpu().double(), g


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-300)).item()


def main() -> int:
    has_cuda = torch.cuda.is_available()
    print("torch", torch.__version__, "| cuda", has_cuda,
          torch.cuda.get_device_name(0) if has_cuda else "")
    golden = np.load(ROOT / "tests" / "data" / "so3c_message_golden.npz")
    ref_mod = load_reference()
    devices = ["cpu"] + (["cuda"] if has_cuda else [])

    print("\n1-3. per case, relative max error (logits / gradient fingerprint)")
    print("%-15s %-21s %-21s %-21s %-21s %-21s" % ("case", "old cpu vs fixture", "new cpu vs fixture",
                                                   "new cuda vs fixture", "new vs old, cuda",
                                                   "cuda vs cpu, old"))
    for name in CONFIGS:
        res = {}
        for dev in devices:
            res[("new", dev)] = run_case(name, dev)
            res[("old", dev)] = run_case(name, dev, ref_mod.SO3CMessageSetClassifier)
        fl, fg = golden[name + "/logits"], golden[name + "/grad"]
        cols = ["%.1e / %.1e" % (rel(res[("old", "cpu")][0], fl), rel(res[("old", "cpu")][1], fg)),
                "%.1e / %.1e" % (rel(res[("new", "cpu")][0], fl), rel(res[("new", "cpu")][1], fg))]
        if has_cuda:
            cols += ["%.1e / %.1e" % (rel(res[("new", "cuda")][0], fl), rel(res[("new", "cuda")][1], fg)),
                     "%.1e / %.1e" % (rel(res[("new", "cuda")][0], res[("old", "cuda")][0]),
                                      rel(res[("new", "cuda")][1], res[("old", "cuda")][1])),
                     "%.1e / %.1e" % (rel(res[("old", "cuda")][0], res[("old", "cpu")][0]),
                                      rel(res[("old", "cuda")][1], res[("old", "cpu")][1]))]
        print("%-15s " % name + " ".join("%-21s" % c for c in cols), flush=True)

    if not has_cuda:
        return 0

    print("\n4. primitives, float64, relative max CPU-vs-CUDA difference")
    g = torch.Generator().manual_seed(0)
    n = 200000
    scale = torch.logspace(-6, 0.5, n, dtype=torch.float64)[:, None]
    a = torch.complex(torch.randn(n, 3, dtype=torch.float64, generator=g),
                      torch.randn(n, 3, dtype=torch.float64, generator=g)) * scale
    z = torch.complex(torch.randn(n, 3, dtype=torch.float64, generator=g),
                      torch.randn(n, 3, dtype=torch.float64, generator=g))
    xr = torch.randn(n, dtype=torch.float64, generator=g) * 10
    tests = {
        "asinh (real)": lambda d: torch.asinh(xr.to(d)),
        "sqrt (complex)": lambda d: torch.sqrt((a[:, 0] * a[:, 1]).to(d)),
        "sin (complex)": lambda d: torch.sin(a[:, 0].to(d)),
        "cos (complex)": lambda d: torch.cos(a[:, 0].to(d)),
        "rodrigues c1": lambda d: _rodrigues_coefficients(a.to(d))[0],
        "rodrigues c2": lambda d: _rodrigues_coefficients(a.to(d))[1],
        "cross (complex)": lambda d: torch.linalg.cross(a.to(d), z.to(d), dim=-1),
        "expm_so3c_apply": lambda d: expm_so3c_apply(a.to(d), z.to(d), t=-1.0),
        "expm_so3c @ z": lambda d: (expm_so3c(a.to(d), t=-1.0) @ z.to(d).unsqueeze(-1)).squeeze(-1),
        "matmul (real)": lambda d: xr[:600].reshape(20, 30).to(d) @ xr[600:1200].reshape(30, 20).to(d),
    }
    for nm, f in tests.items():
        print("  %-18s %.1e" % (nm, rel(f("cuda").cpu(), f("cpu"))))

    print("\n5. submodule outputs for case 'base', CPU vs CUDA, in call order")
    records = {}
    for dev in ("cpu", "cuda"):
        model, x, y = build_case("base")
        model, x = model.to(dev), x.to(dev)
        seen = []

        def hook(mod, inp, out, name=None):
            if isinstance(out, torch.Tensor):
                seen.append((name, out.detach().cpu()))
        for name, m in model.named_modules():
            if name:
                m.register_forward_hook(lambda mod, i, o, name=name: hook(mod, i, o, name))
        with torch.no_grad():
            model(x)
        records[dev] = seen
    for (n1, t1), (n2, t2) in zip(records["cpu"], records["cuda"]):
        print("  %-22s %.1e" % (n1, rel(t2, t1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
