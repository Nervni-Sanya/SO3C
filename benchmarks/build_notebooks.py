"""
benchmarks.build_notebooks
--------------------------
Generate the Kaggle notebooks from source, with the repo's code embedded.

Two reasons the notebooks are generated rather than hand-written:

1. The Kasieczka top-tagging set is not on Kaggle and the GitHub branch
   lags local work, so each notebook carries so3c/, benchmarks/ and tests/
   as an embedded tar.gz. Regenerating keeps that copy in step with the
   working tree instead of silently going stale.
2. Notebook ``source`` must be a list of lines that KEEP their trailing
   newline. Writing it without them produced cells whose lines Kaggle
   concatenated into one ("%cd /kaggle/working/so33!pip install ..."),
   which is how the first data-prep kernel died. ``_src`` enforces it.

Run:
    python -m benchmarks.build_notebooks --list
    python -m benchmarks.build_notebooks --only k64_headline120_seed0
    python -m benchmarks.build_notebooks --all     # rewrites every notebook

With no arguments (and with --help) nothing is written.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
# tests/ ships too: the finish kernel runs the CUDA resume test on the card
# before its real runs, and without the directory pytest found nothing
# to run and the guard stopped the session.
CODE_PACKAGES = ("so3c", "benchmarks", "tests")


def _src(text: str) -> list[str]:
    """Notebook source: lines that keep their trailing newline."""
    return text.strip("\n").splitlines(keepends=True)


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": _src(text),
            "execution_count": None, "outputs": [], "id": None}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text),
            "id": None}


def embed_code() -> str:
    """base64 tar.gz of the repo's python sources (no caches, no data)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for pkg in CODE_PACKAGES:
            for f in sorted((ROOT / pkg).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                tar.add(f, arcname=str(f.relative_to(ROOT)))
        # small test fixtures (the golden outputs the on-card guard compares
        # against); without them the shipped test fails on a missing file
        for f in sorted((ROOT / "tests" / "data").glob("*.npz")):
            tar.add(f, arcname=str(f.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def _finalise(cells: list[dict], path: pathlib.Path) -> None:
    for i, c in enumerate(cells):
        c["id"] = "cell%02d" % i          # nbformat 4.5 requires cell ids
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  %s: %.0f KB, %d cells" % (path.name, path.stat().st_size / 1024,
                                       len(cells)))


UNPACK = '''
import base64, io, tarfile, pathlib
CODE_B64 = "{blob}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz").extractall("/kaggle/working/repo")
print(sorted(p.name for p in pathlib.Path("/kaggle/working/repo").iterdir()))
'''

# Runs a benchmark command inside the unpacked repo. Using subprocess rather
# than a shell "!" line keeps multi-line commands out of Kaggle's line-joining
# path entirely, and surfaces the child's stderr when something fails.
RUNNER = '''
import subprocess, sys

def run(args, cwd="/kaggle/working/repo"):
    """Run a repo module, stream nothing, return (ok, tail-of-output)."""
    r = subprocess.run([sys.executable, "-m"] + args, cwd=cwd,
                       capture_output=True, text=True)
    out = r.stdout[-4000:]
    if r.returncode != 0:
        out += "\\n--- stderr ---\\n" + r.stderr[-3000:]
    print(out)
    return r.returncode == 0
'''


# Kaggle hands out a Tesla P100 (sm_60) for API-pushed kernels and will not
# reliably give a T4 however the accelerator is requested. The preinstalled
# torch is built for sm_70+, so every CUDA op on a P100 dies with
# "no kernel image is available for execution on the device" even though
# torch.cuda.is_available() returns True. Installing a build whose arch list
# covers sm_60 fixes it in ~150 s. P100 is a fine card for us once it works:
# fp64 runs at 3.9 TFLOP/s against 6.5 for fp32, so double precision stays
# usable as a fallback (on a T4 it would be 32x slower).
GPU_SETUP = """
import subprocess, sys, time

def ensure_torch_for_this_gpu():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA; nothing to fix")
        return
    cap = torch.cuda.get_device_capability(0)
    tag = "sm_%d%d" % cap
    if tag in torch.cuda.get_arch_list():
        print("torch", torch.__version__, "already supports", tag)
        return
    print("torch", torch.__version__, "lacks", tag,
          "- installing a compatible build")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "--index-url",
                        "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
    print("pip rc", r.returncode, "in %.0f s" % (time.perf_counter() - t0))
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit("could not install a GPU-compatible torch")

ensure_torch_for_this_gpu()
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "torchdiffeq"],
               check=True)
"""


def build_dataprep(blob: str, k: int = 32) -> None:
    """Data-prep kernel for a given number of retained constituents.

    K is the biggest lever we have: the K-sweep gave AUC 0.852 / 0.935 /
    0.964 / 0.970 at K = 4 / 8 / 16 / 32 and had not saturated, while every
    published model at 0.987 uses the full jet. A K=128 npz also serves
    smaller K for free -- select_leading_constituents re-sorts and truncates
    at load time, so --n-constituents 64 reads the same file.
    """
    cells = [
        md("""
# SO3C data prep - Top Tagging Reference to K=%d npz

CPU kernel: spends no GPU quota. Downloads the Kasieczka set from the
HuggingFace mirror `dl4phys/top_tagging` - the same source the local CPU
results came from - and converts it with the repo's own
`download_top_tagging.py --n-constituents %d`.

Sizes at K=%d: train %.1f GB raw before compression. Kaggle's working dir
holds 20 GB, so this fits; the loader can still read any smaller K from it.
""" % (k, k, k, 1211000 * k * 4 * 4 / 2 ** 30)),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "torchdiffeq", "huggingface_hub"], check=True)
import torch, numpy, pyarrow
print("torch", torch.__version__, "numpy", numpy.__version__,
      "pyarrow", pyarrow.__version__)
K_KEEP = %d
print("K_KEEP =", K_KEEP)
""" % k),
        code("""
from huggingface_hub import hf_hub_download
import pathlib
src = pathlib.Path("/kaggle/working/parquet")
src.mkdir(parents=True, exist_ok=True)
for fn in ("train.parquet", "validation.parquet", "test.parquet"):
    p = hf_hub_download(repo_id="dl4phys/top_tagging", filename=fn,
                        repo_type="dataset", local_dir=str(src))
    print(fn, "%.0f MB" % (pathlib.Path(p).stat().st_size / 2 ** 20))
"""),
        code("""
ok = run(["benchmarks.download_top_tagging",
          "--cache-dir", "/kaggle/working/data",
          "--source-dir", "/kaggle/working/parquet",
          "--skip-download", "--n-constituents", str(K_KEEP)])
assert ok, "conversion failed"
"""),
        code("""
import numpy as np, shutil
EXPECTED = {"train": 1211000, "val": 403000, "test": 404000}
for split, n_exp in EXPECTED.items():
    d = np.load("/kaggle/working/data/top_tagging_%s.npz" % split)
    c, y = d["constituents"], d["labels"]
    pt = np.sqrt(c[:, :, 1] ** 2 + c[:, :, 2] ** 2)
    m2 = c[:, :, 0] ** 2 - (c[:, :, 1:] ** 2).sum(-1)
    monotone = bool((np.diff(pt, axis=1) <= 1e-3).all())
    nz = (np.abs(c).sum(-1) > 0).sum(1)
    print(split, c.shape, "signal=%.3f" % y.mean(),
          "pT-ordered=%s" % monotone, "max|m2|=%.2e" % np.abs(m2).max(),
          "| real constituents: mean %.1f max %d" % (nz.mean(), nz.max()))
    assert c.shape == (n_exp, K_KEEP, 4), "%s: wrong shape" % split
    assert monotone and abs(y.mean() - 0.5) < 0.01
shutil.rmtree("/kaggle/working/parquet", ignore_errors=True)
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
print("OK - npz verified")
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_dataprep_k%d.ipynb" % k))


def build_validate(blob: str) -> None:
    """Cheap port validation before any canonical GPU hours are spent.

    Runs the internal protocol (70k jets, 30 epochs), whose CPU reference is
    known with error bars from three seeds:

        so3c_equivariant_set  AUC 0.9710 +- 0.0011   rej@0.3 189 +- 49
        so3c_invariant_set    AUC 0.9626 +- 0.0005   rej@0.3 129 +- 13
        eta_invariants        AUC 0.9447 +- 0.0004   rej@0.3  46 +-  5

    Minutes on a GPU instead of the ~1.5 h a canonical validation run would
    cost, and it checks against a spread rather than a single number. Also
    exercises checkpoint/resume on CUDA, which has only ever been tested on
    CPU.
    """
    cells = [
        md("""
# SO3C GPU port validation (internal protocol)

Blocking gate. GPU float32 must land inside the CPU float64 reference band
before any canonical scaling run is worth its GPU-hours.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
"""),
        code(GPU_SETUP),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import glob, pathlib
# The data-prep kernel wrote data/top_tagging_*.npz, so the mounted path is
# one level deeper than /kaggle/input/<source>/ -- search recursively and
# show what is actually mounted when nothing matches.
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
if not cands:
    for root in sorted(glob.glob("/kaggle/input/*")):
        print("mounted:", root)
        for sub in sorted(glob.glob(root + "/**/*", recursive=True))[:20]:
            print("   ", sub)
    raise AssertionError("top_tagging_train.npz not found under /kaggle/input")
DATA = str(pathlib.Path(cands[0]).parent)
print("data:", DATA)
REF = {"so3c_equivariant_set": (0.9710, 0.0011),
       "so3c_invariant_set":   (0.9626, 0.0005),
       "eta_invariants":       (0.9447, 0.0004)}
"""),
        code("""
# Run the same protocol twice, in float64 and in float32. The CPU reference
# is float64, so comparing a float32 GPU run against it confounds two
# changes at once: hardware (different reduction order) and precision. The
# float64 GPU run isolates the first; the float32-vs-float64 gap on the
# same card measures the second.
for dt in ("float64", "float32"):
    ok = run(["benchmarks.run_top_tagging",
              "--cache-dir", DATA, "--representation", "constituents",
              "--max-samples", "100000", "--epochs", "30",
              "--normalize", "global", "--seed", "0",
              "--device", "cuda", "--dtype", dt, "--batch-size", "512",
              "--models", ",".join(REF),
              "--results-dir", "/kaggle/working/results_" + dt,
              "--ckpt-dir", "/kaggle/working/checkpoints/" + dt, "--resume",
              "--max-seconds", "36000"])
    assert ok, dt + " run failed"
"""),
        code("""
import json

def load(dt, m):
    p = "/kaggle/working/results_%s/top_tagging_constituents__%s__seed0.json"
    return json.load(open(p % (dt, m)))["test_metrics"]

print("%-24s%10s%10s%10s%12s" % ("model", "GPU f64", "GPU f32",
                                 "CPU f64", "f64-CPU"))
worst_hw = 0.0
worst_prec = 0.0
for m, (ref, sd) in REF.items():
    a64 = load("float64", m)["test_auc"]
    a32 = load("float32", m)["test_auc"]
    worst_hw = max(worst_hw, abs(a64 - ref))
    worst_prec = max(worst_prec, abs(a32 - a64))
    print("%-24s%10.4f%10.4f%10.4f%+12.4f" % (m, a64, a32, ref, a64 - ref))

print()
print("hardware/reduction-order effect (GPU f64 vs CPU f64): %.4f AUC"
      % worst_hw)
print("precision effect (f32 vs f64 on the same card):        %.4f AUC"
      % worst_prec)
# 0.0027 was measured between CPU float64 and GPU float64 on this very
# protocol, and it moves eta_invariants too -- a model with no so3c code in
# it -- so it is BLAS reduction order steering the optimiser, not a port
# bug. The CPU seed spread (0.0004-0.0011) is a within-platform number and
# is the wrong yardstick for a cross-platform comparison; 0.005 is set from
# what was actually measured, with margin.
assert worst_hw < 0.005, "GPU float64 does not reproduce the CPU reference"

# What the scaling study depends on is that the models keep their order and
# their spacing, since every point will be compared against the others on
# this same card. A uniform offset cancels; a change in the gaps would not.
gaps_cpu = {"eq-inv": REF["so3c_equivariant_set"][0] - REF["so3c_invariant_set"][0],
            "inv-eta": REF["so3c_invariant_set"][0] - REF["eta_invariants"][0]}
g32 = {"eq-inv": load("float32", "so3c_equivariant_set")["test_auc"]
                 - load("float32", "so3c_invariant_set")["test_auc"],
       "inv-eta": load("float32", "so3c_invariant_set")["test_auc"]
                  - load("float32", "eta_invariants")["test_auc"]}
for k in gaps_cpu:
    print("gap %-8s CPU %+.4f  GPU %+.4f  diff %+.4f"
          % (k, gaps_cpu[k], g32[k], g32[k] - gaps_cpu[k]))
    assert g32[k] > 0, "model ordering changed on GPU"
    assert abs(g32[k] - gaps_cpu[k]) < 0.005, "model spacing changed on GPU"
print()
print("PORT VALIDATED - the float32 offset is precision, not a port bug"
      if worst_prec < 0.005 else
      "WARNING: float32 costs more than 0.005 AUC; run the campaign in f64")
"""),
        code("""
# Checkpoint/resume on CUDA: interrupt, resume, compare per-epoch history.
import json, shutil, pathlib
shutil.rmtree("/kaggle/working/rc", ignore_errors=True)
base = ["benchmarks.run_top_tagging", "--cache-dir", DATA,
        "--representation", "constituents", "--max-samples", "20000",
        "--epochs", "6", "--normalize", "global", "--seed", "0",
        "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
        "--models", "so3c_invariant_set"]
run(base + ["--results-dir", "/kaggle/working/rc/cont"])
run(base + ["--results-dir", "/kaggle/working/rc/int",
            "--ckpt-dir", "/kaggle/working/rc/ck", "--max-seconds", "20"])
for f in pathlib.Path("/kaggle/working/rc/int").glob("*.json"):
    f.unlink()
run(base + ["--results-dir", "/kaggle/working/rc/int",
            "--ckpt-dir", "/kaggle/working/rc/ck", "--resume"])
name = "top_tagging_constituents__so3c_invariant_set__seed0.json"
c = json.load(open("/kaggle/working/rc/cont/" + name))
i = json.load(open("/kaggle/working/rc/int/" + name))
same = ([h["val_acc"] for h in c["history"]] ==
        [h["val_acc"] for h in i["history"]])
print("resume reproduces continuous training on CUDA:", same)
print("AUC %.10f vs %.10f" % (c["test_metrics"]["test_auc"],
                              i["test_metrics"]["test_auc"]))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_validate.ipynb")


def build_scaling(blob: str) -> None:
    cells = [
        md("""
# SO3C scaling study

Accelerator: **GPU T4 x2** (one GPU is used - the harness has no
DataParallel). Add the data-prep kernel's output as a data source.

The validation cell is blocking: GPU float32 must reproduce the CPU
float64 canonical numbers (AUC 0.9744, rejection 320) to 1e-3 before any
scaling run is worth its GPU-hours.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi",
                      "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
"""),
        code(GPU_SETUP),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import glob, pathlib
# The data-prep kernel wrote data/top_tagging_*.npz, so the mounted path is
# one level deeper than /kaggle/input/<source>/ -- search recursively and
# show what is actually mounted when nothing matches.
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
if not cands:
    for root in sorted(glob.glob("/kaggle/input/*")):
        print("mounted:", root)
        for sub in sorted(glob.glob(root + "/**/*", recursive=True))[:20]:
            print("   ", sub)
    raise AssertionError("top_tagging_train.npz not found under /kaggle/input")
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_scaling"
CKPT = "/kaggle/working/checkpoints"
BASE = ["benchmarks.run_top_tagging", "--cache-dir", DATA,
        "--representation", "constituents", "--canonical-splits",
        "--epochs", "30", "--normalize", "global", "--seed", "0",
        "--device", "cuda", "--dtype", "float32",   # 0.0002 AUC vs float64
        "--models", "so3c_equivariant_set", "--resume",
        "--max-seconds", "39000"]
print("data:", DATA)
"""),
        code("""
ok = run(BASE + ["--batch-size", "512",
                 "--results-dir", "/kaggle/working/results_validate",
                 "--ckpt-dir", CKPT + "/validate"])
assert ok, "validation run failed"

import json
r = json.load(open("/kaggle/working/results_validate/"
                   "top_tagging_canonical__so3c_equivariant_set__seed0.json"))
auc = r["test_metrics"]["test_auc"]
rej = r["test_metrics"]["bg_rej_30"]
print("GPU float32: AUC %.4f rej %.0f   (CPU float64: 0.9744 / 320)"
      % (auc, rej))
# Tolerance is cross-platform, not the CPU seed spread: GPU training sits
# ~0.003 AUC below CPU for reduction-order reasons that affect the non-so3c
# baseline equally (measured in the validation kernel). Every scaling point
# below is compared against the others on this same card, where the offset
# cancels.
assert abs(auc - 0.9744) < 0.006, "AUC %.4f is far from the CPU reference" % auc
print("sanity check passed - scaling runs are safe to start")
"""),
        md("""
## Channel axis (geometry)
"""),
        code("""
for C in [4, 8, 16, 32, 48]:
    bs = "512" if C <= 16 else "256"      # (B,K,K) per channel: bound memory
    print("=== channels=%d ===" % C)
    run(BASE + ["--batch-size", bs,
                "--channels", str(C), "--hidden", "128", "--act-hidden", "32",
                "--results-dir", "%s/channels_%d" % (OUT, C),
                "--ckpt-dir", "%s/channels_%d" % (CKPT, C)])
"""),
        md("""
## Width axis (generic capacity - control)
"""),
        code("""
for H in [64, 128, 256, 512]:
    print("=== hidden=%d ===" % H)
    run(BASE + ["--batch-size", "512",
                "--channels", "4", "--hidden", str(H), "--act-hidden", "32",
                "--results-dir", "%s/width_%d" % (OUT, H),
                "--ckpt-dir", "%s/width_%d" % (CKPT, H)])
"""),
        code("""
import json, glob, pathlib, shutil
rows = []
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    rows.append((pathlib.Path(f).parent.name, r["n_params"],
                 r["test_metrics"]["test_auc"],
                 r["test_metrics"]["bg_rej_30"],
                 r["walltime_sec"] / 3600))
rows.sort(key=lambda x: x[1])
print("%-14s%9s%9s%10s%8s" % ("config", "params", "AUC", "rej@0.3", "hours"))
for c, p, a, j, h in rows:
    print("%-14s%9d%9.4f%10.0f%8.2f" % (c, p, a, j, h))
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
shutil.make_archive("/kaggle/working/results_scaling", "zip", OUT)
print("done")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_scaling.ipynb")


def build_fixed(blob: str) -> None:
    """Canonical runs for the two genuinely equivariant flow models."""
    cells = [
        md("""
# Canonical: the two genuinely equivariant flow models

`so3c_equivariant_set` turned out not to be Lorentz-invariant once trained:
its connection is a function of invariants, so it does not rotate with the
data. Under a rapidity-2 boost a trained model loses 0.095 AUC.

Two models keep the symmetry with a live flow, and neither has been run on
canonical data:

* `so3c_covariant_set` - connection a_a = z_a x sum_b phi(inv) z_b. The
  cross product is covariant for SO(3,C), so a -> Qa and the flow
  conjugates correctly. Closed-form, no solver.
* `so3c_interaction_set` - connection from particle bivectors, integrated
  with dopri5. Equivariant to solver tolerance.

On 20k jets, 8 epochs, the covariant model is both exactly flat under
boosts and the most accurate of the three (0.9565 vs 0.9516 for the
no-flow invariant model and 0.9424 for the broken one).
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_fixed"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Closed-form model first: it is cheap, so a failure surfaces early.
import os
SEEDS = ["1", "2"]
for name, bs in (("so3c_covariant_set", "512"),):
    print("=== %s ===" % name)
    for seed in SEEDS:
      run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", bs, "--models", name,
         "--results-dir", OUT + "/" + name,
         "--ckpt-dir", CKPT + "/" + name, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib, shutil
print("%-26s%9s%9s%10s%8s" % ("model", "params", "AUC", "rej@0.3", "hours"))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    t = r["test_metrics"]
    print("%-26s%9d%9.4f%10.0f%8.2f"
          % (r["model"], r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
print()
print("CPU reference: so3c_equivariant_set (not invariant) 0.9743 / 320")
print("               so3c_invariant_set  (no flow)        0.9689 / 183")
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_fixed.ipynb")


def build_kappa(blob: str) -> None:
    """Tier-1 item 1: does raising K past 32 still buy AUC?

    The K-sweep at K = 4/8/16/32 gave 0.852/0.935/0.964/0.970 and had not
    flattened, and every published model at 0.987 keeps the whole jet. This
    is the largest single lever in the revised plan.

    Cost is O(K^2) in both the covariant connection and the pooled readout:
    K=32 took 0.53 h/seed, so K=64 is ~2 h and K=128 ~8.5 h against a 9 h
    session cap. Hence K=64 runs first (it settles the question cheaply) and
    K=128 carries --max-seconds so it checkpoints and can resume in a second
    session rather than dying at the cap.

    Memory: train_classifier puts the whole train split on the card --
    (1.211M, 128, 5) float32 is 3.1 GB, plus 1.0 GB val -- and the readout
    holds a (B, K, K) complex matrix per channel, 67 MB at B=512/K=128. Batch
    is stepped down accordingly.
    """
    cells = [
        md("""
# K sweep: 32 -> 64 -> 128

`so3c_covariant_set` at K=32 scores 0.9746 +- 0.0001 on this protocol. The
question is whether the truncation to 32 constituents, not the architecture,
is what separates us from the 0.987 published models.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_kappa"
CKPT = "/kaggle/working/checkpoints"
import numpy as np
d = np.load(DATA + "/top_tagging_train.npz", mmap_mode="r")
print("data:", DATA, "stored K =", d["constituents"].shape[1])
"""),
        code("""
# K=128: ~8.5 h against a 9 h cap, so it checkpoints. Rerun this cell in a
# fresh session to continue; --resume picks up mid-training.
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "128", "--n-constituents", "128",
     # The eval chunk, not the training batch, is what ran the card out of
     # memory at K=128: the readout holds (chunk, K, K, hidden), which is
     # 268 MB at K=32/chunk=4096 but 4.3 GB at K=128.
     "--eval-chunk-size", "256",
     "--models", "so3c_covariant_set",
     "--results-dir", OUT + "/k128", "--ckpt-dir", CKPT + "/k128",
     "--resume", "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib
print("%-8s%9s%9s%10s%8s" % ("K", "params", "AUC", "rej@0.3", "hours"))
print("%-8s%9d%9.4f%10.0f%8.2f" % (32, 9078, 0.9746, 312, 0.53))
print("%-8s%9d%9.4f%10.0f%8.2f" % (64, 9078, 0.9772, 637, 1.66))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f))
    t = r["test_metrics"]
    k = pathlib.Path(f).parent.name.lstrip("k")
    print("%-8s%9d%9.4f%10.0f%8.2f"
          % (k, r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_kappa.ipynb")


def build_k64_seed(blob: str, seed: int = 1) -> None:
    """One extra seed at K=64, the operating point the sweep settled on.

    K = 32 / 64 / 128 gave AUC 0.9746 / 0.9772 / 0.9781 and rejection
    312 / 637 / 639. Everything worth having arrives by K=64: doubling to
    128 buys +0.0009 AUC and nothing in rejection for 3.6x the wall clock.
    """
    cells = [
        md("""
# SO3C K=64, extra seed

K=64 is the operating point: the 32 -> 64 step doubled background rejection
(312 -> 637) while 64 -> 128 added nothing (639) at 3.6x the cost. This run
adds a seed so the number carries an error bar.
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_k64"
print("data:", DATA)
"""),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--eval-chunk-size", "1024",
     "--models", "so3c_covariant_set",
     "--results-dir", OUT, "--ckpt-dir", "/kaggle/working/ckpt",
     "--resume", "--max-seconds", "9000"])
""" % seed),
        code("""
import json, glob
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("seed %d: AUC %.4f  rej %.0f  (seed 0: 0.9772 / 637)"
          % (r["seed"], t["test_auc"], t["bg_rej_30"]))
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_k64_seed%d.ipynb" % seed))



def build_message_probe(blob: str) -> None:
    """Cheap head-to-head before committing a session to message passing.

    A full canonical run of the dense 3-round model at K=64 is ~12 h by the
    measured per-iteration cost (7.5x the covariant baseline's 1.66 h), so
    it is worth ~1.4 h to find out first whether the architecture change
    pays at all, and which part of it pays.

    Same protocol for every row -- canonical splits, 400k train jets, 20
    epochs, K=32 -- so the only difference is the model:

      so3c_covariant_set          one covariant round, no scalar channel
      so3c_message_set rounds=1   + the scalar channel, still one round
      so3c_message_set rounds=3   + three rounds
      so3c_message_set rounds=3, neighbors=16   the same, on a kNN graph

    Row 2 minus row 1 isolates the scalar channel; row 3 minus row 2
    isolates the extra rounds; row 4 against row 3 says what the kNN graph
    costs in accuracy for what it saves in time. Truncating the training
    set makes the absolute numbers lower than the canonical 0.9746, so read
    the differences, not the levels.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--epochs", "20", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048",
    ]
    cells = [
        md("""
# Message passing: does it pay, and which half of it pays?

`so3c_covariant_set` applies ONE covariant rotation and carries no scalar
state. Published Lorentz-equivariant taggers do neither: LGEB and PELICAN
run several rounds and keep a scalar embedding beside the vector. This
probe adds the two things separately on a truncated protocol, so the full
run is only paid for if the differences are real.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_probe"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
print("data:", DATA)
""" % (common,)),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_covariant_set", "--results-dir", OUT + "/covariant"])
"""),
        code("""
# The scalar channel alone: one round, so the vector path matches the
# baseline and the only addition is h.
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "1",
     "--results-dir", OUT + "/r1"])
"""),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "3",
     "--results-dir", OUT + "/r3"])
"""),
        code("""
run(["benchmarks.run_top_tagging"] + COMMON +
    ["--models", "so3c_message_set", "--rounds", "3", "--neighbors", "16",
     "--results-dir", OUT + "/r3knn"])
"""),
        code("""
import json, glob, pathlib
print("%-10s%-22s%9s%9s%10s%8s"
      % ("variant", "model", "params", "AUC", "rej@0.3", "hours"))
for f in sorted(glob.glob(OUT + "/*/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-10s%-22s%9d%9.4f%10.0f%8.2f"
          % (pathlib.Path(f).parent.name, r["model"], r["n_params"],
             t["test_auc"], t["bg_rej_30"], r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_probe.ipynb")


def build_message_sweep(blob: str) -> None:
    """How steep is this architecture's capacity slope?

    That number decides the paper. PELICAN's own size sweep (its table 2,
    now in figures/pelican_scaling.csv) reaches AUC 0.9850 and
    rejection 1494 at 3k parameters and only 0.9870 / 2250 at 208k -- a
    +0.0020 slope over 70x the size. If our slope is comparably flat we are
    saturated near 14k parameters and should say so; if it is steep, there
    is room to compete on accuracy and next week's quota should buy a large
    model. Guessing costs nothing to be wrong about and everything later.

    Protocol is the probe's, exactly, so the rows compose with the four we
    already have: canonical splits, 400k train jets, 20 epochs, batch 256,
    K=32, float32, seed 0. rounds=3 / channels=4 is the anchor and is NOT
    re-run -- it scored 0.9786 / 585 in 0.58 h.

    Configurations, cheapest first, so a session cut loses the least:

      scalar_dim=0   10.4k   the clean single-variable test of the scalar
                             channel. The probe's rounds=1 row cannot play
                             this role: it also changed the input
                             normalisation, the connection bound and the
                             channel mixing.
      hidden=256     92.8k   capacity poured into the ordinary readout MLP
      scalar_dim=24  20.7k   capacity in the invariant channel
      channels=8     21.7k   capacity in the geometric state
      rounds=6       17.8k   capacity in depth
      channels=16    44.0k   the same axis, far enough to see a slope

    hidden=256 against channels=16 is the informative pair: 93k parameters
    of plain MLP against 44k parameters of geometry. Which one moves says
    what the model is actually short of. (The earlier "scaling is flat"
    result was measured on the BROKEN model, whose channels were complex
    scalar rescalings z_c = w_c z; here every channel carries its own phi.)
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--epochs", "20", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
    ]
    configs = [
        ("s0",   ["--rounds", "3", "--scalar-dim", "0"]),
        ("w256", ["--rounds", "3", "--hidden", "256"]),
        ("d24",  ["--rounds", "3", "--scalar-dim", "24", "--msg-dim", "24"]),
        ("c8",   ["--rounds", "3", "--channels", "8"]),
        ("r6",   ["--rounds", "6"]),
        ("c16",  ["--rounds", "3", "--channels", "16"]),
    ]
    cells = [
        md("""
# Capacity sweep: where is this architecture short?

Three rounds of covariant message passing scored **0.9786 / 585** at 13 862
parameters on this protocol, against **0.9731 / 266** for the single-round
covariant model. The question now is not whether the construction works but
whether it has headroom.

The comparison that sets the bar is PELICAN's own size sweep: 0.9850 / 1494
at 3k parameters, 0.9858 / 1879 at 11k, 0.9870 / 2250 at 208k. A +0.0020
AUC slope across 70x the parameters. If ours is that flat we are saturated
and the paper says so; if it is steep, there is accuracy left to win.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_sweep"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
# ~5.4 h for all six against a 9 h cap. Each config writes its own results
# dir, so --resume skips whatever already finished if the session is cut
# and this cell is rerun.
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob, pathlib
print("%-8s%-38s%9s%9s%10s%8s"
      % ("tag", "config", "params", "AUC", "rej@0.3", "hours"))
print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
      % ("--", "so3c_covariant_set (1 round, no h)", 9078, 0.9731, 266, 0.19))
print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
      % ("anchor", "rounds=3 channels=4", 13862, 0.9786, 585, 0.58))
rows = []
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        rows.append((r["n_params"], t["test_auc"], t["bg_rej_30"]))
        print("%-8s%-38s%9d%9.4f%10.0f%8.2f"
              % (tag, " ".join(extra), r["n_params"], t["test_auc"],
                 t["bg_rej_30"], r["walltime_sec"] / 3600))
if rows:
    best = max(rows, key=lambda r: r[1])
    print()
    print("slope from the 13862-parameter anchor to the best row:")
    print("  %+.4f AUC, %+.0f rejection, at %.1fx the parameters"
          % (best[1] - 0.9786, best[2] - 585, best[0] / 13862))
    print("PELICAN over 3k -> 208k (70x): +0.0020 AUC, +756 rejection")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_sweep.ipynb")


def build_beams_probe(blob: str) -> None:
    """Beams and the SOTA training recipe, separately and together.

    The capacity sweep showed that more of this architecture buys nothing.
    What both SOTA taggers have and this model lacks is information and a
    recipe, not capacity:

    * beams -- LorentzNet (arXiv:2201.08187 p.9) and PELICAN
      (arXiv:2307.16506 pp.8-9) append two beam particles (1, 0, 0, +-1).
      Their dot products with the constituents give the network lab-frame
      energies and transverse momenta, which no Lorentz invariant of the jet
      alone can supply;
    * recipe -- LorentzNet trains with AdamW (weight decay 0.01), dropout
      0.2 before the decoder, and 35 epochs of linear warm-up, cosine
      restarts and an exponential tail from lr 1e-3; PELICAN (p.12) uses the
      same schedule. This model has only ever trained with Adam at 3e-3, no
      weight decay and no dropout.

    Protocol is the earlier probe's (K=32, canonical splits, 400k train jets,
    batch 256, float32, seed 0), so the rows compose with the anchor
    rounds=3 / channels=4 at 0.9786 / 585 over 20 epochs. The beams row keeps
    20 epochs and the anchor recipe, so it isolates the beams. The recipe
    rows run the published 35 epochs, so their difference from the anchor
    carries the length as well as the optimiser, and the summary says so.

    A one-epoch smoke run with beams and the full recipe goes first and must
    write a result, so a shape or device error costs minutes, not the
    session.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
        "--rounds", "3",
    ]
    recipe = ["--optimizer", "adamw", "--weight-decay", "0.01", "--lr", "1e-3",
              "--schedule", "lorentznet", "--warmup-epochs", "4",
              "--dropout", "0.2"]
    configs = [
        ("beams", ["--epochs", "20", "--beams"]),
        ("recipe", ["--epochs", "35"] + recipe),
        ("beams_recipe", ["--epochs", "35", "--beams"] + recipe),
    ]
    cells = [
        md("""
# Beams and the SOTA recipe

The anchor message-passing model scores 0.9786 / 585 on this protocol, and
no amount of extra capacity moved it. LorentzNet and PELICAN both feed their
networks two beam particles and train with AdamW, dropout and a warm-up /
cosine-restart schedule. This probe adds each, and both.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_probe"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
# Smoke: beams and the whole recipe for one epoch on 20k jets. It has to
# write a result before the real rows are allowed to start.
import glob
smoke = list(COMMON)
smoke[smoke.index("400000")] = "20000"
flags = [f for f in CONFIGS[2][1] if f not in ("--epochs", "35")]
run(["benchmarks.run_top_tagging"] + smoke + ["--epochs", "1"] + flags +
    ["--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
print("%-14s%9s%8s%9s%10s%10s%8s"
      % ("row", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-14s%9d%8d%9.4f%10.0f%10s%8.2f"
      % ("anchor", 13862, 20, 0.9786, 585, "-", 0.58))
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        print("%-14s%9d%8d%9.4f%10.0f%10.0f%8.2f"
              % (tag, r["n_params"], r["epochs_run"], t["test_auc"],
                 t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["walltime_sec"] / 3600))
print()
print("recipe rows ran 35 epochs against the anchor's 20: their gain")
print("includes the longer schedule, not only the optimiser.")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_probe.ipynb")


def build_beams_tuning(blob: str) -> None:
    """One knob at a time, on top of beams.

    Beams moved the probe protocol from 0.9786 / 585 to 0.98088 / 767.7 at
    equal epochs, and they are now the baseline. The published LorentzNet /
    PELICAN recipe was a negative result (0.97769 / 542.8 on its own,
    0.98065 / 701.1 with beams) but it changed the optimiser, the weight
    decay, the dropout, the schedule and the epoch count together, so it
    says nothing about which of them hurt. Each row here changes one thing:

      e35  35 epochs with the anchor optimiser and schedule -- the recipe
           rows ran 35 epochs, so their comparison confounded length with
           optimiser; this isolates length.
      c8   channels 8. The capacity sweep found saturation, but it ran
           WITHOUT beams: parameters could not use information the inputs
           never carried. Worth re-asking now that they do.
      lr6  lr 6e-3 against the inherited 3e-3, which was never tuned. The
           recipe's 1e-3 went the other way and lost.
      reg  dropout 0.05 and weight decay 1e-4: the recipe's 0.2 and 0.01
           are sized for 200k-parameter models, not 15k.

    Protocol stays the probe's (K=32, canonical splits, 400k train jets,
    batch 256, float32, seed 0) so every row composes with the ones already
    measured. About 3.4 GPU-hours in total.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
        "--rounds", "3", "--beams",
    ]
    configs = [
        ("e35", ["--epochs", "35"]),
        ("c8", ["--epochs", "20", "--channels", "8"]),
        ("lr6", ["--epochs", "20", "--lr", "6e-3"]),
        ("reg", ["--epochs", "20", "--dropout", "0.05", "--weight-decay", "1e-4"]),
    ]
    cells = [
        md("""
# Beams, then one knob at a time

Beams are the new baseline: 0.98088 AUC and 767.7 background rejection on
this protocol, against 0.9786 / 585 without them. The published training
recipe lost to the anchor even with 75% more epochs, and it moved five
knobs at once. These four rows move one each: epochs, channels, learning
rate, and mild regularisation.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_tuning"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
print("data:", DATA)
""" % (common, configs)),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
print("%-8s%9s%8s%9s%10s%10s%8s"
      % ("row", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-8s%9d%8d%9.5f%10.1f%10s%8.2f" % ("anchor", 13862, 20, 0.9786, 585.0, "-", 0.58))
print("%-8s%9d%8d%9.5f%10.1f%10.1f%8.2f" % ("beams", 15038, 20, 0.98088, 767.7, 207.7, 0.64))
best = None
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        print("%-8s%9d%8d%9.5f%10.1f%10.1f%8.2f"
              % (tag, r["n_params"], r["epochs_run"], t["test_auc"],
                 t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["walltime_sec"] / 3600))
        if best is None or t["test_auc"] > best[1]:
            best = (tag, t["test_auc"], t["bg_rej_30"])
if best:
    print()
    print("best row %s: %+.5f AUC and %+.1f rejection against beams alone"
          % (best[0], best[1] - 0.98088, best[2] - 767.7))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_tuning.ipynb")


def build_beams_k64(blob: str, seed: int = 0) -> None:
    """The headline run: beams and channels 8 at K=64, canonical protocol.

    Everything measured on the cheap protocol points here. Beams took K=32
    from 0.9786 / 585 to 0.98088 / 767.7, and with beams feeding lab-frame
    information the capacity that had saturated pays again: channels 8 gives
    0.98128 / 834.4. Longer training, a larger or smaller learning rate and
    even mild regularisation all lost, so the recipe stays as it is --
    Adam at 3e-3, cosine, no dropout, no weight decay.

    Cost. The beamless channels-4 model took 7.75 h per seed for 30 epochs at
    K=64. Beams cost about 8% and channels 8 about 27%, so expect ~10.6 h --
    more than a comfortable Kaggle session. --max-seconds 30000 stops it
    cleanly with a checkpoint, and train.py now caps per session, so the
    finish kernel pattern resumes it to epoch 30 (that bug would have made a
    resumed run advance one epoch per session).

    Two seeds fit a weekly quota at this price; a third and the ensemble wait
    for the next one.
    """
    cells = [
        md("""
# Beams + channels 8, K=64, canonical protocol

The configuration every probe pointed at. References on the same canonical
protocol: the beamless message model scored 0.98073 +- 0.00006 AUC and 850
+- 56 rejection over two seeds at K=64, and `so3c_covariant_set` 0.9772 /
638. PELICAN is 0.9870 / 2250 at 208k parameters.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_k64"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Smoke: one epoch on 20k jets with the real configuration. It must write a
# result before the session commits ten hours.
import glob
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "1", "--normalize", "global",
     "--max-train-samples", "20000",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--rounds", "3", "--beams", "--channels", "8",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "64",
     "--rounds", "3", "--beams", "--channels", "8",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", OUT, "--ckpt-dir", CKPT,
     "--resume", "--max-seconds", "30000"])
""" % seed),
        code("""
import json, glob
print("%-34s%9s%8s%9s%10s%10s%8s"
      % ("model", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-34s%9d%8d%9.5f%10.1f%10s%8.2f"
      % ("covariant K=64 (reference)", 9078, 30, 0.9772, 638.0, "-", 1.66))
print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
      % ("message K=64, no beams (ref)", 13862, 30, 0.98073, 850.0, 233.0, 7.75))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
          % ("beams + channels 8, seed %d" % r["seed"], r["n_params"],
             r["epochs_run"], t["test_auc"], t["bg_rej_30"],
             t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
    if r["epochs_run"] < 30:
        print("  stopped at epoch %d on the time cap: resume with the finish "
              "kernel pattern before reporting" % r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_beams_k64_seed%d.ipynb" % seed))


def build_beams_k64_finish(blob: str) -> None:
    """Finish both headline runs, which stopped on the session cap.

    beams + channels 8 costs 1272 s per epoch at K=64, so 30 epochs is 10.6 h
    and both seeds stopped at epoch 24 against --max-seconds 30000. Six
    epochs each remain, about 2.1 h per seed.

    Kaggle hands a new session an empty /kaggle/working, so --resume alone
    would start from epoch 1. This kernel mounts both runs, copies each
    checkpoint to the path the runner derives, and resumes with exactly the
    flags the runs used (--beams --channels 8 --rounds 3 at K=64). The
    per-session cap fix is what makes this work at all: with the old
    cumulative clock a run already past the cap advanced one epoch per
    session.

    The on-card CUDA resume test runs first and must report one pass. It
    earned its place: it caught the RNG-state-on-GPU bug that a CPU-only
    test could not, and a bundle that shipped no tests at all.

    Results go to results_beams_k64_e30, so the epoch-24 files cannot be
    mistaken for these.
    """
    cells = [
        md("""
# Finish the headline runs: epochs 25-30

Both seeds of beams + channels 8 at K=64 stopped at epoch 24 on the time
cap, at AUC 0.98303 / rejection 1109 for seed 0. This resumes each from its
checkpoint and takes it to the protocol's 30 epochs.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib, shutil, subprocess, sys
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_beams_k64_e30"
CKPT = pathlib.Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)
for seed in (0, 1):
    name = "top_tagging_canonical__so3c_message_set__seed%d.pt" % seed
    found = [p for p in glob.glob("/kaggle/input/**/checkpoints/" + name, recursive=True)
             if "beams-k64" in p]
    assert found, "attach nsanya/so3c-beams-k64-seed%d: no checkpoint %s" % (seed, name)
    shutil.copy(found[0], CKPT / name)
    print("seed %d: %s (%d bytes)" % (seed, found[0], (CKPT / name).stat().st_size))
print("data:", DATA)
"""),
        code("""
# The resume path only breaks on a GPU, so the test runs on the card and
# has to pass before the session commits hours to the runs.
r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "tests/test_harness.py", "-k", "cuda_resume"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
assert r.returncode == 0 and "1 passed" in r.stdout, "CUDA resume test did not pass; not starting the runs"
"""),
        code("""
import time
for seed in ("0", "1"):
    print("=== seed %s ===" % seed, flush=True)
    t_seed = time.perf_counter()
    run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", "256", "--n-constituents", "64",
         "--rounds", "3", "--beams", "--channels", "8",
         "--eval-chunk-size", "1024",
         "--models", "so3c_message_set",
         "--results-dir", OUT, "--ckpt-dir", str(CKPT),
         "--resume", "--max-seconds", "26000"])
    print("seed %s: this session took %.1f min" % (seed, (time.perf_counter() - t_seed) / 60))
"""),
        code("""
import json, glob
print("%-34s%9s%8s%9s%10s%10s%8s"
      % ("model", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours"))
print("%-34s%9d%8d%9.5f%10.1f%10s%8.2f"
      % ("covariant K=64 (reference)", 9078, 30, 0.97720, 638.0, "-", 1.66))
print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
      % ("message K=64, no beams (ref)", 13862, 30, 0.98073, 850.0, 233.0, 7.75))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-34s%9d%8d%9.5f%10.1f%10.1f%8.2f"
          % ("beams + channels 8, seed %d" % r["seed"], r["n_params"],
             r["epochs_run"], t["test_auc"], t["bg_rej_30"],
             t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
    assert r["epochs_run"] == 30, "seed %d stopped at epoch %d" % (r["seed"], r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_beams_k64_finish.ipynb")


def build_capacity_cost(blob: str) -> None:
    """Two questions for the last hours of the window.

    c16 -- does capacity keep paying? Channels 8 with beams gave +0.00040 AUC
    and +67 rejection over beams alone on the probe protocol, where the same
    change WITHOUT beams had been flat. If 16 holds that slope, capacity is
    still not the binding constraint.

    K=128 cost -- is more of the jet reachable at all? The structural gap to
    LorentzNet and PELICAN is that they use every constituent (up to 200) and
    this model uses 64. At K=64 it costs 1272 s per epoch on the full
    training set and the pairwise terms grow as K^2, so K=128 is 30+ h per
    seed unless the graph is sparsified. kNN measured useless at K=32 (0.0011
    AUC lost for 2 percent of the wall clock saved) because expm_so3c and the
    pooled readout dominated there -- but those grow as K while the edge
    terms grow as K^2, so the verdict may flip. Two one-epoch runs on 100k
    jets measure it, dense against kNN(16). They are beamless because beams
    are implemented for the dense graph only, and what is being measured here
    is wall clock, not accuracy.
    """
    probe = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--models", "so3c_message_set", "--rounds", "3",
    ]
    configs = [
        ("c16", ["--max-train-samples", "400000", "--epochs", "20",
                 "--n-constituents", "32", "--eval-chunk-size", "2048",
                 "--beams", "--channels", "16"]),
        ("k128_dense", ["--max-train-samples", "100000", "--epochs", "1",
                        "--n-constituents", "128", "--eval-chunk-size", "256",
                        "--channels", "8"]),
        ("k128_knn16", ["--max-train-samples", "100000", "--epochs", "1",
                        "--n-constituents", "128", "--eval-chunk-size", "256",
                        "--channels", "8", "--neighbors", "16"]),
    ]
    cells = [
        md("""
# Capacity, and whether K=128 is reachable

Beams plus channels 8 took the canonical protocol to 0.98333 / 1131 at
K=64. Two questions remain cheap enough for the last hours of this quota:
whether channels 16 keeps the capacity slope, and whether sparsifying the
graph makes K=128 affordable. Every published tagger ahead of us uses the
whole jet; we use 64 constituents of up to 200.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_capacity_cost"
CKPT = "/kaggle/working/checkpoints"
PROBE = %r
PROBE[PROBE.index("DATA")] = DATA
CONFIGS = %r
import numpy as np
d = np.load(DATA + "/top_tagging_train.npz", mmap_mode="r")
print("data:", DATA, "| stored K =", d["constituents"].shape[1])
""" % (probe, configs)),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + PROBE + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
print("%-12s%9s%8s%8s%9s%10s%9s"
      % ("row", "params", "K", "epochs", "AUC", "rej@0.3", "s/epoch"))
print("%-12s%9d%8d%8d%9.5f%10.1f%9s" % ("beams (ref)", 15038, 32, 20, 0.98088, 767.7, "115"))
print("%-12s%9d%8d%8d%9.5f%10.1f%9s" % ("c8 (ref)", 22834, 32, 20, 0.98128, 834.4, "146"))
rows = {}
for tag, extra in CONFIGS:
    K = extra[extra.index("--n-constituents") + 1]
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        per = r["walltime_sec"] / max(r["epochs_run"], 1)
        rows[tag] = per
        print("%-12s%9d%8s%8d%9.5f%10.1f%9.0f"
              % (tag, r["n_params"], K, r["epochs_run"], t["test_auc"],
                 t["bg_rej_30"], per))
if "k128_dense" in rows and "k128_knn16" in rows:
    d, k = rows["k128_dense"], rows["k128_knn16"]
    print()
    print("K=128 on 100k jets: dense %.0f s/epoch, kNN(16) %.0f s/epoch, %.2fx"
          % (d, k, d / k))
    print("full 1.211M train, 30 epochs: dense %.1f h, kNN %.1f h"
          % (d * 12.11 * 30 / 3600, k * 12.11 * 30 / 3600))
    print("(K=64 with beams and channels 8 was 1272 s/epoch, 10.6 h)")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_capacity_cost.ipynb")


def build_gap_screen(blob: str) -> None:
    """Screen every candidate from the gap-to-SOTA analysis on one protocol.

    Base configuration is the best probe row, beams + channels 8 (0.98128 AUC,
    834.4 rejection on this protocol, commit c325950). Each row adds one
    candidate, then one row combines the architectural ones:

      fixes    --no-mass-input --no-self-edges: the two verified defects. The
               per-constituent m^2 is rounding noise, and LorentzNet masks the
               self-edge. CPU cost 1.00x.
      vector   --vector-channel: the bivector lift discards each constituent's
               component along the jet axis; arXiv:2606.21790 finds vector
               channels dominant in L-GATr. CPU cost 1.10x.
      relnorm  --relnorm-edge: LorentzNet's |x_i - x_j|^2 analogue. CPU 1.30x.
      pair8    --pair-latent 8: a rank-2 pair state through the rounds with 7 of
               PELICAN's 15 equivariant aggregators; by PELICAN's table 2 the
               pair state is the likeliest carrier of its 20x parameter
               efficiency over this model.
      combo    vector + pair8 + fixes.
      rounds6  --rounds 6. Depth was flat WITHOUT beams (-0.0006), but so was
               capacity, which paid once beams supplied the information, so
               depth is re-asked with beams. ~2x the per-round cost.
      falpha3  --falpha 3: CPU cost 3.54x, so it runs last and a session cut
               costs it rather than the cheaper rows.

    Protocol is the probe's (K=32, canonical splits, 400k train jets, 20
    epochs, batch 256, float32, seed 0), so every row compares directly with
    the rows already measured. One seed per row: at this protocol a gain of
    ~0.0004 AUC is about two seed spreads, so a winner is confirmed at K=64
    before it is believed.

    A one-epoch smoke run with every new switch at once runs first and must
    write a result before any real row starts.
    """
    common = [
        "--cache-dir", "DATA", "--representation", "constituents",
        "--canonical-splits", "--normalize", "global",
        "--max-train-samples", "400000",
        "--seed", "0", "--device", "cuda", "--dtype", "float32",
        "--batch-size", "256", "--n-constituents", "32",
        "--eval-chunk-size", "2048", "--models", "so3c_message_set",
        "--rounds", "3", "--beams", "--channels", "8",
    ]
    configs = [
        ("fixes", ["--epochs", "20", "--no-mass-input", "--no-self-edges"]),
        ("vector", ["--epochs", "20", "--vector-channel"]),
        ("relnorm", ["--epochs", "20", "--relnorm-edge"]),
        ("pair8", ["--epochs", "20", "--pair-latent", "8"]),
        ("combo", ["--epochs", "20", "--vector-channel", "--pair-latent", "8",
                   "--no-mass-input", "--no-self-edges"]),
        ("rounds6", ["--epochs", "20", "--rounds", "6"]),
        ("falpha3", ["--epochs", "20", "--falpha", "3"]),
    ]
    smoke_flags = ["--vector-channel", "--pair-latent", "8", "--no-mass-input",
                   "--no-self-edges", "--relnorm-edge", "--falpha", "3"]
    cells = [
        md("""
# Gap screen: every candidate on one protocol

Beams + channels 8 scores 0.98128 AUC and 834.4 rejection on this protocol.
Each row adds one candidate from the gap-to-SOTA analysis -- the two verified
defect fixes, a vector channel, a relative-norm edge feature, a rank-2 pair
latent -- then the architectural ones together, then the expensive f_alpha
embedding last.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_gap_screen"
CKPT = "/kaggle/working/checkpoints"
COMMON = %r
COMMON[COMMON.index("DATA")] = DATA
CONFIGS = %r
SMOKE = %r
print("data:", DATA)
""" % (common, configs, smoke_flags)),
        code("""
# Smoke: every new switch at once, one epoch on 20k jets. It must write a
# result before the real rows are allowed to start.
import glob
smoke = list(COMMON)
smoke[smoke.index("400000")] = "20000"
run(["benchmarks.run_top_tagging"] + smoke + ["--epochs", "1"] + SMOKE +
    ["--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
for tag, extra in CONFIGS:
    print("=== %s ===" % tag, flush=True)
    run(["benchmarks.run_top_tagging"] + COMMON + extra +
        ["--results-dir", OUT + "/" + tag,
         "--ckpt-dir", CKPT + "/" + tag, "--resume",
         "--max-seconds", "26000"])
"""),
        code("""
import json, glob
REF_AUC, REF_REJ = 0.98128, 834.4
print("%-9s%9s%9s%10s%10s%10s%9s%8s"
      % ("row", "params", "AUC", "dAUC", "rej@0.3", "d rej", "rej@0.5", "hours"))
print("%-9s%9d%9.5f%10s%10.1f%10s%9s%8.2f" % ("beams", 15038, 0.98088, "-", 767.7, "-", "207.7", 0.64))
print("%-9s%9d%9.5f%10s%10.1f%10s%9s%8.2f" % ("c8 (base)", 22834, REF_AUC, "-", REF_REJ, "-", "225.9", 0.81))
for tag, extra in CONFIGS:
    for f in sorted(glob.glob(OUT + "/" + tag + "/*.json")):
        r = json.load(open(f)); t = r["test_metrics"]
        print("%-9s%9d%9.5f%+10.5f%10.1f%+10.1f%9.1f%8.2f"
              % (tag, r["n_params"], t["test_auc"], t["test_auc"] - REF_AUC,
                 t["bg_rej_30"], t["bg_rej_30"] - REF_REJ,
                 t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
print()
print("one seed per row: treat about +0.0004 AUC as the edge of noise, and")
print("confirm any winner at K=64 before believing it.")
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_gap_screen.ipynb")


def build_gpu_profile(blob: str) -> None:
    """Where the GPU time goes, measured on the card.

    Beams + channels 8 at K=64 costs 1272 s per epoch on a P100, about
    0.24 s per training step of 256 jets. A bandwidth estimate from the
    tensor sizes puts a step several times lower, and CPU profiles cannot
    settle it (complex batched matmuls run as per-matrix loops there). This
    kernel times the training step and the eval pass and ranks the CUDA
    kernels, for the headline configuration and the two costliest screen
    candidates, on synthetic jets: cost depends only on shapes. No dataset
    is attached. About 10 minutes of GPU including the torch reinstall.
    """
    cells = [
        md("""
# GPU profile of so3c_message_set

Times the training step and eval forward pass at K=64 / batch 256 and ranks
the CUDA kernels for beams + channels 8, with the vector channel, and with the
pair latent. Synthetic jets; the cost depends only on shapes.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
OUT = "/kaggle/working/profile_out"
ok = run(["benchmarks.profile_gpu", "--out", OUT, "--k", "64",
          "--batch-size", "256", "--eval-chunk-size", "1024",
          "--config", "base", "--config", "vector", "--config", "pair8"])
assert ok, "profile failed"
ok = run(["benchmarks.profile_gpu", "--out", OUT + "_k32", "--k", "32",
          "--batch-size", "256", "--eval-chunk-size", "2048",
          "--config", "base", "--no-profile"])
"""),
        code("""
# torch.compile on the same card, same shapes. Triton needs sm_70+, so on a
# P100 this reports the failure and the eager numbers above stand alone.
ok = run(["benchmarks.profile_gpu", "--out", OUT + "_compiled", "--k", "64",
          "--batch-size", "256", "--eval-chunk-size", "1024", "--compile",
          "--config", "base", "--config", "vector", "--config", "pair8",
          "--no-profile"])
"""),
        code("""
import pathlib
for f in ["base_by_kernel.txt", "base_by_shape.txt", "base_by_line.txt"]:
    p = pathlib.Path(OUT) / f
    print("=====", f)
    print(chr(10).join(p.read_text().splitlines()[:45]))
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_gpu_profile.ipynb")


# Canonical K=64 runs built with build_k64_run: (tag, extra flags, seed).
K64_RUNS = [
    # (tag, extra flags, seed, epochs)
    # third headline seed, on the 2x faster forward pass (47ec4ad): checks
    # that the rewrite reproduces the headline on real data and measures the
    # real epoch time before any winner is confirmed with it
    ("headline", [], 2, 30),
    # Is the headline converged at 30 epochs? Validation loss is lowest at
    # the FINAL epoch in all four K=64 runs, and on the older 9k
    # so3c_equivariant_set at K=32 the same doubling (results_60ep, one seed
    # against three) bought +0.00024 AUC and +30 rejection, 6.4 and 5.2
    # standard deviations of that model's seed spread. The "longer training
    # is negative" verdict rests on a single probe-protocol row (35 vs 20
    # epochs on 400k jets), a regime where nothing capacity-like can pay.
    ("headline60", [], 0, 60),
    # Window 2026-10-03: does width transfer to K=64? w64s24 and a control
    # rerun of the headline configuration, both seed 0, 30 epochs, pushed
    # together so they run the same software (T4, torch 2.10, compiled) and
    # see the same batch order. The old seed-0 headline ran on a P100 under
    # torch 2.5.1, where matching CUDA randperm output is unverified, so the
    # control is rerun rather than reused. --keep-best on both; it draws no
    # randomness and cannot change training.
    ("w64s24", ["--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24",
                "--keep-best"], 0, 30),
    ("headline_t4", ["--keep-best"], 0, 30),
]

# Window 2026-10-17: the lab-frame pair features at K=64, the protocol the
# headline numbers are measured on. --pair-lab gained +0.00051 on the probe
# protocol and +0.00064 at the K=32 long schedule (unpaired against pooled
# controls, Welch t = 5.9 and 6.8) at no extra training time. Seeds 0 and 1,
# 60 epochs, read unpaired against the 60-epoch headline (seed 0, one run,
# torch 2.10: 0.98363 / 1140.8). Pinned to the T4 with torch 2.11 like this
# window's other runs; --keep-best is free. The headline took 8.73 h of
# kernel time for 60 epochs, inside the 39000 s cap.
K64_RUNS_PINNED = [
    ("pairlab60", ["--keep-best", "--pair-lab"], 0, 60),
    ("pairlab60", ["--keep-best", "--pair-lab"], 1, 60),
]

# The K=64 references a pinned run is printed against (results_k64/): the
# 60-epoch headline at epoch 60 (AUC, rejection at 30%, final validation
# loss) and its continuation to 120 epochs.
K64_REF60 = (0.98363, 1140.8, 0.16276)
K64_REF120 = (0.98392, 1201.9, 0.16184)


def k64_metadata(tag: str, seed: int) -> pathlib.Path:
    """Kernel metadata for a build_k64_run notebook; returns its path."""
    name = "k64-%s-seed%d" % (tag.replace("_", "-"), seed)
    meta = {
        "id": "nsanya/so3c-" + name,
        "title": "SO3C K64 %s seed%d" % (tag, seed),
        "code_file": "kaggle_k64_%s_seed%d.ipynb" % (tag, seed),
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    path = NB_DIR / ("kernel-metadata-%s.json" % name)
    path.write_text(json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return path


def build_k64_run(blob: str, tag: str, flags: list[str], seed: int,
                  epochs: int = 30, max_seconds: int = 39000,
                  pinned: bool = False) -> str:
    """One canonical K=64 run of so3c_message_set: beams + channels 8 plus
    ``flags``, one seed. Returns the notebook file name.

    The generic form of build_beams_k64, for confirming screen winners and
    for extra headline seeds. Since 2a3b344/47ec4ad the headline
    configuration costs ~593 s per epoch on a T4 (1219 s before), so 30
    epochs fit one session; --max-seconds still stops cleanly with a
    checkpoint if a heavier variant does not, and the finish-kernel pattern
    resumes it.

    Provenance: results land in results_k64/<tag>, and the notebook records
    the sha256 of its code bundle, so a result can be tied to the exact code
    that produced it.

    With `pinned`, the T4 / torch 2.11.0+cu128 setup of the screens replaces
    GPU_SETUP, and a last cell asserts every result's stack, prints its
    batch-order fingerprint and its differences from the 60- and 120-epoch
    headline runs.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8",
            # 1.24x on a T4 and checked against eager on the first batch;
            # a P100 falls back to eager by itself
            "--compile"] + list(flags)
    cells = [
        md("""
# K=64 canonical run: %s, seed %d

so3c_message_set with beams + channels 8%s, %d epochs. Reference on the same
protocol: beams + channels 8, seeds 0 and 1, 0.98333 +- 0.00010 AUC and
1131 +- 22 rejection at 30%% signal efficiency. PELICAN: 0.9870 / 2250.%s
""" % (tag, seed, (" + " + " ".join(flags)) if flags else "", epochs,
       (chr(10) + chr(10) + "T4 with torch 2.11.0+cu128 pinned and asserted.") if pinned else "")),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED if pinned else GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEED, EPOCHS = %r, %d, %d
OUT = "/kaggle/working/results_k64/" + TAG
CKPT = "/kaggle/working/checkpoints/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, seed, epochs, base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# Smoke: one epoch on 20k jets with the real configuration. It must write a
# result before the session commits hours.
import glob
run(["benchmarks.run_top_tagging"] + BASE +
    ["--epochs", "1", "--max-train-samples", "20000", "--seed", "0",
     "--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code("""
ok = run(["benchmarks.run_top_tagging"] + BASE +
         ["--epochs", str(EPOCHS), "--seed", str(SEED),
          "--results-dir", OUT, "--ckpt-dir", CKPT,
          "--resume", "--max-seconds", "%d"])
assert ok, "training run failed"
""" % max_seconds),
        code("""
import json, glob
print("%-30s%9s%8s%9s%10s%10s%8s%10s"
      % ("run", "params", "epochs", "AUC", "rej@0.3", "rej@0.5", "hours", "s/epoch"))
print("%-30s%9d%8d%9.5f%10.1f%10.1f%8.2f%10s"
      % ("beams + c8, 2 seeds (ref)", 22834, 30, 0.98333, 1131.0, 292.0, 10.6, "1272"))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-30s%9d%8d%9.5f%10.1f%10.1f%8.2f%10.0f"
          % ("%s seed %d" % (TAG, r["seed"]), r["n_params"], r["epochs_run"],
             t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
             r["walltime_sec"] / 3600, r["walltime_sec"] / max(r["epochs_run"], 1)))
    if r["epochs_run"] < EPOCHS:
        print("  stopped at epoch %d on the time cap: finish before reporting"
              % r["epochs_run"])
    print()
    print("epoch-by-epoch (is it still improving at the end?)")
    print("%6s%12s%11s%10s" % ("epoch", "train_loss", "val_loss", "val_acc"))
    for e in r["history"]:
        print("%6d%12.5f%11.5f%10.5f"
              % (e["epoch"], e["train_loss"], e["val_loss"], e["val_acc"]))
    vl = [e["val_loss"] for e in r["history"]]
    best = min(range(len(vl)), key=vl.__getitem__)
    print("best val_loss at epoch %d (%.5f); last %.5f; last 5 epochs moved %+.5f"
          % (r["history"][best]["epoch"], vl[best], vl[-1], vl[-1] - vl[-6]))
"""),
    ]
    if pinned:
        cells.append(code("""
# The stack and batch order of every result, and its differences from the
# K=64 headline: the 60-epoch run %s (torch 2.10, seed 0) and its
# continuation to 120 epochs %s. Same-seed runs in different kernels do not
# reproduce on this stack, so these are unpaired comparisons.
import json, glob
REF60, REF120 = %r, %r
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]; env = r["env"]; bo = r["batch_order"]
    print("seed %%d: torch %%s on %%s | batch order epoch 1 %%s, chain %%s"
          %% (r["seed"], env["torch"], env.get("gpu"), bo["epoch1"], bo["chain"]))
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    vl = r["history"][-1]["val_loss"]
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the time cap; finish before comparing"
              %% (r["epochs_run"], EPOCHS))
        continue
    for name, ref in (("60-epoch headline", REF60), ("120-epoch headline", REF120)):
        print("  against the %%s: dAUC %%+.5f, d rej@0.3 %%+.1f, d final val loss %%+.5f"
              %% (name, t["test_auc"] - ref[0], t["bg_rej_30"] - ref[1], vl - ref[2]))
    if r.get("test_metrics_best_val"):
        tb = r["test_metrics_best_val"]
        print("  best-val weights (epoch %%d, val loss %%.5f): AUC %%.5f, rej@0.3 %%.1f"
              %% (r["best_val_epoch"], r["best_val_loss"], tb["test_auc"], tb["bg_rej_30"]))
""" % (K64_REF60, K64_REF120, K64_REF60, K64_REF120)))
    name = "kaggle_k64_%s_seed%d.ipynb" % (tag, seed)
    _finalise(cells, NB_DIR / name)
    k64_metadata(tag, seed)
    return name


# Two seeds in one kernel, one per card. Kaggle's T4 sessions have two
# T4s ("Tesla T4" twice in every nvidia-smi line since 2026-09-19) and charge
# quota per session hour, so a single-card kernel leaves half the paid
# hardware idle. Each seed runs in its own process with
# CUDA_VISIBLE_DEVICES set, after a smoke test that runs on both cards at
# once; if either card fails it, the kernel runs the first seed alone.
K64_DUAL_RUNS = [
    # (tag, extra flags, seeds, epochs). Window 2026-10-17: the lab-frame
    # pair features at K=64 (see K64_RUNS_PINNED), and a same-stack control
    # of the 60-epoch headline: the headline ran on torch 2.10 (8.73 h), and
    # on 2.11 runs do not reproduce across kernels.
    ("pairlab60", ["--keep-best", "--pair-lab"], (0, 1), 60),
    ("base60", ["--keep-best"], (0, 1), 60),
    # Window 2026-10-24: the K=64 confirmation of the pair latent on top of
    # --pair-lab, which paid at the K=32 long schedule (+0.00033 over
    # pairlab211, both runs above both of its runs; 3beaff8). At ~1.49x
    # pairlab60's 517-541 s per epoch, 60 epochs do not fit one session: the
    # run stops on --max-seconds near epoch 49 (nearer 37 for a seed whose
    # compile check falls back to eager) and build_k64_finish_dual resumes it.
    ("pairlatlab60", ["--keep-best", "--pair-lab", "--pair-latent", "8"], (0, 1), 60),
    # Window 2026-10-31: the K=64 confirmation of width on top of --pair-lab,
    # which passed at the K=32 long schedule (+0.00047 over pairlab211,
    # 8ace328). float32 pair features, as its K=32 runs and pairlab60. At
    # ~1.72x pairlab60's epoch (T4 profile ca4f885) it stops near epoch 40 and
    # build_k64_finish_dual resumes it.
    ("w64s24lab60", ["--keep-best", "--pair-lab", "--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24"], (0, 1), 60),
]

DUAL_LAUNCH = """
import os, subprocess, sys

def launch(args, gpu, log):
    \"\"\"Start a repo module on one card; its output goes to `log`.\"\"\"
    # unbuffered, with faulthandler: a killed or crashed child still leaves
    # its last lines and a traceback in `log`
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONFAULTHANDLER="1")
    return subprocess.Popen([sys.executable, "-m"] + args, cwd="/kaggle/working/repo",
                            env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)

def tail(log, n=3000):
    return open(log).read()[-n:]

cards = [l for l in subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.splitlines() if l.strip()]
NGPU = len(cards)
print("cards:", cards)
"""


def build_k64_dual(blob: str, tag: str, flags: list[str], seeds: tuple[int, ...],
                   epochs: int = 60, max_seconds: int = 39000) -> str:
    """Two K=64 seeds of one configuration in one kernel, one per T4.

    The command is build_k64_run's (beams + channels 8 plus ``flags``) on the
    pinned T4 / torch 2.11 setup. A smoke test runs one epoch on 20k jets on
    both cards at once; the real runs go in parallel only if both write a
    result, otherwise the first seed runs alone on card 0. Each seed has its
    own results and checkpoint directory (results_k64/<tag>/seed<s>), its
    own --max-seconds, and its own log in /kaggle/working/logs.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8",
            "--compile"] + list(flags)
    cells = [
        md("""
# K=64 canonical runs, two cards: %s, seeds %s

so3c_message_set with beams + channels 8%s, %d epochs, one seed per T4 in
parallel. T4 with torch 2.11.0+cu128 pinned and asserted. References: the
60-epoch headline %s (torch 2.10, seed 0) and its 120-epoch continuation %s.
""" % (tag, ", ".join(str(s) for s in seeds),
       (" + " + " ".join(flags)) if flags else "", epochs, K64_REF60, K64_REF120)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS, EPOCHS, MAX_SECONDS = %r, %r, %d, %d
OUT = "/kaggle/working/results_k64/" + TAG
CKPT = "/kaggle/working/checkpoints/" + TAG
LOGS = "/kaggle/working/logs/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
for d in (OUT, LOGS):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, list(seeds), epochs, max_seconds, base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# Smoke on both cards at once, as the real runs will start, each card with
# its own seed, so both seeds' initial weights go through the first-batch
# compile check (on 20k jets' normalisation, not the full set's): one epoch
# on 20k jets per card. Each must write a result.
import glob, json, time
smoke = []
for i in range(min(NGPU, len(SEEDS))):
    d = "/kaggle/working/smoke/%s/card%d" % (TAG, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", str(SEEDS[i]), "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
works = []
for i, ((d, _), rc) in enumerate(zip(smoke, codes)):
    found = glob.glob(d + "/*.json")
    print("card %d: exit %d, result %s" % (i, rc, bool(found)))
    if found:
        r = json.load(open(found[0]))
        print("   seed %d: %.0f s, AUC %.4f, compile %s, %s"
              % (r["seed"], r["walltime_sec"], r["test_metrics"]["test_auc"],
                 r.get("compile_check"), r["env"]))
        works.append(i)
    else:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert works, "the smoke run failed on every card"
PARALLEL = len(works) >= len(SEEDS)
print("parallel:" if PARALLEL else "only card(s) %s work: seed %d alone" % (works, SEEDS[0]),
      "seeds", SEEDS if PARALLEL else SEEDS[:1])
"""),
        code("""
import glob, time
todo = SEEDS if PARALLEL else SEEDS[:1]
procs = []
for i, s in enumerate(todo):
    gpu = i if PARALLEL else works[0]
    procs.append((s, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", str(EPOCHS), "--seed", str(s),
                             "--results-dir", OUT + "/seed%d" % s,
                             "--ckpt-dir", CKPT + "/seed%d" % s,
                             "--resume", "--max-seconds", str(MAX_SECONDS)],
                            gpu, LOGS + "/seed%d.log" % s)))
    print("seed %d on card %d" % (s, gpu), flush=True)
BAD = []
for s, p in procs:
    rc = p.wait()
    print("=== seed %d: exit %d ===" % (s, rc))
    print(tail(LOGS + "/seed%d.log" % s, 2000))
    if rc != 0 or not glob.glob(OUT + "/seed%d/*.json" % s):
        BAD.append(s)
# a failed seed is reported after the summary below, so the other seed's
# result is still printed
print("failed seeds:", BAD or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code("""
# Every result against the K=64 headline: the 60-epoch run (torch 2.10, seed 0)
# and its continuation to 120 epochs. Same-seed runs in different kernels do
# not reproduce on this stack, so these are unpaired comparisons.
import json, glob
REF60, REF120 = %r, %r
for f in sorted(glob.glob(OUT + "/seed*/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]; env = r["env"]; bo = r["batch_order"]
    h = r["history"]; vl = [e["val_loss"] for e in h]
    print("seed %%d: %%d epochs, %%.2f h (%%.0f s/epoch) | torch %%s on %%s | batch order epoch 1 %%s"
          %% (r["seed"], r["epochs_run"], r["walltime_sec"] / 3600,
             r["walltime_sec"] / max(r["epochs_run"], 1), env["torch"], env.get("gpu"), bo["epoch1"]))
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    print("  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f  (lowest %%.5f at epoch %%d)"
          %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), vl[-1],
             min(vl), h[vl.index(min(vl))]["epoch"]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the time cap; finish before comparing"
              %% (r["epochs_run"], EPOCHS))
        continue
    for name, ref in (("60-epoch headline", REF60), ("120-epoch headline", REF120)):
        print("  against the %%s: dAUC %%+.5f, d rej@0.3 %%+.1f, d final val loss %%+.5f"
              %% (name, t["test_auc"] - ref[0], t["bg_rej_30"] - ref[1], vl[-1] - ref[2]))
    if r.get("test_metrics_best_val"):
        tb = r["test_metrics_best_val"]
        print("  best-val weights (epoch %%d): AUC %%.5f, rej@0.3 %%.1f"
              %% (r["best_val_epoch"], tb["test_auc"], tb["bg_rej_30"]))
assert not BAD, "seeds failed (exit code or no result): %%s" %% BAD
""" % (K64_REF60, K64_REF120)),
    ]
    stem = "k64_%s_seeds%s" % (tag, "".join(str(s) for s in seeds))
    name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / name)
    slug = "k64-%s-seeds%s" % (tag.replace("_", "-"), "".join(str(s) for s in seeds))
    meta = {
        "id": "nsanya/so3c-" + slug,
        "title": "SO3C K64 %s seeds%s" % (tag, "".join(str(s) for s in seeds)),
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


# Two-card K=64 runs that stop on --max-seconds before their last epoch,
# finished in a second kernel: (tag, flags, seeds, epochs) as in
# K64_DUAL_RUNS. Prepared before the first kernel runs; push it only after
# that kernel has stopped, so its output can be mounted.
# A finish kernel's --max-seconds when the week's quota cannot cover the rest
# of the run: it then stops cleanly with a checkpoint, and a later kernel of
# the same kind resumes it. Default 39000 s.
K64_FINISH_MAX_SECONDS = {}

K64_FINISH_DUAL = [
    ("pairlatlab60", ["--keep-best", "--pair-lab", "--pair-latent", "8"], (0, 1), 60),
    ("w64s24lab60", ["--keep-best", "--pair-lab", "--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24"], (0, 1), 60),
]

# Reading of a finished K=64 confirmation, fixed before the push, on the final
# weights against pairlab60's two runs (the same stack, two-card kernel, 60
# epochs): confirmed if the mean gains at least +0.0003 AUC and every run lies
# above both pairlab60 runs; not confirmed if the mean gains nothing (<= 0);
# otherwise unresolved. The same form as LONG_DUAL_RULE. At K=64 the pooled
# within-configuration sd of base60 and pairlab60 is ~0.0003 (2 df), so
# +0.0003 is about one standard error of a two-against-two difference.
K64_CONFIRM_RULE = (0.0003, 0.0)


def _k64_references() -> dict:
    """The same-stack K=64 runs (T4, torch 2.11, two-card kernels) a
    confirmation is read against, final weights: pairlab60 and base60 at 60
    epochs, pairlab120 at 120."""
    refs = {}
    for tag, epochs in (("pairlab60", 60), ("base60", 60), ("pairlab120", 120)):
        rows = []
        for f in sorted((ROOT / "results_k64" / tag).glob("seed*/*.json")):
            r = json.loads(f.read_text())
            assert r["env"]["torch"] == "2.11.0+cu128" and r["epochs_run"] == epochs, f
            t, h = r["test_metrics"], r["history"][-1]
            rows.append({"seed": r["seed"], "auc": t["test_auc"], "rej30": t["bg_rej_30"],
                         "val_loss": h["val_loss"], "gap": h["val_loss"] - h["train_loss"],
                         "order": r["batch_order"]["epoch1"]})
        refs[tag] = rows
    assert [[x["seed"] for x in v] for v in refs.values()] == [[0, 1]] * 3, refs
    return refs


def _pooled_sd(groups) -> tuple[float, int]:
    """Pooled within-group sd of a list of lists, and its degrees of freedom."""
    import math
    ss = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups if len(g) > 1)
    df = sum(len(g) - 1 for g in groups if len(g) > 1)
    return math.sqrt(ss / df), df


def build_k64_finish_dual(blob: str, tag: str, flags: list[str],
                          seeds: tuple[int, ...] = (0, 1), epochs: int = 60,
                          max_seconds: int = 39000) -> str:
    """Finish a two-card K=64 run (build_k64_dual) that stopped on the session cap.

    Mounts the first kernel's output (nsanya/so3c-k64-<tag>-seeds01) through
    kernel_sources. For each seed its last checkpoint must agree with the
    result JSON that kernel wrote when it stopped (epoch, batch order); it is
    copied where --resume looks, and the run resumes with the same command and
    the same --epochs, so the cosine schedule is unchanged. A seed that
    already finished there is read from its mounted result; a seed with no
    checkpoint is reported, not restarted. The smoke and the parallel launch
    are build_k64_dual's. The summary reads the finished runs against
    pairlab60 by K64_CONFIRM_RULE, and prints base60 and pairlab120 beside.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8",
            "--compile"] + list(flags)
    seeds_s = "".join(str(s) for s in seeds)
    src_kernel = "nsanya/so3c-k64-%s-seeds%s" % (tag.replace("_", "-"), seeds_s)
    refs = _k64_references()
    pooled = _pooled_sd([[x["auc"] for x in refs["pairlab60"]],
                         [x["auc"] for x in refs["base60"]]])
    cells = [
        md("""
# Finish, two cards: K=64 %s, seeds %s

Resumes the two-card K=64 run %s (beams + channels 8 + %s, %d epochs) from
the last checkpoints in its mounted output, with the same command and
horizon, one seed per T4. Push only after that kernel has stopped. T4 with
torch 2.11.0+cu128 pinned and asserted. Read against pairlab60.
""" % (tag, ", ".join(str(s) for s in seeds), src_kernel, " ".join(flags), epochs)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS, EPOCHS, MAX_SECONDS = %r, %r, %d, %d
SRC = %r
OUT = "/kaggle/working/results_k64/" + TAG
CKPT = "/kaggle/working/checkpoints/" + TAG
LOGS = "/kaggle/working/logs/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
for d in (OUT, LOGS):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, list(seeds), epochs, max_seconds, src_kernel, base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# The first kernel's last checkpoints, from its mounted output. Each must
# agree with the result JSON that kernel wrote when it stopped (epoch, batch
# order) and goes where --resume looks; the result JSON does not, so the
# resumed run is not skipped.
import glob, json, pathlib, shutil, torch
TODO, DONE = [], {}
for s in SEEDS:
    pat = "/kaggle/input/**/%s/" + TAG + "/seed%d/*seed%d.%s"
    cks = glob.glob(pat % ("checkpoints", s, s, "pt"), recursive=True)
    js = glob.glob(pat % ("results_k64", s, s, "json"), recursive=True)
    if len(cks) != 1:
        print("seed %d: %d mounted checkpoints, not finished here: %r" % (s, len(cks), cks))
        continue
    st = torch.load(cks[0], map_location="cpu", weights_only=False)
    rec = json.load(open(js[0])) if len(js) == 1 else None
    print("seed %d: checkpoint at epoch %d (history %d), first kernel's result at epoch %s"
          % (s, st["epoch"], len(st["history"]), rec["epochs_run"] if rec else None))
    assert len(st["history"]) == st["epoch"], (len(st["history"]), st["epoch"])
    if rec is not None:
        assert rec["epochs_run"] == st["epoch"], (rec["epochs_run"], st["epoch"])
        assert rec["batch_order"]["epoch1"] == st["batch_order_epoch1"], s
    if st["epoch"] >= EPOCHS:
        assert rec is not None, "seed %d finished but its result is not mounted" % s
        DONE[s] = rec
        continue
    ck = pathlib.Path(CKPT, "seed%d" % s, pathlib.Path(cks[0]).name)
    ck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cks[0], ck)
    TODO.append(s)
    del st
assert TODO or DONE, "nothing mounted: attach the output of " + SRC
print("to finish:", TODO, "| finished in the first kernel:", sorted(DONE))
"""),
        code("""
# Smoke on both cards at once, each card with its own seed: one epoch on 20k
# jets per card, fresh models in separate directories. Each must write a
# result.
import glob, json
smoke = []
for i, s in enumerate(TODO[:NGPU]):
    d = "/kaggle/working/smoke/%s/card%d" % (TAG, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", str(s), "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
works = []
for i, ((d, _), rc) in enumerate(zip(smoke, codes)):
    found = glob.glob(d + "/*.json")
    print("card %d: exit %d, result %s" % (i, rc, bool(found)))
    if found:
        r = json.load(open(found[0]))
        print("   seed %d: %.0f s, compile %s" % (r["seed"], r["walltime_sec"], r.get("compile_check")))
        works.append(i)
    else:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert works or not TODO, "the smoke run failed on every card"
PARALLEL = len(works) >= len(TODO)
print("parallel:" if PARALLEL else "only card(s) %s work: seed %s alone" % (works, TODO[:1]),
      "seeds", TODO if PARALLEL else TODO[:1])
"""),
        code("""
import glob, time
todo = TODO if PARALLEL else TODO[:1]
procs = []
for i, s in enumerate(todo):
    gpu = i if PARALLEL else works[0]
    procs.append((s, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", str(EPOCHS), "--seed", str(s),
                             "--results-dir", OUT + "/seed%d" % s,
                             "--ckpt-dir", CKPT + "/seed%d" % s,
                             "--resume", "--max-seconds", str(MAX_SECONDS)],
                            gpu, LOGS + "/seed%d.log" % s)))
    print("seed %d on card %d" % (s, gpu), flush=True)
BAD = []
for s, p in procs:
    rc = p.wait()
    print("=== seed %d: exit %d ===" % (s, rc))
    print(tail(LOGS + "/seed%d.log" % s, 2000))
    if rc != 0 or not glob.glob(OUT + "/seed%d/*.json" % s):
        BAD.append(s)
# a failed seed is reported after the summary below
print("failed seeds:", BAD or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code("""
# The finished runs against the same-stack K=64 references (T4, torch 2.11,
# two-card kernels): pairlab60 (60 epochs; the command without the arm's
# own flags), base60, and pairlab120 (continued to 120 epochs, the best
# models so far). Unpaired, by the reading fixed before the push (final
# weights): confirmed if the mean gains >= %+.4f AUC on pairlab60 and every
# run lies above both pairlab60 runs; not confirmed if it gains <= %+.4f;
# otherwise unresolved.
import json, glob, math, statistics
REFS, RULE = %r, %r
POOLED_SD, POOLED_DF = %r   # within-configuration sd of base60 and pairlab60
mean = statistics.mean
recs = {s: r for s, r in DONE.items()}
for f in sorted(glob.glob(OUT + "/seed*/*.json")):
    r = json.load(open(f)); recs[r["seed"]] = r
done = []
for s in sorted(recs):
    r = recs[s]; t = r["test_metrics"]; env = r["env"]; h = r["history"]; vl = [e["val_loss"] for e in h]
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    ref = {x["seed"]: x for x in REFS["pairlab60"]}.get(s)
    print("seed %%d: %%d epochs (history %%d), %%.2f h cumulative, %%d params | compile %%s | batch order %%s pairlab60's"
          %% (s, r["epochs_run"], len(h), r["walltime_sec"] / 3600, r["n_params"], r.get("compile_check"),
             "same as" if ref and ref["order"] == r["batch_order"]["epoch1"] else "DIFFERENT from"))
    print("  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f (lowest %%.5f at epoch %%d)  val - train %%+.5f"
          %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), vl[-1],
             min(vl), h[vl.index(min(vl))]["epoch"], vl[-1] - h[-1]["train_loss"]))
    if r.get("test_metrics_best_val"):
        tb = r["test_metrics_best_val"]
        print("  best-val weights (epoch %%d): AUC %%.5f, rej@0.3 %%.1f"
              %% (r["best_val_epoch"], tb["test_auc"], tb["bg_rej_30"]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the session cap again" %% (r["epochs_run"], EPOCHS))
        continue
    done.append({"auc": t["test_auc"], "rej30": t["bg_rej_30"], "val_loss": vl[-1],
                 "gap": vl[-1] - h[-1]["train_loss"]})
for name, rows in REFS.items():
    print("%%-11s AUC %%s | rej@0.3 %%s | final val loss %%s | val - train %%s"
          %% (name, " ".join("%%.5f" %% x["auc"] for x in rows),
             " ".join("%%.1f" %% x["rej30"] for x in rows),
             " ".join("%%.5f" %% x["val_loss"] for x in rows),
             " ".join("%%+.5f" %% x["gap"] for x in rows)))

def p_two_sided(t, df):
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        return float("nan")

if done:
    a = [x["auc"] for x in done]
    for name, rows in REFS.items():
        b = [x["auc"] for x in rows]
        line = "%%s (%%d runs) - %%s: dAUC %%+.5f, d rej@0.3 %%+.1f, d final val loss %%+.5f" %% (
            TAG, len(done), name, mean(a) - mean(b),
            mean(x["rej30"] for x in done) - mean(x["rej30"] for x in rows),
            mean(x["val_loss"] for x in done) - mean(x["val_loss"] for x in rows))
        t_p = (mean(a) - mean(b)) / (POOLED_SD * math.sqrt(1 / len(a) + 1 / len(b)))
        line += " | against the pooled sd %%.5f: t %%.2f, df %%d, p %%.3f" %% (POOLED_SD, t_p, POOLED_DF, p_two_sided(t_p, POOLED_DF))
        print(line)
    ref = [x["auc"] for x in REFS["pairlab60"]]
    d = mean(a) - mean(ref)
    if len(done) < len(SEEDS):
        verdict = "no verdict: %%d of %%d runs complete" %% (len(done), len(SEEDS))
    elif d >= RULE[0] and min(a) > max(ref):
        verdict = "CONFIRMED at K=64"
    elif d <= RULE[1]:
        verdict = "NOT CONFIRMED"
    else:
        verdict = "unresolved"
    print("verdict (%%s against pairlab60, dAUC %%+.5f, runs above both pairlab60 runs: %%d of %%d): %%s"
          %% (TAG, d, sum(x > max(ref) for x in a), len(a), verdict))
assert not BAD, "seeds failed (exit code or no result): %%s" %% BAD
""" % (K64_CONFIRM_RULE[0], K64_CONFIRM_RULE[1], refs, K64_CONFIRM_RULE, pooled)),
    ]
    stem = "k64_%s_finish_seeds%s" % (tag, seeds_s)
    name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / name)
    slug = "k64-%s-finish-seeds%s" % (tag.replace("_", "-"), seeds_s)
    meta = {
        "id": "nsanya/so3c-" + slug,
        "title": "SO3C K64 %s finish seeds%s" % (tag, seeds_s),
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128", src_kernel],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


# Window 2026-10-31: the third seed of the K=64 pair-latent confirmation,
# which read unresolved by 0.00002 on two seeds (fa8ecec). One two-card
# kernel runs the pair latent at seed 2 on one card and its reference
# (--pair-lab) at seed 2 on the other, both with float64 pair features (the
# user, 2026-10-31): float32 seed 1 fell back to eager twice, and float64 kept
# the pair latent compiled on the probe (8269496) with no resolved AUC change.
# Both seed-2 runs use it, so the arm and the reference pools each hold two
# float32 runs and one float64 run, and the seed-2 pair differs only by the
# pair latent. (tag, flags, seed); flags are added to the K=64 headline
# command.
K64_SEED2_NAME = "seed2fp64"
K64_SEED2_JOBS = [
    ("pairlatlab60fp64", ["--keep-best", "--pair-lab", "--pair-latent", "8", "--pair-lab-fp64"], 2),
    ("pairlab60fp64", ["--keep-best", "--pair-lab", "--pair-lab-fp64"], 2),
]


def _k64_base_common() -> list:
    """The K=64 headline command (build_k64_dual's), without any arm flags."""
    return ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8", "--compile"]


_JOBS_CONFIG = """
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
NAME, JOBS, EPOCHS, MAX_SECONDS = %r, %r, %d, 39000
OUT = "/kaggle/working/results_k64"
CKPT = "/kaggle/working/checkpoints"
LOGS = "/kaggle/working/logs/" + NAME
BASE = %r
BASE[BASE.index("DATA")] = DATA
pathlib.Path(LOGS).mkdir(parents=True, exist_ok=True)
for tag, _, _ in JOBS:
    pathlib.Path(OUT, tag).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUT, tag, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
"""

_JOBS_SMOKE = """
# Smoke on both cards at once, each card with its own job's flags and seed:
# one epoch on 20k jets. Each must write a result.
import glob, json
smoke = []
for i, (tag, flags, s) in enumerate(TODO[:NGPU]):
    d = "/kaggle/working/smoke/%s/card%d" % (NAME, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE + flags +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", str(s), "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
works = []
for i, ((d, _), rc) in enumerate(zip(smoke, codes)):
    found = glob.glob(d + "/*.json")
    print("card %d: exit %d, result %s" % (i, rc, bool(found)))
    if found:
        r = json.load(open(found[0]))
        print("   %s seed %d: %.0f s, compile %s" % (TODO[i][0], r["seed"], r["walltime_sec"], r.get("compile_check")))
        works.append(i)
    else:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert works or not TODO, "the smoke run failed on every card"
PARALLEL = len(works) >= len(TODO)
print("parallel:" if PARALLEL else "only card(s) %s work: %s alone" % (works, TODO[:1]),
      [j[0] for j in (TODO if PARALLEL else TODO[:1])])
"""

_JOBS_RUN = """
import glob, time
todo = TODO if PARALLEL else TODO[:1]
procs = []
for i, (tag, flags, s) in enumerate(todo):
    gpu = i if PARALLEL else works[0]
    procs.append((tag, s, launch(["benchmarks.run_top_tagging"] + BASE + flags +
                                 ["--epochs", str(EPOCHS), "--seed", str(s),
                                  "--results-dir", OUT + "/%s/seed%d" % (tag, s),
                                  "--ckpt-dir", CKPT + "/%s/seed%d" % (tag, s),
                                  "--resume", "--max-seconds", str(MAX_SECONDS)],
                                 gpu, LOGS + "/%s_seed%d.log" % (tag, s))))
    print("%s seed %d on card %d" % (tag, s, gpu), flush=True)
BAD = []
for tag, s, p in procs:
    rc = p.wait()
    print("=== %s seed %d: exit %d ===" % (tag, s, rc))
    print(tail(LOGS + "/%s_seed%d.log" % (tag, s), 2000))
    if rc != 0 or not glob.glob(OUT + "/%s/seed%d/*.json" % (tag, s)):
        BAD.append((tag, s))
print("failed:", BAD or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""

_JOBS_SUMMARY = """
# The K=64 pair-latent reading over three seeds: the committed float32 seeds 0
# and 1 of pairlatlab60 and pairlab60 (separate kernels) and this kernel's
# float64 seed-2 pair. Fixed before the push (K64_CONFIRM_RULE): confirmed if
# the arm's mean gains >= %+.4f AUC on the reference and every arm run lies
# above every reference run; not confirmed if it gains <= %+.4f; otherwise
# unresolved. The seed-2 pair, started together, differs only by the pair
# latent.
import json, glob, math, statistics
REFS, RULE = %r, %r
POOLED_SD, POOLED_DF = %r   # within-configuration sd of base60 and pairlab60
mean = statistics.mean
recs = {k: v for k, v in DONE.items()}
for tag, _, s in JOBS:
    for f in glob.glob(OUT + "/%%s/seed%%d/*.json" %% (tag, s)):
        recs[(tag, s)] = json.load(open(f))
done = {}
for (tag, s), r in sorted(recs.items()):
    t = r["test_metrics"]; env = r["env"]; h = r["history"]; vl = [e["val_loss"] for e in h]
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    print("%%s seed %%d: %%d epochs (history %%d), %%.2f h cumulative, %%d params | kwargs %%s | compile %%s | batch order %%s"
          %% (tag, s, r["epochs_run"], len(h), r["walltime_sec"] / 3600, r["n_params"], r["so3c_kwargs"],
             r.get("compile_check"), r["batch_order"]["epoch1"][:12]))
    print("  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f (lowest %%.5f at epoch %%d)  val - train %%+.5f"
          %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), vl[-1], min(vl),
             h[vl.index(min(vl))]["epoch"], vl[-1] - h[-1]["train_loss"]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the session cap" %% (r["epochs_run"], EPOCHS))
        continue
    done[tag] = t["test_auc"]

def p_two_sided(t, df):
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        return float("nan")

arm_tag, ref_tag = JOBS[0][0], JOBS[1][0]
if arm_tag in done and ref_tag in done:
    a = REFS["pairlatlab60"] + [done[arm_tag]]
    b = REFS["pairlab60"] + [done[ref_tag]]
    d = mean(a) - mean(b)
    tp = d / (POOLED_SD * math.sqrt(1 / len(a) + 1 / len(b)))
    print("seed-2 pair (float64 features, this kernel): pair latent %%+.5f" %% (done[arm_tag] - done[ref_tag]))
    print("pair latent %%s mean %%.5f | reference %%s mean %%.5f | d %%+.5f | against the pooled sd %%.5f: t %%.2f, df %%d, p %%.3f"
          %% (" ".join("%%.5f" %% x for x in a), mean(a), " ".join("%%.5f" %% x for x in b), mean(b), d,
             POOLED_SD, tp, POOLED_DF, p_two_sided(tp, POOLED_DF)))
    if d >= RULE[0] and min(a) > max(b):
        verdict = "CONFIRMED at K=64"
    elif d <= RULE[1]:
        verdict = "NOT CONFIRMED"
    else:
        verdict = "unresolved"
    print("verdict (three seeds, arm runs above every reference run: %%d of %%d): %%s"
          %% (sum(x > max(b) for x in a), len(a), verdict))
else:
    print("no verdict yet: complete runs", sorted(done))
assert not BAD, "jobs failed: %%s" %% BAD
"""


def _seed2_summary() -> str:
    refs = _k64_references()
    pooled = _pooled_sd([[x["auc"] for x in refs["pairlab60"]], [x["auc"] for x in refs["base60"]]])
    arm = [json.loads(f.read_text())["test_metrics"]["test_auc"]
           for f in sorted((ROOT / "results_k64" / "pairlatlab60").glob("seed*/*.json"))]
    assert len(arm) == 2, arm
    r = {"pairlatlab60": arm, "pairlab60": [x["auc"] for x in refs["pairlab60"]]}
    return _JOBS_SUMMARY % (K64_CONFIRM_RULE[0], K64_CONFIRM_RULE[1], r, K64_CONFIRM_RULE, pooled)


def build_k64_jobs_dual(blob: str, name: str, jobs: list, epochs: int = 60) -> str:
    """Two different K=64 runs in one kernel, one per T4: (tag, flags, seed)
    each, flags added to the headline command. The smoke runs each card with
    its own job; if only one card passes, the first job runs alone. Results
    land in results_k64/<tag>/seed<s>. A run that would overrun stops on
    --max-seconds with a checkpoint for build_k64_jobs_finish."""
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    cells = [
        md("""
# K=64, two cards, one job each: %s

%s, %d epochs, the K=64 headline command plus each job's flags. T4 with
torch 2.11.0+cu128 pinned and asserted. A run that does not fit the session
stops with a checkpoint for the finish kernel.
""" % (name, "; ".join("%s seed %d (%s)" % (tg, s, " ".join(f)) for tg, f, s in jobs), epochs)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code(_JOBS_CONFIG % (name, [list(j) for j in jobs], epochs, _k64_base_common(), bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
TODO, DONE = [tuple(j) for j in JOBS], {}
"""),
        code(_JOBS_SMOKE),
        code(_JOBS_RUN),
        code(_seed2_summary()),
    ]
    stem = "k64_%s" % name
    nb_name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / nb_name)
    slug = "k64-%s" % name.replace("_", "-")
    meta = {
        "id": "nsanya/so3c-" + slug, "title": "SO3C K64 %s" % name,
        "code_file": nb_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return nb_name


def build_k64_jobs_finish(blob: str, name: str, jobs: list, epochs: int = 60) -> str:
    """Finish build_k64_jobs_dual's kernel: mount its output, check each
    job's last checkpoint against the result JSON it wrote, read a job that
    finished from its mounted result, and resume the others with the same
    command and horizon. Push only after the first kernel has stopped."""
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    src_kernel = "nsanya/so3c-k64-%s" % name.replace("_", "-")
    cells = [
        md("""
# Finish, two cards: K=64 %s

Resumes %s from the last checkpoints in its mounted output, with the same
command and horizon; jobs that already finished are read from the mount. T4
with torch 2.11.0+cu128 pinned and asserted.
""" % (name, src_kernel)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code(_JOBS_CONFIG % (name, [list(j) for j in jobs], epochs, _k64_base_common(), bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# The first kernel's last checkpoints, from its mounted output. Each must
# agree with the result JSON that kernel wrote (epoch, batch order) and goes
# where --resume looks; the result JSON does not, so the run is not skipped.
import glob, json, pathlib, shutil, torch
TODO, DONE = [], {}
for tag, flags, s in JOBS:
    pat = "/kaggle/input/**/%s/" + tag + "/seed%d/*seed%d.%s"
    cks = glob.glob(pat % ("checkpoints", s, s, "pt"), recursive=True)
    js = glob.glob(pat % ("results_k64", s, s, "json"), recursive=True)
    rec = json.load(open(js[0])) if len(js) == 1 else None
    if len(cks) != 1:
        print("%s seed %d: %d mounted checkpoints, not finished here" % (tag, s, len(cks)))
        continue
    st = torch.load(cks[0], map_location="cpu", weights_only=False)
    print("%s seed %d: checkpoint at epoch %d (history %d), first kernel's result at epoch %s"
          % (tag, s, st["epoch"], len(st["history"]), rec["epochs_run"] if rec else None))
    assert len(st["history"]) == st["epoch"], (len(st["history"]), st["epoch"])
    if rec is not None:
        assert rec["epochs_run"] == st["epoch"], (rec["epochs_run"], st["epoch"])
        assert rec["batch_order"]["epoch1"] == st["batch_order_epoch1"], (tag, s)
    if st["epoch"] >= EPOCHS:
        assert rec is not None, "%s seed %d finished but its result is not mounted" % (tag, s)
        DONE[(tag, s)] = rec
        continue
    ck = pathlib.Path(CKPT, tag, "seed%d" % s, pathlib.Path(cks[0]).name)
    ck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cks[0], ck)
    TODO.append((tag, flags, s))
    del st
assert TODO or DONE, "nothing mounted: attach the output of " + @SRC@
print("to finish:", [(t, s) for t, _, s in TODO], "| finished in the first kernel:", sorted(DONE))
""".replace("@SRC@", repr(src_kernel))),
        code(_JOBS_SMOKE),
        code(_JOBS_RUN),
        code(_seed2_summary()),
    ]
    stem = "k64_%s_finish" % name
    nb_name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / nb_name)
    slug = "k64-%s-finish" % name.replace("_", "-")
    meta = {
        "id": "nsanya/so3c-" + slug, "title": "SO3C K64 %s finish" % name,
        "code_file": nb_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128", src_kernel],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return nb_name


# Window 2026-10-31: both first kernels stopped on the session cap (width at
# epoch 45 on both seeds, the pair latent's seed 2 at epoch 47; its reference
# finished). Separate finish kernels would leave a card idle for ~3 h and need
# ~7.3 GPU-h against ~7.35 h left of the week, so one kernel resumes all three
# as a queue: card 0 takes the pair latent first (the priority), then width
# seed 1; card 1 takes width seed 0. ~6.9 h. (tag, flags, seed) in priority
# order; a job whose checkpoint already reached 60 epochs is read from its
# mounted result. A job started late gets --max-seconds from what is left of
# BUDGET, so a shortfall stops it cleanly with a checkpoint.
K64_FINISH_QUEUE_NAME = "finish1031"
K64_FINISH_QUEUE_SOURCES = ["nsanya/so3c-k64-seed2fp64", "nsanya/so3c-k64-w64s24lab60-seeds01"]
_W64 = ["--keep-best", "--pair-lab", "--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24"]
K64_FINISH_QUEUE_JOBS = [
    ("pairlatlab60fp64", ["--keep-best", "--pair-lab", "--pair-latent", "8", "--pair-lab-fp64"], 2),
    ("w64s24lab60", _W64, 0),
    ("w64s24lab60", _W64, 1),
    ("pairlab60fp64", ["--keep-best", "--pair-lab", "--pair-lab-fp64"], 2),
]
K64_FINISH_QUEUE_BUDGET_H = 7.05


_FINISH_QUEUE_SUMMARY = """
# Both K=64 readings of the window, by the rule fixed before the push
# (K64_CONFIRM_RULE): confirmed if the arm's mean gains >= %+.4f AUC on its
# reference and every arm run lies above every reference run; not confirmed
# if it gains <= %+.4f; otherwise unresolved. A float is a committed run's
# test AUC; [tag, seed] a run of this window.
import json, glob, math, statistics
READINGS, RULE = %r, %r
POOLED_SD, POOLED_DF = %r   # within-configuration sd of base60 and pairlab60
mean = statistics.mean
recs = dict(DONE)
for tag, _, s in JOBS:
    for f in glob.glob(OUT + "/%%s/seed%%d/*.json" %% (tag, s)):
        recs[(tag, s)] = json.load(open(f))
for (tag, s), r in sorted(recs.items()):
    t = r["test_metrics"]; env = r["env"]; h = r["history"]; vl = [e["val_loss"] for e in h]
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    print("%%s seed %%d: %%d epochs (history %%d), %%.2f h cumulative, %%d params | kwargs %%s | compile %%s | batch order %%s"
          %% (tag, s, r["epochs_run"], len(h), r["walltime_sec"] / 3600, r["n_params"], r["so3c_kwargs"],
             (r.get("compile_check") or {}).get("used"), r["batch_order"]["epoch1"][:12]))
    print("  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f (lowest %%.5f at epoch %%d)  val - train %%+.5f"
          %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), vl[-1], min(vl),
             h[vl.index(min(vl))]["epoch"], vl[-1] - h[-1]["train_loss"]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the budget; resume it next window" %% (r["epochs_run"], EPOCHS))

def value(item):
    if isinstance(item, float):
        return item
    r = recs.get(tuple(item))
    return r["test_metrics"]["test_auc"] if r and r["epochs_run"] >= EPOCHS else None

def p_two_sided(t, df):
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        return float("nan")

for name, arm, ref in READINGS:
    a, b = [value(x) for x in arm], [value(x) for x in ref]
    if None in a or None in b:
        print("%%s: no verdict, runs incomplete" %% name)
        continue
    d = mean(a) - mean(b)
    tp = d / (POOLED_SD * math.sqrt(1 / len(a) + 1 / len(b)))
    if d >= RULE[0] and min(a) > max(b):
        verdict = "CONFIRMED at K=64"
    elif d <= RULE[1]:
        verdict = "NOT CONFIRMED"
    else:
        verdict = "unresolved"
    print("%%s: arm %%s mean %%.5f | reference %%s mean %%.5f | d %%+.5f | against the pooled sd %%.5f: t %%.2f, df %%d, p %%.3f | arm runs above every reference run: %%d of %%d -> %%s"
          %% (name, " ".join("%%.5f" %% x for x in a), mean(a), " ".join("%%.5f" %% x for x in b), mean(b), d,
             POOLED_SD, tp, POOLED_DF, p_two_sided(tp, POOLED_DF), sum(x > max(b) for x in a), len(a), verdict))
pl, rf = value(["pairlatlab60fp64", 2]), value(["pairlab60fp64", 2])
if pl is not None and rf is not None:
    print("seed-2 pair (float64 features, started together): pair latent %%+.5f" %% (pl - rf))
assert not BAD, "jobs failed: %%s" %% BAD
"""


def _finish_queue_readings() -> tuple:
    refs = _k64_references()
    pooled = _pooled_sd([[x["auc"] for x in refs["pairlab60"]], [x["auc"] for x in refs["base60"]]])
    arm = [json.loads(f.read_text())["test_metrics"]["test_auc"]
           for f in sorted((ROOT / "results_k64" / "pairlatlab60").glob("seed*/*.json"))]
    ref = [x["auc"] for x in refs["pairlab60"]]
    assert len(arm) == 2 and len(ref) == 2, (arm, ref)
    readings = [
        ("pair latent at K=64, three seeds", arm + [["pairlatlab60fp64", 2]], ref + [["pairlab60fp64", 2]]),
        ("width at K=64", [["w64s24lab60", 0], ["w64s24lab60", 1]], ref),
    ]
    return readings, pooled


def build_k64_finish_queue(blob: str, name: str, jobs: list, sources: list,
                           budget_h: float, epochs: int = 60) -> str:
    """Resume several K=64 runs that stopped on the session cap, from the
    mounted outputs of their first kernels, drained by the two cards as a
    queue in the order given. Each resume is checked against the result JSON
    its first kernel wrote (epoch, batch order); a job that already finished
    is read from the mount. A job gets --max-seconds from what is left of
    budget_h, less a margin for one epoch and the final evaluation. No smoke:
    the same code ran in the first kernels. The summary prints the window's
    readings by K64_CONFIRM_RULE."""
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    readings, pooled = _finish_queue_readings()
    cells = [
        md("""
# Finish queue, two cards: %s

Resumes, in this order on whichever card is free: %s. Mounts %s. T4 with
torch 2.11.0+cu128 pinned and asserted. Kernel budget %.2f h.
""" % (name, "; ".join("%s seed %d" % (tg, s) for tg, _, s in jobs), ", ".join(sources), budget_h)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
NAME, JOBS, EPOCHS, BUDGET = %r, %r, %d, %r * 3600
MARGIN = 0.40 * 3600          # one epoch's overrun and the final evaluation
OUT = "/kaggle/working/results_k64"
CKPT = "/kaggle/working/checkpoints"
LOGS = "/kaggle/working/logs/" + NAME
BASE = %r
BASE[BASE.index("DATA")] = DATA
pathlib.Path(LOGS).mkdir(parents=True, exist_ok=True)
for tag, _, _ in JOBS:
    pathlib.Path(OUT, tag).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUT, tag, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r, "| budget %%.2f h" %% (BUDGET / 3600))
""" % (name, [list(j) for j in jobs], epochs, budget_h, _k64_base_common(), bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# The first kernels' last checkpoints, from their mounted outputs. Each must
# agree with the result JSON its kernel wrote (epoch, batch order) and goes
# where --resume looks; the result JSON does not, so the run is not skipped.
import glob, json, pathlib, shutil, torch
TODO, DONE = [], {}
for tag, flags, s in JOBS:
    pat = "/kaggle/input/**/%s/" + tag + "/seed%d/*seed%d.%s"
    cks = glob.glob(pat % ("checkpoints", s, s, "pt"), recursive=True)
    js = glob.glob(pat % ("results_k64", s, s, "json"), recursive=True)
    assert len(cks) == 1 and len(js) == 1, (tag, s, cks, js)
    st = torch.load(cks[0], map_location="cpu", weights_only=False)
    rec = json.load(open(js[0]))
    print("%s seed %d: checkpoint at epoch %d (history %d), first kernel's result at epoch %d"
          % (tag, s, st["epoch"], len(st["history"]), rec["epochs_run"]))
    assert len(st["history"]) == st["epoch"] == rec["epochs_run"], (tag, s)
    assert rec["batch_order"]["epoch1"] == st["batch_order_epoch1"], (tag, s)
    if st["epoch"] >= EPOCHS:
        DONE[(tag, s)] = rec
        continue
    ck = pathlib.Path(CKPT, tag, "seed%d" % s, pathlib.Path(cks[0]).name)
    ck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cks[0], ck)
    TODO.append((tag, flags, s))
    del st
print("queue:", [(t, s) for t, _, s in TODO], "| finished in the first kernels:", sorted(DONE))
"""),
        code("""
# The queue: each free card takes the next job, with --max-seconds from what
# is left of the budget; no job starts with less than half an hour left.
import glob, time
todo = list(TODO)
running = {}                      # card -> (tag, seed, process)
BAD, SKIPPED = [], []
while todo or running:
    for c in range(NGPU):
        if c in running:
            tag, s, p = running[c]
            if p.poll() is None:
                continue
            ok = p.returncode == 0 and glob.glob(OUT + "/%s/seed%d/*.json" % (tag, s))
            print("%7.2f h  card %d: %s seed %d %s" % ((time.time() - T0) / 3600, c, tag, s,
                                                       "done" if ok else "FAILED, exit %d" % p.returncode), flush=True)
            if not ok:
                BAD.append((tag, s)); print(tail(LOGS + "/%s_seed%d.log" % (tag, s), 1500))
            del running[c]
        if todo and c not in running:
            left = int(BUDGET - (time.time() - T0) - MARGIN)
            if left < 1800:
                SKIPPED += [(t, s) for t, _, s in todo]; todo = []
                print("budget: the remaining jobs are skipped:", SKIPPED); continue
            tag, flags, s = todo.pop(0)
            running[c] = (tag, s, launch(["benchmarks.run_top_tagging"] + BASE + flags +
                                         ["--epochs", str(EPOCHS), "--seed", str(s),
                                          "--results-dir", OUT + "/%s/seed%d" % (tag, s),
                                          "--ckpt-dir", CKPT + "/%s/seed%d" % (tag, s),
                                          "--resume", "--max-seconds", str(left)],
                                         c, LOGS + "/%s_seed%d.log" % (tag, s)))
            print("%7.2f h  card %d: start %s seed %d (max %.2f h)" % ((time.time() - T0) / 3600, c, tag, s, left / 3600), flush=True)
    time.sleep(20)
print("failed:", BAD or "none", "| skipped:", SKIPPED or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code(_FINISH_QUEUE_SUMMARY % (K64_CONFIRM_RULE[0], K64_CONFIRM_RULE[1], readings, K64_CONFIRM_RULE, pooled)),
    ]
    nb_name = "kaggle_k64_%s.ipynb" % name
    _finalise(cells, NB_DIR / nb_name)
    slug = "k64-%s" % name.replace("_", "-")
    meta = {
        "id": "nsanya/so3c-" + slug, "title": "SO3C K64 %s" % name,
        "code_file": nb_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"] + list(sources),
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return nb_name


def build_cuda_diag(blob: str) -> None:
    """Why the on-card golden guard failed: float64 on CUDA differed from
    the CPU fixture by ~4e-8 (case 'base'), where the CPU matches it to
    1e-14 and the cases amplify a 1e-15 perturbation only tenfold.
    benchmarks/diag_cuda_golden.py separates platform, device, old-vs-new
    implementation, primitive ops and submodules. No dataset; minutes.
    """
    cells = [
        md("""
# CUDA float64 vs the golden fixture

Separates platform (this container's CPU vs the fixture), device (CPU vs
GPU here), implementation (pre-rewrite 16302b1 vs current, same device),
primitive ops and submodules.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "benchmarks.diag_cuda_golden"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-3000:])
open("/kaggle/working/diag_cuda_golden.txt", "w").write(r.stdout + chr(10) + r.stderr)
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_cuda_diag.ipynb")


# Seed replicates on the probe protocol: (tag, extra flags). The gap screen
# put every candidate within 0.0004 AUC of the base on one seed each, and
# this protocol's spread was unmeasured.
PROBE_SEED_RUNS = [
    ("base", [], (0, 1, 2)),
    ("pair8", ["--pair-latent", "8"], (0, 1, 2)),
    # the vector channel supplies <p_a, p_b> directly, which the bivector
    # lift only exposes through combinations; it is information, not
    # capacity, so the short protocol can test it
    ("vector", ["--vector-channel"], (0, 1, 2)),
    # The edge width, the one capacity axis the sweep never moved: every
    # pairwise interaction passes through act_hidden tanh units.
    ("w32", ["--act-hidden", "32"], (0, 1)),
    ("w64s24", ["--act-hidden", "64", "--msg-dim", "24",
                "--scalar-dim", "24"], (0, 1)),
    ("pair16w32", ["--pair-latent", "16", "--act-hidden", "32"], (0, 1)),
]


def build_probe_seeds(blob: str, tag: str, flags: list[str],
                      seeds: tuple[int, ...] = (0, 1, 2)) -> str:
    """Repeat one probe-protocol configuration over several seeds.

    The gap screen put seven candidates within 0.0004 AUC of the base on a
    single seed each, and this protocol's seed spread had never been
    measured: the +-0.00014 figure comes from K=64 runs on the full training
    set. Without it the screen cannot separate "no candidate helps" from
    "the noise swamps everything". These runs measure it directly -- same
    protocol, same configuration, different seeds -- and the same treatment
    of a candidate makes the comparison paired.

    Protocol as in the screen: K=32, canonical splits, 400k training jets,
    20 epochs, batch 256, float32, beams + channels 8. The forward pass is
    the rewritten one with --compile, so all rows here share it; the
    screen's own rows were run eagerly on the old code, which is why the
    base is repeated here rather than taken from the screen.
    """
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--max-train-samples", "400000", "--epochs", "20",
            "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
            "--n-constituents", "32", "--eval-chunk-size", "2048",
            "--models", "so3c_message_set", "--rounds", "3", "--beams",
            "--channels", "8", "--compile"] + list(flags)
    cells = [
        md("""
# Seed spread on the probe protocol: %s

Same configuration, seeds %s, K=32 / 400k jets / 20 epochs. The screen's
reference numbers on this protocol (old forward pass, eager, seed 0) are
0.98128 AUC and 834.4 rejection for the base, and 0.98124 / 827.5 for the
pair latent.
""" % (tag, ", ".join(str(s) for s in seeds))),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS = %r, %r
OUT = "/kaggle/working/results_probe_seeds/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
print("data:", DATA)
""" % (tag, list(seeds), base)),
        code("""
# On-card guard: the forward pass must be the published function here too.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1200:], g.stderr[-800:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
import glob
for seed in SEEDS:
    print("=== seed %d ===" % seed, flush=True)
    run(["benchmarks.run_top_tagging"] + BASE +
        ["--seed", str(seed), "--results-dir", OUT + "/seed%d" % seed,
         "--ckpt-dir", "/kaggle/working/checkpoints/%s_seed%d" % (TAG, seed),
         "--resume", "--max-seconds", "30000"])
    assert glob.glob(OUT + "/seed%d/*.json" % seed), "seed %d wrote no result" % seed
"""),
        code("""
import json, glob, statistics
aucs, rejs = [], []
print("%-8s%9s%10s%10s%9s" % ("seed", "AUC", "rej@0.3", "rej@0.5", "hours"))
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); t = r["test_metrics"]
        aucs.append(t["test_auc"]); rejs.append(t["bg_rej_30"])
        print("%-8d%9.5f%10.1f%10.1f%9.2f"
              % (seed, t["test_auc"], t["bg_rej_30"],
                 t.get("bg_rej_50", float("nan")), r["walltime_sec"] / 3600))
if len(aucs) > 1:
    print("mean %.5f sd %.5f AUC | mean %.1f sd %.1f rejection"
          % (statistics.mean(aucs), statistics.stdev(aucs),
             statistics.mean(rejs), statistics.stdev(rejs)))
"""),
    ]
    name = "kaggle_probe_seeds_%s.ipynb" % tag
    _finalise(cells, NB_DIR / name)
    meta = {
        "id": "nsanya/so3c-probe-seeds-" + tag.replace("_", "-"),
        "title": "SO3C probe seeds %s" % tag,
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-probe-seeds-%s.json" % tag.replace("_", "-"))
     ).write_text(json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


# Seed pairing compares runs that saw the same batch order, which needs the
# same torch build: CUDA randperm is not specified to be stable across
# versions. Kaggle's T4 image moved from torch 2.10 to 2.11 between windows,
# so the screens below pin the build, refuse any card but a T4, and every
# result JSON records both (tabular_runner "env").
GPU_SETUP_T4_PINNED = """
import subprocess, sys, time
WANT = "2.11.0+cu128"
name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print("device:", name)
assert "T4" in name, "paired with T4 runs only; got %r -- stopping before any GPU time is spent" % name

def torch_version():
    return subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                          capture_output=True, text=True).stdout.strip()

v = torch_version()
if v != WANT:
    print("torch", v, "- installing", WANT)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.11.0",
                        "--index-url", "https://download.pytorch.org/whl/cu128"],
                       capture_output=True, text=True)
    print("pip rc", r.returncode, "in %.0f s" % (time.perf_counter() - t0))
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit("could not install torch " + WANT)
    v = torch_version()
assert v == WANT, "torch %s, expected %s" % (v, WANT)
print("torch", v, "on", name)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "torchdiffeq"], check=True)
"""

# The conditioning screen (so3c_notes/gap_generic_ml.md sections 3.2 and 8.7):
# a fresh base control and the two switches, three seeds each, on the probe
# protocol, all on torch 2.11 so the arms pair seed by seed with the control.
# Budgets are per kernel, in hours, and sum to 4.75 h of the 4.85 h left this
# window; measured kernel time for the base on 2.10 was 1.44 h.
PROBE_SCREEN_RUNS = [
    ("base", [], (0, 1, 2), 1.55),
    ("edgenorm", ["--edge-norm"], (0, 1, 2), 1.62),
    ("logknee", ["--log-knee"], (0, 1, 2), 1.58),
]

# The representation screen (plan_2026-10-10 section 3, R1-R3): one switch
# per arm, seeds 0-2 on the probe protocol, paired with the 10-05 control
# above (results_message/probe_screen/base, the same T4 / torch 2.11.0+cu128
# stack) if the pairing check at the start of R1 passes. The last field puts
# that check in the kernel. Budgets are kernel caps in hours, not estimates:
# the control's three seeds trained for 1.31 h in all; R3 measured 1.5x per
# step on the CPU (C5) and has not been timed on the card.
REPR_SCREEN_RUNS = [
    ("beame002", ["--beam-energy", "0.02"], (0, 1, 2), 1.75, True),
    ("labscalars", ["--lab-scalars"], (0, 1, 2), 1.75, False),
    ("pairlab", ["--pair-lab"], (0, 1, 2), 2.4, False),
]
SCREEN_CONTROL_DIR = "results_message/probe_screen/base"

# R1's pairing check failed (max |diff| 4.3e-3 in the first two epochs, as
# large as the gap between seeds), so the arms are read against a fresh
# control on the same stack, plan section 3. Its seed 0 repeats the check's
# rerun of control seed 0 with the same default-path code, which measures
# whether training on this stack reproduces itself.
FRESH_CONTROL_RUNS = [
    ("base1010", [], (0, 1, 2), 1.6),
]
REPR_CHECK_DIR = "results_message/repr_screen/beame002/pairing_check"


def _check_rerun() -> dict:
    """R1's pairing-check rerun of control seed 0: its losses and batch order."""
    f = next((ROOT / REPR_CHECK_DIR / "session2").glob("*seed0.json"))
    r = json.loads(f.read_text())
    assert r["env"]["torch"] == "2.11.0+cu128" and "T4" in r["env"]["gpu"], r["env"]
    assert r["epochs_run"] == 2 and not r["so3c_kwargs"].get("beam_energy"), f
    return {"history": [(h["train_loss"], h["val_loss"]) for h in r["history"]],
            "order": r["batch_order"]["epoch1"]}


REPRODUCIBILITY = """
# Same-stack reproducibility. The pairing check in probe-screen-beame002 ran
# control seed 0 for two epochs on this stack (T4, torch 2.11.0+cu128) with the
# same default-path code, and did not reproduce the 2026-10-05 control (max
# |diff| 4.3e-3). Seed 0 here repeats that rerun. The same batch order with
# epoch-1 losses within 1e-5 means this stack reproduces itself and the 10-05
# control differed for another reason; the same batch order with different
# losses means training here is not bitwise reproducible, so a shared seed
# shares the initialisation and the batch order but not the trajectory. Epoch 2
# of the check came from a resumed session, so epoch 1 decides.
import json, glob
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); env = r["env"]; bo = r["batch_order"]
        print("seed %d: torch %s on %s | batch order epoch 1 %s, chain %s"
              % (seed, env["torch"], env.get("gpu"), bo["epoch1"], bo["chain"]))
f0 = sorted(glob.glob(OUT + "/seed0/*.json"))
if f0:
    r = json.load(open(f0[0])); h = r["history"]
    same = r["batch_order"]["epoch1"] == CHECK_ORDER
    diff = [max(abs(h[i]["train_loss"] - CHECK_RERUN[i][0]),
                abs(h[i]["val_loss"] - CHECK_RERUN[i][1])) for i in range(2)]
    print("seed 0 against the R1 check's rerun: batch order %s; max |diff| epoch 1 %.1e, epoch 2 %.1e"
          % ("the same" if same else "DIFFERENT", diff[0], diff[1]))
    if not same:
        print("  different batch order under one seed on one stack: the fingerprint or the RNG is broken")
    elif diff[0] <= 1e-5:
        print("  this stack reproduces itself; the 10-05 control differed for another reason")
    else:
        print("  same batch order, different trajectory: training here is not bitwise reproducible")
"""


def _screen_control() -> dict:
    """The 10-05 control: every seed's test result, seed 0's first two epochs."""
    test, history = {}, None
    for f in sorted((ROOT / SCREEN_CONTROL_DIR).glob("seed*/*.json")):
        r = json.loads(f.read_text())
        assert r["env"]["torch"] == "2.11.0+cu128" and "T4" in r["env"]["gpu"], f
        assert r["epochs_run"] == 20 and not r["so3c_kwargs"].get("beam_energy"), f
        test[r["seed"]] = (r["test_metrics"]["test_auc"], r["test_metrics"]["bg_rej_30"])
        if r["seed"] == 0:
            history = [(h["train_loss"], h["val_loss"]) for h in r["history"][:2]]
    assert history and sorted(test) == [0, 1, 2], (history, sorted(test))
    return {"test": test, "history": history}


# Pairing check (plan_2026-10-10 section 3): control seed 0 rerun for the first
# two epochs of its 20-epoch schedule, compared with the stored history.
PAIRING_CHECK = """
# Pairing check (plan_2026-10-10 section 3). Rerun control seed 0 for the first
# two epochs of its 20-epoch schedule and compare with the stored history.
# The control's three seeds differ there by 3.7e-3 to 6.1e-2 in the largest of
# these four losses, so agreement within 1e-5 means the same initial weights
# and the same batch order. --max-seconds 0 stops after every
# epoch, so this is two one-epoch sessions: the second resumes from the
# checkpoint, RNG state included, and keeps the control's cosine horizon. Each
# session writes its own results directory, because with --resume the runner
# skips a run whose result file already exists.
import json, glob, pathlib
CHECK = OUT + "/pairing_check"
for session in (1, 2):
    assert run(["benchmarks.run_top_tagging"] + CONTROL_BASE +
               ["--seed", "0", "--results-dir", CHECK + "/session%d" % session,
                "--ckpt-dir", "/kaggle/working/checkpoints/pairing_check",
                "--resume", "--max-seconds", "0"]), "pairing check session %d failed" % session
found = glob.glob(CHECK + "/session2/*seed0.json")
assert found, "the pairing check wrote no result"
chk = json.load(open(found[0]))
hist = chk["history"]
assert len(hist) == 2, "pairing check ran %d epochs, expected 2" % len(hist)
assert chk["env"]["torch"] == "2.11.0+cu128", chk["env"]
worst = 0.0
print("%-7s%-7s%12s%12s%10s" % ("epoch", "loss", "stored", "rerun", "|diff|"))
for i, pair in enumerate(CONTROL_HISTORY):
    for kind, want in zip(("train", "val"), pair):
        got = hist[i][kind + "_loss"]
        worst = max(worst, abs(got - want))
        print("%-7d%-7s%12.7f%12.7f%10.1e" % (i + 1, kind, want, got, abs(got - want)))
PAIRED = worst <= 1e-5
print("pairing check %s: max |diff| %.1e (pass if <= 1e-5)"
      % ("PASSED" if PAIRED else "FAILED", worst))
if not PAIRED:
    print("  the arms cannot be read against the 10-05 control: push a fresh control")
CHECK_ORDER = chk["batch_order"]
print("batch order, seed 0: epoch 1 %s, chain %s" % (CHECK_ORDER["epoch1"], CHECK_ORDER["chain"]))
pathlib.Path(CHECK, "verdict.json").write_text(json.dumps(
    {"paired": PAIRED, "max_abs_diff": worst, "batch_order": CHECK_ORDER,
     "stored": CONTROL_HISTORY,
     "rerun": [(e["train_loss"], e["val_loss"]) for e in hist]}, indent=2))
"""

# Summary of a screen arm against the stored control, seed by seed, with the
# plan_2026-10-10 section 3 decision rule.
PAIRED_SUMMARY = """
import json, glob, statistics
print("%-6s%9s%10s%9s%8s%8s%7s%14s" % ("seed", "AUC", "dAUC", "rej@0.3", "drej",
                                        "epochs", "hours", "batch order"))
deltas, drejs, orders = {}, {}, {}
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); t = r["test_metrics"]; env = r.get("env", {})
        c_auc, c_rej = CONTROL_TEST[seed]
        orders[seed] = r["batch_order"]["epoch1"]
        print("%-6d%9.5f%+10.5f%9.1f%+8.1f%8d%7.2f%14s"
              % (seed, t["test_auc"], t["test_auc"] - c_auc, t["bg_rej_30"],
                 t["bg_rej_30"] - c_rej, r["epochs_run"], r["walltime_sec"] / 3600,
                 orders[seed][:12]))
        assert env.get("torch") == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
        if r["epochs_run"] < 20:
            print("  INCOMPLETE: stopped at epoch %d on the kernel budget" % r["epochs_run"])
        else:
            deltas[seed] = t["test_auc"] - c_auc
            drejs[seed] = t["bg_rej_30"] - c_rej
if "CHECK_ORDER" in globals() and 0 in orders:
    same = orders[0] == CHECK_ORDER["epoch1"]
    print("seed 0 batch order %s the pairing check's" % ("equals" if same else "DIFFERS FROM"))
d = [deltas[s] for s in SEEDS if s in deltas]
if len(d) < len(SEEDS):
    print("%d of %d seeds complete: no verdict" % (len(d), len(SEEDS)))
else:
    m = statistics.mean(d)
    if m >= 0.00015 and min(d) > 0:
        verdict = "PROMOTE to the K=32 long schedule"
    elif abs(m) < 0.0001 or min(d) < 0 < max(d):
        verdict = "CLOSE at the probe level"
    elif max(d) < 0:
        verdict = "UNRESOLVED by the rule, but every seed is negative: not a candidate"
    else:
        verdict = "UNRESOLVED"
    print("paired dAUC %s: mean %+.5f sd %.5f | drej mean %+.1f"
          % (" ".join("%+.5f" % x for x in d), m, statistics.stdev(d),
             statistics.mean(drejs.values())))
    print("verdict: %s" % verdict)
    print("(valid only if the pairing check in probe-screen-beame002 passed;"
          " rejection is reported, never decides)")
"""


def build_probe_screen(blob: str, tag: str, flags: list[str],
                       seeds: tuple[int, ...], budget_h: float,
                       pairing_check: bool = False,
                       control: dict | None = None,
                       reproduce: dict | None = None) -> str:
    """One arm of a probe-protocol screen: seeds 0-2.

    Protocol as build_probe_seeds (K=32, canonical splits, 400k training
    jets, 20 epochs, batch 256, float32, beams + channels 8, --compile), on a
    T4 with torch 2.11.0+cu128 pinned and asserted. Each seed gets what is
    left of the kernel budget as --max-seconds, so a slow arm stops with a
    checkpoint instead of overrunning the week's quota; a seed is skipped when
    too little budget is left for it to finish.

    With `control` (from _screen_control) the arm is read against that stored
    control seed by seed and the summary applies the plan's decision rule;
    `pairing_check` first reruns the control's seed 0 for two epochs.
    `reproduce` (from _check_rerun) adds a last cell that compares seed 0
    with that rerun.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    common = ["--cache-dir", "DATA", "--representation", "constituents",
              "--canonical-splits", "--normalize", "global",
              "--max-train-samples", "400000", "--epochs", "20",
              "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
              "--n-constituents", "32", "--eval-chunk-size", "2048",
              "--models", "so3c_message_set", "--rounds", "3", "--beams",
              "--channels", "8", "--compile"]
    base = common + list(flags)
    if pairing_check and control is None:
        raise ValueError("the pairing check needs the stored control")
    if control is None:
        pairing = "the base control of the" + chr(10) + "same screen"   # as built on 10-05
        heading = "Conditioning screen"
    else:
        pairing = "the 2026-10-05 control (`%s`)" % SCREEN_CONTROL_DIR
        heading = "Representation screen"
        if pairing_check:
            pairing += (". The kernel first reruns control seed 0 for two "
                        "epochs: the pairing check")
    cells = [
        md("""
# %s, probe protocol: %s

Flags beyond the base: `%s`. Seeds %s, K=32 / 400k jets / 20 epochs, T4 with
torch 2.11.0+cu128 pinned, paired seed by seed with %s. Kernel budget %.2f h.
""" % (heading, tag, " ".join(flags) or "(none: the control)",
           ", ".join(str(s) for s in seeds), pairing, budget_h)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(GPU_SETUP_T4_PINNED),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS = %r, %r
OUT = "/kaggle/working/results_probe_screen/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
BUDGET = %r * 3600           # this kernel's share of the week's GPU quota
EVAL_MARGIN = 240            # final test evaluation and writing results
MIN_SEED = 1500              # a seed needs ~0.45 h; less than this cannot finish
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, list(seeds), base, budget_h, bundle_sha, bundle_sha)),
    ]
    if control is not None:
        cells.append(code("""
# The control this arm pairs with (%s): per seed, (test AUC,
# rejection at 30%%); seed 0's (train, val) loss in epochs 1 and 2; and its
# command, which is BASE without this arm's flags.
CONTROL_TEST = %r
CONTROL_HISTORY = %r
CONTROL_BASE = %r
CONTROL_BASE[CONTROL_BASE.index("DATA")] = DATA
""" % (SCREEN_CONTROL_DIR, control["test"], control["history"], common)))
    cells.append(code("""
# On-card guard: the forward pass must be the published function here too.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1200:], g.stderr[-800:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""))
    if pairing_check:
        cells.append(code(PAIRING_CHECK))
    cells += [
        code("""
import glob, time
for seed in SEEDS:
    left = BUDGET - (time.time() - T0) - EVAL_MARGIN
    print("=== seed %d: %.2f h of kernel budget left ===" % (seed, left / 3600), flush=True)
    if left < MIN_SEED:
        print("not enough budget left; seed %d skipped" % seed)
        continue
    run(["benchmarks.run_top_tagging"] + BASE +
        ["--seed", str(seed), "--results-dir", OUT + "/seed%d" % seed,
         "--ckpt-dir", "/kaggle/working/checkpoints/%s_seed%d" % (TAG, seed),
         "--resume", "--max-seconds", str(int(left))])
    assert glob.glob(OUT + "/seed%d/*.json" % seed), "seed %d wrote no result" % seed
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
    ]
    if control is not None:
        cells.append(code(PAIRED_SUMMARY))
    else:
        cells.append(code("""
import json, glob, statistics
aucs, rejs = [], []
print("%-6s%9s%10s%10s%8s%8s%20s%10s" % ("seed", "AUC", "rej@0.3", "rej@0.5", "epochs", "hours", "torch", "gpu"))
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); t = r["test_metrics"]; env = r.get("env", {})
        print("%-6d%9.5f%10.1f%10.1f%8d%8.2f%20s%10s"
              % (seed, t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["epochs_run"], r["walltime_sec"] / 3600, env.get("torch"),
                 env.get("gpu", "").replace("Tesla ", "")))
        assert env.get("torch") == "2.11.0+cu128", "result recorded torch %r" % env.get("torch")
        if r["epochs_run"] < 20:
            print("  INCOMPLETE: stopped at epoch %d on the kernel budget" % r["epochs_run"])
        else:
            aucs.append(t["test_auc"]); rejs.append(t["bg_rej_30"])
if len(aucs) > 1:
    print("complete seeds: mean %.5f sd %.5f AUC | mean %.1f sd %.1f rejection"
          % (statistics.mean(aucs), statistics.stdev(aucs),
             statistics.mean(rejs), statistics.stdev(rejs)))
"""))
    if reproduce is not None:
        cells.append(code(
            "CHECK_RERUN = %r" % reproduce["history"] + chr(10)
            + "CHECK_ORDER = %r" % reproduce["order"] + chr(10) + REPRODUCIBILITY))
    name = "kaggle_probe_screen_%s.ipynb" % tag
    _finalise(cells, NB_DIR / name)
    meta = {
        "id": "nsanya/so3c-probe-screen-" + tag,
        "title": "SO3C probe screen %s" % tag,
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-probe-screen-%s.json" % tag)
     ).write_text(json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


# 2026-10-18: which of the four lab-frame pair features carries the gain.
# Probe protocol (the 10-05 control's command plus the flags), three seeds per
# arm, read unpaired against the pooled probe controls and the full
# --pair-lab probe runs. A fourth wave runs one full --pair-lab run and one
# control at a new seed, so both reference levels are also measured in this
# kernel (earlier kernels' levels differ by up to ~0.0001, about 15% of the
# gain). All eight runs go through one two-card kernel as a queue: each card
# takes the next run when it is free (four waves of ~0.5-0.6 h, about 2.3
# GPU-h in all).
PROBE_QUEUES = [
    # the two arms interleaved, so a budget cut in the last wave leaves both
    # arms with the same number of seeds
    ("ablation-pairlab", [
        (arm, ["--pair-lab", "--pair-lab-set", subset], s)
        for s in (0, 1, 2)
        for arm, subset in (("pairlab_mass", "mass"), ("pairlab_frame", "frame"))
    ] + [
        ("pairlab_full", ["--pair-lab"], 3),     # anchors: last, so a budget
        ("control", [], 3),                      # cut drops them first
    ], 2.6),
    # 2026-10-25: the float64 pair features (--pair-lab-fp64) against float32,
    # both in this kernel at the same seeds, each pair started together on the
    # two cards. Last, four one-epoch pair-latent runs on 20k jets read the
    # first-batch compile check with and without the switch (at K=32 long
    # seed 1 fell back to eager at d logit 1.3e-3).
    ("fp64-pairlab", [
        (arm, flags, s)
        for s in (0, 1, 2)
        for arm, flags in (("pairlab_fp64", ["--pair-lab", "--pair-lab-fp64"]),
                           ("pairlab_f32", ["--pair-lab"]))
    ] + [
        (arm, flags + ["--pair-latent", "8", "--epochs", "1", "--max-train-samples", "20000"], s)
        for s in (0, 1)
        for arm, flags in (("compile_pairlat_fp64", ["--pair-lab", "--pair-lab-fp64"]),
                           ("compile_pairlat_f32", ["--pair-lab"]))
    ], 2.6),
]

# Which summary cell a probe queue prints (default "ablation").
PROBE_QUEUE_SUMMARY = {"fp64-pairlab": "fp64"}

# Reading of the float64 screen, fixed before the push: d = mean(pairlab_fp64)
# - mean(pairlab_f32), three runs each in the same kernel. "pays" if
# d >= +0.00015 and Welch p < 0.05 (carry the switch into later runs, with
# same-stack references); "hurts" if d <= -0.00015 and p < 0.05 (keep
# float32); otherwise no resolved difference (carry it only where it saves
# compute, by the compile check). +0.00015 is the probe protocol's promotion
# line (plan_2026-10-10 section 3); the probe's run-to-run sd is ~0.0001.
FP64_RULE = (0.00015, 0.05)


def _probe_references() -> dict:
    """Test AUC and rejection of the probe runs the ablation is read against."""
    refs = {}
    for name, d in (("control", ["results_message/probe_screen/base", "results_message/repr_screen/base1010"]),
                    ("pairlab", ["results_message/repr_screen/pairlab"])):
        rows = []
        for sub in d:
            for f in sorted((ROOT / sub).glob("seed*/*.json")):
                r = json.loads(f.read_text())
                assert r["env"]["torch"] == "2.11.0+cu128" and r["epochs_run"] == 20, f
                rows.append((r["test_metrics"]["test_auc"], r["test_metrics"]["bg_rej_30"]))
        refs[name] = rows
    assert len(refs["control"]) == 6 and len(refs["pairlab"]) == 3, {k: len(v) for k, v in refs.items()}
    return refs


def _probe_references_fp64() -> dict:
    """The earlier float32 --pair-lab probe runs (10-10 and the ablation's
    seed-3 anchor) and the probe controls (10-05, base1010, the ablation's
    control): test AUC of each."""
    refs = {}
    for name, d in (("pairlab", ["results_message/repr_screen/pairlab",
                                 "results_message/repr_screen/ablation/pairlab_full"]),
                    ("control", ["results_message/probe_screen/base", "results_message/repr_screen/base1010",
                                 "results_message/repr_screen/ablation/control"])):
        rows = []
        for sub in d:
            for f in sorted((ROOT / sub).glob("seed*/*.json")):
                r = json.loads(f.read_text())
                assert r["env"]["torch"] == "2.11.0+cu128" and r["epochs_run"] == 20, f
                rows.append(r["test_metrics"]["test_auc"])
        refs[name] = rows
    assert len(refs["pairlab"]) == 4 and len(refs["control"]) == 7, {k: len(v) for k, v in refs.items()}
    return refs


_FP64_SUMMARY = """
# The float64 pair features against float32, both from this kernel, three
# seeds each, read by the rule fixed before the push: d = mean(fp64) -
# mean(f32); "pays" if d >= +%.5f and Welch p < %.2f, "hurts" if
# d <= -%.5f and p < %.2f, otherwise no resolved difference. Beside it, the
# earlier float32 --pair-lab probe runs pooled with this kernel's, the probe
# controls, and the compile check of every run.
import json, glob, statistics as st, math
from scipy import stats
REFS, RULE = %r, %r
def load(arm, epochs=20):
    rs = [json.load(open(f)) for f in sorted(glob.glob(OUT + "/%%s/seed*/*.json" %% arm))]
    for r in rs:
        t = r["test_metrics"]; env = r["env"]; cc = r.get("compile_check") or {}
        assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
        print("%%-21s seed %%d: AUC %%.5f  rej@0.3 %%.1f  %%d epochs  %%.2f h  compile %%s d_logit %%.1e  kwargs %%s"
              %% (arm, r["seed"], t["test_auc"], t["bg_rej_30"], r["epochs_run"], r["walltime_sec"] / 3600,
                 cc.get("used"), cc.get("d_logit", float("nan")), r["so3c_kwargs"]))
    return {r["seed"]: r["test_metrics"]["test_auc"] for r in rs if r["epochs_run"] == epochs}
def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return st.mean(a) - st.mean(b), t, df, 2 * stats.t.sf(abs(t), df)
fp, f32 = load("pairlab_fp64"), load("pairlab_f32")
for arm in ("compile_pairlat_fp64", "compile_pairlat_f32"):
    load(arm, epochs=1)
a, b = list(fp.values()), list(f32.values())
if len(a) >= 2 and len(b) >= 2:
    d, t, df, p = welch(a, b)
    if d >= RULE[0] and p < RULE[1]:
        verdict = "PAYS: carry --pair-lab-fp64 into later runs, with same-stack references"
    elif d <= -RULE[0] and p < RULE[1]:
        verdict = "HURTS: keep float32"
    else:
        verdict = "no resolved difference"
    print("fp64 %%d runs %%.5f +- %%.5f | f32 here %%d runs %%.5f +- %%.5f | d %%+.5f (Welch t %%.2f, df %%.1f, p %%.3f) -> %%s"
          %% (len(a), st.mean(a), st.stdev(a), len(b), st.mean(b), st.stdev(b), d, t, df, p, verdict))
    same = sorted(set(fp) & set(f32))
    print("same-seed differences (both runs started together):", " ".join("s%%d %%+.5f" %% (s, fp[s] - f32[s]) for s in same))
    pooled = REFS["pairlab"] + b
    d2, t2, df2, p2 = welch(a, pooled)
    print("against all %%d float32 --pair-lab probe runs (%%.5f +- %%.5f): %%+.5f (t %%.2f, df %%.1f, p %%.3f)"
          %% (len(pooled), st.mean(pooled), st.stdev(pooled), d2, t2, df2, p2))
    for name, rows in (("f32 here", b), ("fp64", a)):
        d3, t3, df3, p3 = welch(rows, REFS["control"])
        print("%%-8s against the %%d probe controls (%%.5f): %%+.5f (t %%.1f, p %%.3f)"
              %% (name, len(REFS["control"]), st.mean(REFS["control"]), d3, t3, p3))
    print("float32 --pair-lab here against the earlier float32 runs: %%+.5f (kernel offset)"
          %% (st.mean(b) - st.mean(REFS["pairlab"])))
else:
    print("fewer than two complete runs per arm: no reading")
assert not BAD, "jobs failed: %%s" %% BAD
"""


def build_probe_queue(blob: str, tag: str, jobs: list, budget_h: float,
                      summary: str = "ablation") -> str:
    """Probe-protocol runs (arm, flags, seed) drained by two T4s as a queue.

    The command is the 10-05 control's (build_probe_screen's `common`) plus
    each job's flags, on the pinned T4 / torch 2.11 setup. A smoke test runs
    on both cards at once first; the queue then hands the next job to
    whichever card is free, and stops handing out jobs once less than one
    run's time is left of `budget_h`. With summary="ablation" the summary
    reads each subset arm unpaired against the pooled probe controls and the
    full --pair-lab runs; with "fp64" it reads the float64 pair features
    against float32 by FP64_RULE.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    common = ["--cache-dir", "DATA", "--representation", "constituents",
              "--canonical-splits", "--normalize", "global",
              "--max-train-samples", "400000", "--epochs", "20",
              "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
              "--n-constituents", "32", "--eval-chunk-size", "2048",
              "--models", "so3c_message_set", "--rounds", "3", "--beams",
              "--channels", "8", "--compile"]
    assert summary in ("ablation", "fp64"), summary
    refs = _probe_references()
    cells = [
        md("""
# Probe-protocol queue, two cards: %s

%d runs (%s) on the probe protocol (K=32, 400k training jets, 20 epochs),
the 2026-10-05 control's command plus each run's flags, drained by two T4s
as a queue. T4 with torch 2.11.0+cu128 pinned and asserted. Kernel budget %.1f h.
""" % (tag, len(jobs), ", ".join(sorted({j[0] for j in jobs})), budget_h)),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, JOBS, BUDGET = %r, %r, %r * 3600
OUT = "/kaggle/working/results_probe_queue"
CKPT = "/kaggle/working/checkpoints"
LOGS = "/kaggle/working/logs/" + TAG
RUN_TIME = 0.55 * 3600       # one probe run end to end, two cards busy (0.49-0.60 h for pair_lab)
BASE = %r
BASE[BASE.index("DATA")] = DATA
for d in (OUT, LOGS):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r, "|", len(JOBS), "jobs")
""" % (tag, [list(j) for j in jobs], budget_h, common, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# Smoke on both cards at once with the first job's flags: one epoch on 20k
# jets per card. Each card must write a result to be given jobs.
import glob
smoke = []
for i in range(NGPU):
    d = "/kaggle/working/smoke/%s/card%d" % (TAG, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE + JOBS[0][1] +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", "0", "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
CARDS = [i for i, ((d, _), rc) in enumerate(zip(smoke, codes)) if glob.glob(d + "/*.json")]
for i, rc in enumerate(codes):
    print("card %d: exit %d, result %s" % (i, rc, i in CARDS))
    if i not in CARDS:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert CARDS, "the smoke run failed on every card"
print("cards in use:", CARDS)
"""),
        code("""
# The queue: each free card takes the next job; no job starts once less than
# one run's time is left of the budget.
import glob, time
todo = list(JOBS)
running = {}                     # card -> (arm, seed, process)
BAD, SKIPPED = [], []
while todo or running:
    for c in CARDS:
        if c in running:
            arm, s, p = running[c]
            if p.poll() is None and time.time() - T0 > BUDGET + 0.1 * 3600:
                p.terminate(); p.wait()          # hung past the budget: its checkpoint is kept
                print("card %d: %s seed %d terminated past the budget" % (c, arm, s), flush=True)
            if p.poll() is None:
                continue
            ok = p.returncode == 0 and glob.glob(OUT + "/%s/seed%d/*.json" % (arm, s))
            print("%7.2f h  card %d: %s seed %d %s" % ((time.time() - T0) / 3600, c, arm, s,
                                                       "done" if ok else "FAILED, exit %d" % p.returncode), flush=True)
            if not ok:
                BAD.append((arm, s)); print(tail(LOGS + "/%s_seed%d.log" % (arm, s), 1500))
            del running[c]
        if todo and c not in running:
            if time.time() - T0 + RUN_TIME > BUDGET:
                SKIPPED += [(a, s) for a, _, s in todo]; todo = []
                print("budget: the remaining jobs are skipped:", SKIPPED); continue
            arm, flags, s = todo.pop(0)
            cap = int(min(RUN_TIME * 1.5, BUDGET - (time.time() - T0) - 600))
            running[c] = (arm, s, launch(["benchmarks.run_top_tagging"] + BASE + flags +
                                         ["--seed", str(s), "--results-dir", OUT + "/%s/seed%d" % (arm, s),
                                          "--ckpt-dir", CKPT + "/%s_seed%d" % (arm, s), "--resume",
                                          "--max-seconds", str(cap)],
                                         c, LOGS + "/%s_seed%d.log" % (arm, s)))
            print("%7.2f h  card %d: start %s seed %d" % ((time.time() - T0) / 3600, c, arm, s), flush=True)
    time.sleep(20)
print("failed:", BAD or "none", "| skipped:", SKIPPED or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code("""
# Each subset arm unpaired against the pooled probe controls (10-05, base1010
# and this kernel's control) and the full --pair-lab runs (10-10 and this
# kernel's), with the reading rule fixed before the push:
#   f = (arm - control) / (full - control), the fraction of the gain recovered;
#   "carries the gain" if f >= 0.7 and the arm beats the control at p < 0.05,
#   "carries none" if f <= 0.3 and the arm is below the full set at p < 0.05,
#   otherwise unresolved.
import json, glob, statistics as st, math
from scipy import stats
REFS = %r
def load(arm):
    rs = [json.load(open(f)) for f in sorted(glob.glob(OUT + "/%%s/seed*/*.json" %% arm))]
    for r in rs:
        t = r["test_metrics"]; env = r["env"]
        assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
        print("%%-14s seed %%d: AUC %%.5f  rej@0.3 %%.1f  %%d epochs  %%.2f h  kwargs %%s"
              %% (arm, r["seed"], t["test_auc"], t["bg_rej_30"], r["epochs_run"], r["walltime_sec"] / 3600, r["so3c_kwargs"]))
    return [r["test_metrics"]["test_auc"] for r in rs if r["epochs_run"] == 20]
def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return st.mean(a) - st.mean(b), t, df, 2 * stats.t.sf(abs(t), df)
new_full, new_ctrl = load("pairlab_full"), load("control")
old_full = [r[0] for r in REFS["pairlab"]]; old_ctrl = [r[0] for r in REFS["control"]]
for name, new, old in (("full pair_lab", new_full, old_full), ("control", new_ctrl, old_ctrl)):
    if new:
        print("anchor %%s here %%s against the earlier runs' %%.5f +- %%.5f: offset %%+.5f"
              %% (name, " ".join("%%.5f" %% x for x in new), st.mean(old), st.stdev(old), st.mean(new) - st.mean(old)))
full, ctrl = old_full + new_full, old_ctrl + new_ctrl
print("references: control %%d runs %%.5f +- %%.5f | full pair_lab %%d runs %%.5f +- %%.5f"
      %% (len(ctrl), st.mean(ctrl), st.stdev(ctrl), len(full), st.mean(full), st.stdev(full)))
for arm in ("pairlab_mass", "pairlab_frame"):
    a = load(arm)
    if len(a) < 2:
        print("  %%s: %%d complete runs, no reading" %% (arm, len(a))); continue
    d, t, df, pc = welch(a, ctrl); d2, t2, df2, pf = welch(a, full)
    A, C, F = st.mean(a), st.mean(ctrl), st.mean(full); D = F - C
    vA, vC, vF = st.variance(a) / len(a), st.variance(ctrl) / len(ctrl), st.variance(full) / len(full)
    f = (A - C) / D
    se = math.sqrt(vA / D ** 2 + vF * ((A - C) / D ** 2) ** 2 + vC * ((A - F) / D ** 2) ** 2)
    if f >= 0.7 and d > 0 and pc < 0.05:
        verdict = "carries the gain"
    elif f <= 0.3 and d2 < 0 and pf < 0.05:
        verdict = "carries none of the gain"
    else:
        verdict = "unresolved"
    print("  %%s: mean %%.5f +- %%.5f | vs control %%+.5f (t %%.1f, df %%.1f, p %%.3f) | vs full %%+.5f (t %%.1f, df %%.1f, p %%.3f) | fraction of the gain %%.2f +- %%.2f -> %%s"
          %% (arm, A, st.stdev(a), d, t, df, pc, d2, t2, df2, pf, f, se, verdict))
assert not BAD, "jobs failed: %%s" %% BAD
""" % refs),
    ]
    if summary == "fp64":
        cells[-1] = code(_FP64_SUMMARY % (FP64_RULE[0], FP64_RULE[1], FP64_RULE[0], FP64_RULE[1],
                                          _probe_references_fp64(), FP64_RULE))
    name = "kaggle_probe_queue_%s.ipynb" % tag.replace("-", "_")
    _finalise(cells, NB_DIR / name)
    meta = {
        "id": "nsanya/so3c-probe-queue-" + tag,
        "title": "SO3C probe queue %s" % tag,
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-probe-queue-%s.json" % tag)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


def build_width_profile(blob: str) -> None:
    """Time the capacity candidates on the card before a window commits to them.

    The long-schedule screen (K=32, full training set, 60 epochs) runs each
    arm for hours per seed, and CPU cost ratios have misled three times, so
    every candidate is timed on the GPU first: compiled, K=32 and K=64,
    synthetic jets (cost depends only on shapes). No dataset; ~20 minutes.
    """
    configs = ["base", "w32", "w64s24", "w128s32", "c16", "pair8", "pair16w32"]
    cells = [
        md("""
# Width and capacity candidates: compiled cost on the card

Per-epoch estimates at K=32 (batch 256, eval chunk 2048) and K=64 (eval
chunk 1024), with `--compile`, for the configurations a long-schedule screen
would compare.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
CONFIGS = %r
for k, chunk in (("32", "2048"), ("64", "1024")):
    args = ["benchmarks.profile_gpu", "--out", "/kaggle/working/width_k" + k,
            "--k", k, "--batch-size", "256", "--eval-chunk-size", chunk,
            "--compile", "--no-profile", "--steps", "15"]
    for c in CONFIGS:
        args += ["--config", c]
    ok = run(args)
    assert ok, "profile at K=%%s failed" %% k
""" % configs),
    ]
    _finalise(cells, NB_DIR / "kaggle_width_profile.ipynb")
    meta = {"id": "nsanya/so3c-width-profile", "title": "SO3C width profile",
            "code_file": "kaggle_width_profile.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": []}
    (NB_DIR / "kernel-metadata-width-profile.json").write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")



# 2026-10-25: the compiled per-epoch cost, on the pinned T4 / torch 2.11
# setup, of what the next windows would run: the K=64 arms and their
# combination, and so3c_knn_set against dense pair features at K=128.
# (K, eval chunk, configs); synthetic jets, cost depends only on shapes.
KNN_PROFILE = [
    (64, 1024, ["pairlab", "pairlat8lab", "w64s24lab", "pairlat8w64s24lab", "knn16"]),
    (128, 512, ["pairlab", "knn8", "knn16", "knn32"]),
]


def build_knn_profile(blob: str) -> str:
    """Time so3c_knn_set and the K=64 candidates on the card before a window
    commits to them: CPU cost ratios have misled three times (STATUS,
    infrastructure traps). No dataset; ~30 minutes."""
    cells = [
        md("""
# Compiled cost on the card: K=64 candidates and so3c_knn_set at K=128

Per-epoch estimates (batch 256) with `--compile`, synthetic jets, T4 with
torch 2.11.0+cu128 pinned: %s.
""" % "; ".join("K=%d: %s" % (k, ", ".join(c)) for k, _, c in KNN_PROFILE)),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code("""
PROFILE = %r
for k, chunk, configs in PROFILE:
    args = ["benchmarks.profile_gpu", "--out", "/kaggle/working/knn_profile_k%%d" %% k,
            "--k", str(k), "--batch-size", "256", "--eval-chunk-size", str(chunk),
            "--compile", "--no-profile", "--steps", "15"]
    for c in configs:
        args += ["--config", c]
    ok = run(args)
    assert ok, "profile at K=%%d failed" %% k
""" % KNN_PROFILE),
        code("""
import json
for k, _, _ in PROFILE:
    s = json.load(open("/kaggle/working/knn_profile_k%d/summary.json" % k))
    print("K=%d on %s, torch %s" % (k, s["device"], s["torch"]))
    for r in s["rows"]:
        if "compile_error" in r:
            print("  %-18s compile error" % r["config"]); continue
        print("  %-18s %6d params | epoch ~%5.0f s (%.1f h for 60) | peak %.2f GiB%s"
              % (r["config"], r["n_params"], r["epoch_estimate_s"], r["epoch_estimate_s"] * 60 / 3600,
                 r["peak_train_gib"], r["compile_note"]))
"""),
    ]
    name = "kaggle_knn_profile.ipynb"
    _finalise(cells, NB_DIR / name)
    meta = {"id": "nsanya/so3c-knn-profile", "title": "SO3C knn profile",
            "code_file": name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": []}
    (NB_DIR / "kernel-metadata-knn-profile.json").write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name

# Long-schedule screen, window 2026-09-26: the width scaling curve,
# (tag, flags, seed), one seed per kernel so the order can adapt. Measured
# compiled cost at K=32 over 60 epochs: base ~3.0 h, w64s24 ~4.3 h,
# w128s32 ~6.4 h (results_profile/t4_width_1007eaa).
LONG_RUNS = [
    ("base", [], 0),
    ("w128s32", ["--act-hidden", "128", "--msg-dim", "32", "--scalar-dim", "32"], 0),
    ("w64s24", ["--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24"], 0),
    ("base", [], 1),
    ("w128s32", ["--act-hidden", "128", "--msg-dim", "32", "--scalar-dim", "32"], 1),
    ("w64s24", ["--act-hidden", "64", "--msg-dim", "24", "--scalar-dim", "24"], 1),
    # third base seed: the protocol's own spread decides whether the width
    # gain is real; replaces w64s24 seed 1, which would land at the quota edge
    ("base", [], 2),
    # Window 2026-10-03: regularise the wide model. Decoupled weight decay
    # 0.002 (milder than PELICAN's 0.005: over 60 cosine epochs at lr 3e-3 it
    # shrinks an untrained weight by exp(-0.002 x 426) ~ 0.43) and
    # --keep-best, at seeds 0 and 1. A shared seed gives the same
    # initialisation and the same batch order, and AdamW with zero decay is
    # Adam, so these are paired against the unregularised w128s32 seed 0
    # and 1 runs above, differing only in the decay.
    ("w128s32wd2", ["--act-hidden", "128", "--msg-dim", "32", "--scalar-dim", "32",
                    "--optimizer", "adamw", "--weight-decay", "0.002",
                    "--keep-best"], 0),
    ("w128s32wd2", ["--act-hidden", "128", "--msg-dim", "32", "--scalar-dim", "32",
                    "--optimizer", "adamw", "--weight-decay", "0.002",
                    "--keep-best"], 1),
]


# Plan_2026-10-10 phase 2: whichever way phase 1 goes, the long-schedule arms
# need a control on their own stack (the results_long_seeds bases ran on torch
# 2.10, and pairing across 2.10 -> 2.11 is not shown). Base configuration plus
# --keep-best (free: a CPU copy of the best-validation weights), seeds 0 and 1,
# T4 / torch 2.11 pinned. The 2.10 base took 3.37-3.52 h per seed.
LONG_RUNS_PINNED = [
    ("base211", ["--keep-best"], 0),
    ("base211", ["--keep-best"], 1),
    # Phase 2 if no representation arm is promoted: the pair latent, the one
    # structural candidate with a hint (+0.00012 by seed on the probe
    # protocol, unpaired t = 1.2), against base211 above. It cost 1.34x the
    # base on the probe protocol (torch 2.10, compiled): ~4.7 h per seed here.
    ("pairlat211", ["--keep-best", "--pair-latent", "8"], 0),
    ("pairlat211", ["--keep-best", "--pair-latent", "8"], 1),
    # Phase 2, the branch taken: R3 (--pair-lab) was promoted on the probe
    # protocol (+0.00055 by seed against base1010, every seed positive;
    # +0.00051 against the pooled six-run control, Welch t = 5.9). Plan
    # section 4: the arm at the long schedule on seeds 0 and 1 against
    # base211. It cost 1.05-1.13x the control per seed on the probe protocol.
    ("pairlab211", ["--keep-best", "--pair-lab"], 0),
    ("pairlab211", ["--keep-best", "--pair-lab"], 1),
]


def build_long_seeds(blob: str, tag: str, flags: list[str],
                     seeds: tuple[int, ...], part: str = "",
                     pinned: bool = False) -> str:
    """The long-schedule screen: K=32 on the FULL training set, 60 epochs.

    Why this protocol. 60 epochs at K=64 beat the 30-epoch mean by +0.00037
    AUC with a final validation loss below all three complete 30-epoch
    schedules (f12cd65): the model was undertrained, so every capacity
    candidate had been screened at 20 epochs on 400k jets, where more
    parameters cannot pay. This keeps the two things that matter for
    capacity -- all 1.211M training jets and a 60-epoch schedule -- and cuts
    only K, to 32. It is also the regime of the results_60ep evidence.

    Several multi-hour seeds share one 12 h Kaggle session, so before each
    run the notebook gives it what is left of an 11.2 h budget as
    --max-seconds, minus a margin for the final evaluation. A run that would
    overrun stops cleanly with a checkpoint and is reported as incomplete.

    With `pinned`, the T4 / torch 2.11.0+cu128 setup of the probe screens
    replaces GPU_SETUP, and a last cell asserts the stack of every result and
    prints its batch-order fingerprint, so later arms can be paired with it.
    """
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global", "--epochs", "60",
            "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
            "--n-constituents", "32", "--eval-chunk-size", "2048",
            "--models", "so3c_message_set", "--rounds", "3", "--beams",
            "--channels", "8", "--compile"] + list(flags)
    suffix = ("_" + part) if part else ""
    cells = [
        md("""
# Long-schedule screen: %s, seeds %s

K=32, all 1.211M training jets, 60 epochs, beams + channels 8%s, compiled.%s
""" % (tag, ", ".join(str(s) for s in seeds),
       (" + " + " ".join(flags)) if flags else "",
       " T4 with torch 2.11.0+cu128 pinned and asserted." if pinned else "")),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED if pinned else GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS = %r, %r
OUT = "/kaggle/working/results_long_seeds/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
BUDGET = 11.2 * 3600          # the whole session, minus a margin under 12 h
EVAL_MARGIN = 900             # final test evaluation and writing results
print("data:", DATA)
""" % (tag, list(seeds), base)),
        code("""
# On-card guard: the forward pass must be the published function here too.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1200:], g.stderr[-800:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
import glob, time
for seed in SEEDS:
    left = BUDGET - (time.time() - T0) - EVAL_MARGIN
    print("=== seed %d: %.2f h of session budget left ===" % (seed, left / 3600), flush=True)
    if left < 1800:
        print("not enough session left; seed %d skipped" % seed)
        continue
    run(["benchmarks.run_top_tagging"] + BASE +
        ["--seed", str(seed), "--results-dir", OUT + "/seed%d" % seed,
         "--ckpt-dir", "/kaggle/working/checkpoints/%s_seed%d" % (TAG, seed),
         "--resume", "--max-seconds", str(int(left))])
    assert glob.glob(OUT + "/seed%d/*.json" % seed), "seed %d wrote no result" % seed
"""),
        code("""
import json, glob, statistics
aucs, rejs = [], []
print("%-6s%9s%10s%10s%8s%9s%11s" % ("seed", "AUC", "rej@0.3", "rej@0.5", "epochs", "hours", "val_loss"))
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); t = r["test_metrics"]; h = r["history"]
        print("%-6d%9.5f%10.1f%10.1f%8d%9.2f%11.5f"
              % (seed, t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")),
                 r["epochs_run"], r["walltime_sec"] / 3600, h[-1]["val_loss"]))
        if r["epochs_run"] < 60:
            print("  INCOMPLETE: stopped at epoch %d on the session budget" % r["epochs_run"])
        else:
            aucs.append(t["test_auc"]); rejs.append(t["bg_rej_30"])
        if r.get("test_metrics_best_val"):
            tb = r["test_metrics_best_val"]
            print("  best-val weights (epoch %d, val_loss %.5f): AUC %.5f, rej@0.3 %.1f, rej@0.5 %.1f"
                  % (r["best_val_epoch"], r["best_val_loss"], tb["test_auc"],
                     tb["bg_rej_30"], tb.get("bg_rej_50", float("nan"))))
        vl = [e["val_loss"] for e in h]
        print("  val_loss: epoch 30 %.5f, 50 %.5f, last %.5f; last 5 epochs %+.5f"
              % (vl[29] if len(vl) > 29 else float("nan"),
                 vl[49] if len(vl) > 49 else float("nan"), vl[-1], vl[-1] - vl[-6]))
if len(aucs) > 1:
    print("complete seeds: mean %.5f sd %.5f AUC | mean %.1f sd %.1f rejection"
          % (statistics.mean(aucs), statistics.stdev(aucs),
             statistics.mean(rejs), statistics.stdev(rejs)))
"""),
    ]
    if pinned:
        cells.append(code("""
# The stack every result ran on, and its batch order: a later arm pairs with
# this run seed by seed only if both match.
import json, glob
for seed in SEEDS:
    for f in sorted(glob.glob(OUT + "/seed%d/*.json" % seed)):
        r = json.load(open(f)); env = r["env"]; bo = r["batch_order"]
        print("seed %d: torch %s on %s | batch order epoch 1 %s, chain %s"
              % (seed, env["torch"], env.get("gpu"), bo["epoch1"], bo["chain"]))
        assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
"""))
    name = "kaggle_long_seeds_%s%s.ipynb" % (tag, suffix)
    _finalise(cells, NB_DIR / name)
    slug = ("long-seeds-%s%s" % (tag, ("-" + part) if part else "")).replace("_", "-")
    meta = {
        "id": "nsanya/so3c-" + slug,
        "title": "SO3C long seeds %s%s" % (tag, (" " + part) if part else ""),
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


# Window 2026-10-24: two structural candidates on top of --pair-lab (all four
# features) at the K=32 long schedule, seeds 0 and 1 in one two-card kernel
# each, read unpaired against pairlab211 (the same command without the
# candidate's flags; one single-card kernel per seed, 3.30 and 3.35 h). Run
# order: the pair latent first (PELICAN's rank-2 pair state, the structural
# candidate aimed at the remaining gap; +0.00012 by seed on the probe
# protocol), then width (w64s24; at this schedule without --pair-lab, torch
# 2.10, seed 0: 0.98286 against the base's 0.98261, 4.39 h).
LONG_DUAL_RUNS = [
    ("pairlatlab211", ["--keep-best", "--pair-lab", "--pair-latent", "8"], (0, 1)),
    ("w64s24lab211", ["--keep-best", "--pair-lab", "--act-hidden", "64",
                      "--msg-dim", "24", "--scalar-dim", "24"], (0, 1)),
]

# Decision rule, fixed before the push. On the final weights, the arm's runs
# against pairlab211's two runs: it pays (confirm at K=64) if the mean gains
# at least +0.0003 AUC and every run lies above both pairlab211 runs; close if
# the mean gains nothing (<= 0); otherwise unresolved. Two runs against two,
# unpaired, at a per-run sd of ~0.0002 give the difference an SE of ~0.0002,
# so the close line sits 1.5 SE below the pays line as that sits above zero
# (a +0.0001 close line, as in plan_2026-10-10 section 2, would close a real
# +0.0003 about one time in six). Rejection, the best-validation weights and
# the validation - train gap are reported but do not decide.
LONG_DUAL_RULE = (0.0003, 0.0)


def _long_pooled_sd() -> tuple[float, int]:
    """Pooled within-configuration sd of the final test AUC at the K=32 long
    schedule, and its degrees of freedom: every results_long_seeds
    configuration with two or more complete runs."""
    import math
    ss, df = 0.0, 0
    for d in sorted((ROOT / "results_long_seeds").iterdir()):
        runs = [json.loads(f.read_text()) for f in sorted(d.glob("seed*/*.json"))]
        a = [r["test_metrics"]["test_auc"] for r in runs if r["epochs_run"] == 60]
        if len(a) > 1:
            m = sum(a) / len(a)
            ss += sum((x - m) ** 2 for x in a)
            df += len(a) - 1
    return math.sqrt(ss / df), df


def _long_references() -> dict:
    """The same-stack K=32 long runs a two-card arm is read against:
    pairlab211 and base211, seeds 0 and 1 each, final weights."""
    refs = {}
    for tag in ("pairlab211", "base211"):
        rows = []
        for f in sorted((ROOT / "results_long_seeds" / tag).glob("seed*/*.json")):
            r = json.loads(f.read_text())
            assert r["env"]["torch"] == "2.11.0+cu128" and r["epochs_run"] == 60, f
            t, h = r["test_metrics"], r["history"][-1]
            rows.append({"seed": r["seed"], "auc": t["test_auc"], "rej30": t["bg_rej_30"],
                         "val_loss": h["val_loss"], "gap": h["val_loss"] - h["train_loss"],
                         "hours": r["walltime_sec"] / 3600, "params": r["n_params"],
                         "order": r["batch_order"]["epoch1"]})
        refs[tag] = rows
    assert [[x["seed"] for x in v] for v in refs.values()] == [[0, 1], [0, 1]], refs
    return refs


def build_long_dual(blob: str, tag: str, flags: list[str],
                    seeds: tuple[int, ...] = (0, 1)) -> str:
    """Two seeds of one configuration at the K=32 long schedule, one per T4.

    The command is build_long_seeds's (K=32, all 1.211M training jets, 60
    epochs, beams + channels 8, compiled) plus ``flags``, on the pinned T4 /
    torch 2.11 setup. As in build_k64_dual, a smoke test runs one epoch on
    20k jets on both cards at once; the seeds run in parallel only if both
    cards pass it, otherwise the first seed runs alone. Both get what is
    left of an 11.2 h session budget as --max-seconds, minus a margin for
    the final evaluation; a run that would overrun stops cleanly with a
    checkpoint and is reported as incomplete. The summary reads the runs
    against pairlab211 and base211 by LONG_DUAL_RULE.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global", "--epochs", "60",
            "--device", "cuda", "--dtype", "float32", "--batch-size", "256",
            "--n-constituents", "32", "--eval-chunk-size", "2048",
            "--models", "so3c_message_set", "--rounds", "3", "--beams",
            "--channels", "8", "--compile"] + list(flags)
    refs = _long_references()
    pooled = _long_pooled_sd()
    seeds_s = "".join(str(s) for s in seeds)
    cells = [
        md("""
# Long schedule, two cards: %s, seeds %s

K=32, all 1.211M training jets, 60 epochs, beams + channels 8 + %s, compiled,
one seed per T4 in parallel. T4 with torch 2.11.0+cu128 pinned and asserted.
Read unpaired against pairlab211 (the same command without the arm's own
flags) and base211, seeds 0 and 1 each.
""" % (tag, ", ".join(str(s) for s in seeds), " ".join(flags))),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS, EPOCHS = %r, %r, 60
OUT = "/kaggle/working/results_long_seeds/" + TAG
CKPT = "/kaggle/working/checkpoints/" + TAG
LOGS = "/kaggle/working/logs/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
BUDGET = 11.2 * 3600          # the whole session, minus a margin under 12 h
EVAL_MARGIN = 900             # final test evaluation and writing results
for d in (OUT, LOGS):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, list(seeds), base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# Smoke on both cards at once, as the real runs will start, each card with
# its own seed, so both seeds' initial weights go through the first-batch
# compile check (on 20k jets' normalisation, not the full set's): one epoch
# on 20k jets per card. Each must write a result.
import glob, json, time
smoke = []
for i in range(min(NGPU, len(SEEDS))):
    d = "/kaggle/working/smoke/%s/card%d" % (TAG, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", str(SEEDS[i]), "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
works = []
for i, ((d, _), rc) in enumerate(zip(smoke, codes)):
    found = glob.glob(d + "/*.json")
    print("card %d: exit %d, result %s" % (i, rc, bool(found)))
    if found:
        r = json.load(open(found[0]))
        print("   seed %d: %.0f s, AUC %.4f, %d params, compile %s, %s"
              % (r["seed"], r["walltime_sec"], r["test_metrics"]["test_auc"], r["n_params"],
                 r.get("compile_check"), r["env"]))
        works.append(i)
    else:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert works, "the smoke run failed on every card"
PARALLEL = len(works) >= len(SEEDS)
print("parallel:" if PARALLEL else "only card(s) %s work: seed %d alone" % (works, SEEDS[0]),
      "seeds", SEEDS if PARALLEL else SEEDS[:1])
"""),
        code("""
import glob, time
todo = SEEDS if PARALLEL else SEEDS[:1]
left = int(BUDGET - (time.time() - T0) - EVAL_MARGIN)
print("%.2f h of session budget left for each run" % (left / 3600), flush=True)
assert left > 3600, "not enough session left"
procs = []
for i, s in enumerate(todo):
    gpu = i if PARALLEL else works[0]
    procs.append((s, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--seed", str(s), "--results-dir", OUT + "/seed%d" % s,
                             "--ckpt-dir", CKPT + "/seed%d" % s,
                             "--resume", "--max-seconds", str(left)],
                            gpu, LOGS + "/seed%d.log" % s)))
    print("seed %d on card %d" % (s, gpu), flush=True)
BAD = []
for s, p in procs:
    rc = p.wait()
    print("=== seed %d: exit %d ===" % (s, rc))
    print(tail(LOGS + "/seed%d.log" % s, 2000))
    if rc != 0 or not glob.glob(OUT + "/seed%d/*.json" % s):
        BAD.append(s)
# a failed seed is reported after the summary below, so the other seed's
# result is still printed
print("failed seeds:", BAD or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code("""
# Every result against the same-stack long runs: pairlab211 (this command
# without the arm's own flags) and base211, seeds 0 and 1 each. Same-seed
# runs in different kernels do not reproduce on this stack, so the reading
# is unpaired, by the rule fixed before the push (final weights): pays if
# the mean gains >= %+.4f AUC on pairlab211 and every run lies above both
# pairlab211 runs; close if it gains <= %+.4f; otherwise unresolved. The two
# arm runs share a kernel, so a per-kernel offset would move both.
import json, glob, math, statistics
REFS, RULE = %r, %r
POOLED_SD, POOLED_DF = %r   # within-configuration sd of the AUC at this schedule
mean = statistics.mean
done = []
for f in sorted(glob.glob(OUT + "/seed*/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]; env = r["env"]; bo = r["batch_order"]
    h = r["history"]; vl = [e["val_loss"] for e in h]
    print("seed %%d: %%d epochs, %%.2f h (%%.0f s/epoch), %%d params | torch %%s on %%s | compile %%s"
          %% (r["seed"], r["epochs_run"], r["walltime_sec"] / 3600,
             r["walltime_sec"] / max(r["epochs_run"], 1), r["n_params"], env["torch"], env.get("gpu"),
             r.get("compile_check")))
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    ref = {x["seed"]: x for x in REFS["pairlab211"]}.get(r["seed"])
    print("  batch order epoch 1 %%s: %%s pairlab211's at this seed"
          %% (bo["epoch1"][:12], "same as" if ref and ref["order"] == bo["epoch1"] else "DIFFERENT from"))
    print("  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f (lowest %%.5f at epoch %%d)  val - train %%+.5f"
          %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), vl[-1],
             min(vl), h[vl.index(min(vl))]["epoch"], vl[-1] - h[-1]["train_loss"]))
    if r.get("test_metrics_best_val"):
        tb = r["test_metrics_best_val"]
        print("  best-val weights (epoch %%d): AUC %%.5f, rej@0.3 %%.1f"
              %% (r["best_val_epoch"], tb["test_auc"], tb["bg_rej_30"]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): stopped on the session budget; finish before reading"
              %% (r["epochs_run"], EPOCHS))
        continue
    done.append({"auc": t["test_auc"], "rej30": t["bg_rej_30"], "val_loss": vl[-1],
                 "gap": vl[-1] - h[-1]["train_loss"]})
for name, rows in REFS.items():
    print("%%-11s AUC %%s | rej@0.3 %%s | final val loss %%s | val - train %%s | %%s h"
          %% (name, " ".join("%%.5f" %% x["auc"] for x in rows),
             " ".join("%%.1f" %% x["rej30"] for x in rows),
             " ".join("%%.5f" %% x["val_loss"] for x in rows),
             " ".join("%%+.5f" %% x["gap"] for x in rows),
             " ".join("%%.2f" %% x["hours"] for x in rows)))

def p_two_sided(t, df):
    try:
        from scipy import stats
        return 2 * stats.t.sf(abs(t), df)
    except Exception:
        return float("nan")

def welch(a, b):
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    t = (mean(a) - mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df, p_two_sided(t, df)

def pooled(a, b):
    # two runs' sample sd can be far from this schedule's spread (pairlab211's
    # two runs differ by only 0.0001), so also against the pooled sd
    t = (mean(a) - mean(b)) / (POOLED_SD * math.sqrt(1 / len(a) + 1 / len(b)))
    return t, POOLED_DF, p_two_sided(t, POOLED_DF)

if done:
    a = [x["auc"] for x in done]
    for name, rows in REFS.items():
        b = [x["auc"] for x in rows]
        line = "%%s (%%d runs) - %%s: dAUC %%+.5f, d rej@0.3 %%+.1f, d final val loss %%+.5f, d val - train %%+.5f" %% (
            TAG, len(done), name, mean(a) - mean(b),
            mean(x["rej30"] for x in done) - mean(x["rej30"] for x in rows),
            mean(x["val_loss"] for x in done) - mean(x["val_loss"] for x in rows),
            mean(x["gap"] for x in done) - mean(x["gap"] for x in rows))
        if len(a) > 1:
            line += " | Welch t %%.2f, df %%.1f, p %%.3f" %% welch(a, b)
        line += " | against the pooled sd %%.5f: t %%.2f, df %%d, p %%.3f" %% ((POOLED_SD,) + pooled(a, b))
        print(line)
    ref = [x["auc"] for x in REFS["pairlab211"]]
    d = mean(a) - mean(ref)
    if len(done) < len(SEEDS):
        verdict = "no verdict: %%d of %%d runs complete" %% (len(done), len(SEEDS))
    elif d >= RULE[0] and min(a) > max(ref):
        verdict = "PAYS: confirm at K=64"
    elif d <= RULE[1]:
        verdict = "CLOSE"
    else:
        verdict = "unresolved"
    print("verdict (%%s against pairlab211, dAUC %%+.5f, runs above both pairlab211 runs: %%d of %%d): %%s"
          %% (TAG, d, sum(x > max(ref) for x in a), len(a), verdict))
assert not BAD, "seeds failed (exit code or no result): %%s" %% BAD
""" % (LONG_DUAL_RULE[0], LONG_DUAL_RULE[1], refs, LONG_DUAL_RULE, pooled)),
    ]
    stem = "long_%s_seeds%s" % (tag, seeds_s)
    name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / name)
    slug = "long-%s-seeds%s" % (tag.replace("_", "-"), seeds_s)
    meta = {
        "id": "nsanya/so3c-" + slug,
        "title": "SO3C long %s seeds%s" % (tag, seeds_s),
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128"],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


HEADLINE60_CKPT = ("results_k64/headline60/checkpoints/"
                   "top_tagging_canonical__so3c_message_set__seed0.pt")
CONT_TAG = "headline120"
# The epoch-60 reference the continuation is judged against
# (results_k64/headline60, f12cd65): AUC, 1/eps_B at 0.3, final val loss.
CONT_REF = (0.98363, 1140.8, 0.16276)


def _k64_headline_base() -> list[str]:
    """The flags of the 60-epoch headline run, without --epochs and --seed."""
    return ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8", "--compile"]


_CONT_RUN = """
ok = run(["benchmarks.run_top_tagging"] + BASE +
         ["--epochs", "120", "--seed", "0", "--keep-best", "--resume",
          "--results-dir", OUT, "--ckpt-dir", CKPT_DIR,
          "--max-seconds", "39000"])
assert ok, "continuation run failed"
"""

_CONT_SUMMARY = """
import json, glob
REF_AUC, REF_REJ, REF_VL = %r
r = json.load(open(glob.glob(OUT + "/*seed0.json")[0]))
t, h = r["test_metrics"], r["history"]
print("epochs run %%d of 120, session-cumulative wall clock %%.2f h, env %%s"
      %% (r["epochs_run"], r["walltime_sec"] / 3600, r.get("env")))
if r["epochs_run"] < 120:
    print("INCOMPLETE: stopped on the session cap; push the finish kernel")
print("final weights:  AUC %%.5f  rej@0.3 %%.1f  rej@0.5 %%.1f  final val loss %%.5f"
      %% (t["test_auc"], t["bg_rej_30"], t.get("bg_rej_50", float("nan")), h[-1]["val_loss"]))
b = r.get("test_metrics_best_val")
if b:
    print("best-val weights (epoch %%d, val loss %%.5f; epochs 61-120 only):  AUC %%.5f  rej@0.3 %%.1f"
          %% (r["best_val_epoch"], r["best_val_loss"], b["test_auc"], b["bg_rej_30"]))
print()
print("validation curve")
for e in h:
    if e["epoch"] in (55, 60) or (e["epoch"] > 60 and e["epoch"] %% 5 == 0) or e["epoch"] == 61:
        print("  epoch %%3d  train %%.5f  val %%.5f  val_acc %%.5f"
              %% (e["epoch"], e["train_loss"], e["val_loss"], e["val_acc"]))
d_auc = t["test_auc"] - REF_AUC
print()
print("against epoch 60 of the same model (%%.5f / %%.1f / val %%.5f): dAUC %%+.5f, d rej %%+.1f, d val loss %%+.5f"
      %% (REF_AUC, REF_REJ, REF_VL, d_auc, t["bg_rej_30"] - REF_REJ, h[-1]["val_loss"] - REF_VL))
if r["epochs_run"] == 120:
    if d_auc >= 0.0003 and h[-1]["val_loss"] < REF_VL:
        print("DECISION (plan 2.4): longer training pays at K=64")
    elif d_auc <= 0.0001:
        print("DECISION (plan 2.4): training length exhausted near 60 epochs; close")
    else:
        print("DECISION (plan 2.4): unresolved; record it")
""" % (CONT_REF,)


def build_k64_continuation(blob: str) -> str:
    """60 -> 120 epochs: resume the 60-epoch K=64 headline (plan_2026-10-10 §2a).

    The epoch-60 checkpoint of results_k64/headline60 (torch 2.10, T4) is
    embedded in the notebook, so the run does not depend on the 2026-09-20
    kernel's output still existing. It resumes with --epochs 120: the
    harness re-anchors the cosine schedule on the new horizon (647dd38), so
    the learning rate restarts at 1.5e-3 -- a warm restart, as the plan
    says. --keep-best reports the best-validation weights as well; the
    checkpoint predates keep-best, so those come from epochs 61-120 only.

    Before training, the checkpoint's sha256, epoch and history are checked,
    and its CUDA RNG state (saved under torch 2.10) is tried on the card: if
    2.11 rejects it, it is dropped from the copy, since a continuation needs
    no reproducible batch order. The decision rule of plan §2 is printed at
    the end, against epoch 60 of the same model.
    """
    import base64
    import hashlib
    raw = (ROOT / HEADLINE60_CKPT).read_bytes()
    ckpt_sha = hashlib.sha256(raw).hexdigest()
    ckpt_b64 = base64.b64encode(raw).decode()
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = _k64_headline_base()
    cells = [
        md("""
# K=64 headline, 60 -> 120 epochs (continuation)

Resumes the 60-epoch K=64 headline (0.98363 AUC, 1140.8 rejection at 30%%
signal efficiency, final validation loss 0.16276) from its epoch-60
checkpoint, embedded here, to 120 epochs. The cosine schedule is re-anchored
on the 120-epoch horizon, so the learning rate restarts at 1.5e-3.
Decision rule (plan_2026-10-10 §2), on the final weights against epoch 60:
dAUC >= +0.0003 with a lower final validation loss -> longer training pays;
dAUC <= +0.0001 -> training length exhausted near 60 epochs.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG = %r
OUT = "/kaggle/working/results_k64/" + TAG
CKPT_DIR = "/kaggle/working/checkpoints/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (CONT_TAG, base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function here too.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1200:], g.stderr[-800:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
        code("""
# The embedded epoch-60 checkpoint: write it where --resume looks, verify it,
# and make sure this torch accepts its CUDA RNG state.
import base64, hashlib, pathlib, torch
CKPT_B64 = "%s"
raw = base64.b64decode(CKPT_B64)
assert hashlib.sha256(raw).hexdigest() == %r, "checkpoint corrupted in transit"
ck_path = pathlib.Path(CKPT_DIR) / "top_tagging_canonical__so3c_message_set__seed0.pt"
ck_path.parent.mkdir(parents=True, exist_ok=True)
ck_path.write_bytes(raw)
state = torch.load(ck_path, map_location="cpu", weights_only=False)
assert state["epoch"] == 60 and len(state["history"]) == 60, (state["epoch"], len(state["history"]))
assert abs(state["history"][-1]["val_loss"] - 0.16276) < 1e-5, state["history"][-1]["val_loss"]
print("checkpoint ok: epoch", state["epoch"], "| val loss %%.5f" %% state["history"][-1]["val_loss"],
      "| torch here", torch.__version__)
try:
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), 0)
    print("CUDA RNG state from torch 2.10 accepted")
except Exception as exc:
    print("CUDA RNG state rejected (%%s); dropping it from the copy" %% str(exc)[:200])
    state["cuda_rng"] = None
    torch.save(state, ck_path)
del state, raw, CKPT_B64
""" % (ckpt_b64, ckpt_sha)),
        code("""
# Smoke: one epoch on 20k jets with the real configuration (a fresh model,
# separate directories), to prove the data path and the compiled model
# before the session commits hours.
import glob
run(["benchmarks.run_top_tagging"] + BASE +
    ["--epochs", "1", "--max-train-samples", "20000", "--seed", "0",
     "--results-dir", "/kaggle/working/smoke"])
assert glob.glob("/kaggle/working/smoke/*.json"), "smoke run wrote no result"
"""),
        code(_CONT_RUN),
        code(_CONT_SUMMARY),
    ]
    name = "kaggle_k64_%s_seed0.ipynb" % CONT_TAG
    _finalise(cells, NB_DIR / name)
    k64_metadata(CONT_TAG, 0)
    return name


# Window 2026-10-17: both K=64 pair-feature seeds continued 60 -> 120 epochs in
# one two-card kernel, as the headline was (+0.00029 AUC against its own
# epoch 60). (source tag, continuation tag, flags of the source run, seeds)
K64_DUAL_CONTINUATIONS = [
    ("pairlab60", "pairlab120", ["--keep-best", "--pair-lab"], (0, 1)),
]


def build_k64_continuation_dual(blob: str, src: str, tag: str, flags: list[str],
                                seeds: tuple[int, ...]) -> str:
    """60 -> 120 epochs for two K=64 seeds of one run, one per T4.

    The epoch-60 checkpoints come from the source kernel's own output,
    mounted through kernel_sources: Kaggle refuses a kernel whose source is
    over 1 MB, and two embedded checkpoints come to 1.45 MB. Each is checked
    on the card against the sha256 of the committed copy in
    results_k64/<src>/seed<s>, epoch 60 and the final validation loss its
    result JSON recorded, and copied where --resume looks. The command is the source run's (build_k64_dual) with --epochs
    120: the cosine schedule is re-anchored on the new horizon, a warm
    restart at learning rate 1.5e-3. A smoke test runs on both cards at once
    first; if either card fails it, the first seed continues alone.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = ["--cache-dir", "DATA", "--representation", "constituents",
            "--canonical-splits", "--normalize", "global",
            "--device", "cuda", "--dtype", "float32",
            "--batch-size", "256", "--n-constituents", "64",
            "--eval-chunk-size", "1024", "--models", "so3c_message_set",
            "--rounds", "3", "--beams", "--channels", "8",
            "--compile"] + list(flags)
    ckpts = {}
    for s in seeds:
        d = ROOT / "results_k64" / src / ("seed%d" % s)
        rec = json.loads(next(d.glob("*seed%d.json" % s)).read_text())
        assert rec["epochs_run"] == 60 and rec["env"]["torch"] == "2.11.0+cu128", (s, rec["epochs_run"])
        raw = next(d.glob("*seed%d.pt" % s)).read_bytes()
        ckpts[s] = (None, hashlib.sha256(raw).hexdigest(),
                    rec["history"][-1]["val_loss"], rec["test_metrics"]["test_auc"],
                    rec["test_metrics"]["bg_rej_30"])
    src_kernel = "nsanya/so3c-k64-%s-seeds%s" % (src.replace("_", "-"),
                                                 "".join(str(s) for s in seeds))
    refs = {s: c[2:] for s, c in ckpts.items()}
    cells = [
        md("""
# K=64 %s, 60 -> 120 epochs, two cards: seeds %s

Resumes results_k64/%s (beams + channels 8 + %s, 60 epochs) from its
epoch-60 checkpoints, mounted from that kernel's output, to 120 epochs, one
seed per T4. The cosine
schedule is re-anchored on the 120-epoch horizon (learning rate restarts at
1.5e-3). T4 with torch 2.11.0+cu128 pinned and asserted.
""" % (src, ", ".join(str(s) for s in seeds), src, " ".join(flags))),
        code("""
import subprocess, time
T0 = time.time()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP_T4_PINNED),
        code(DUAL_LAUNCH),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG, SEEDS, EPOCHS, MAX_SECONDS = %r, %r, 120, 39000
OUT = "/kaggle/working/results_k64/" + TAG
CKPT = "/kaggle/working/checkpoints/" + TAG
LOGS = "/kaggle/working/logs/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
for d in (OUT, LOGS):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT, "bundle_sha256.txt").write_text(%r + chr(10))
print("data:", DATA, "| bundle", %r)
""" % (tag, list(seeds), base, bundle_sha, bundle_sha)),
        code("""
# On-card guard: the forward pass must be the published function on this
# GPU too, in float64. "1 passed" is required; a skip means no CUDA.
import subprocess, sys
g = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_so3c_models.py", "-k", "golden_cuda"],
                   cwd="/kaggle/working/repo", capture_output=True, text=True)
print(g.stdout[-1500:], g.stderr[-1500:])
assert "1 passed" in g.stdout, "golden guard did not pass on the card"
"""),
    ]
    for s in seeds:
        _, sha, vl, _, _ = ckpts[s]
        cells.append(code("""
# The epoch-60 checkpoint of seed %d, from the mounted output of %s: check it
# against the committed copy's sha256 and copy it where --resume looks (the
# experiment and model in the file name are the run's).
import glob, hashlib, pathlib, torch
found = glob.glob("/kaggle/input/**/checkpoints/%s/seed%d/*seed%d.pt", recursive=True)
assert len(found) == 1, "expected one mounted seed-%d checkpoint, found %%r" %% found
raw = pathlib.Path(found[0]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == %r, "mounted checkpoint differs from the committed one"
ck = pathlib.Path(CKPT) / "seed%d" / "top_tagging_canonical__so3c_message_set__seed%d.pt"
ck.parent.mkdir(parents=True, exist_ok=True)
ck.write_bytes(raw)
state = torch.load(ck, map_location="cpu", weights_only=False)
assert state["epoch"] == 60 and len(state["history"]) == 60, (state["epoch"], len(state["history"]))
assert abs(state["history"][-1]["val_loss"] - %r) < 1e-9, state["history"][-1]["val_loss"]
print("seed %d checkpoint ok: epoch 60, val loss %%.5f" %% state["history"][-1]["val_loss"])
del state, raw
""" % (s, src_kernel, src, s, s, s, sha, s, s, vl, s)))
    cells += [
        code("""
# Smoke on both cards at once (fresh models, separate directories): one epoch
# on 20k jets per card. Each must write a result.
import glob, json
smoke = []
for i in range(min(NGPU, len(SEEDS))):
    d = "/kaggle/working/smoke/%s/card%d" % (TAG, i)
    smoke.append((d, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", "1", "--max-train-samples", "20000",
                             "--seed", "0", "--results-dir", d],
                            i, LOGS + "/smoke_card%d.log" % i)))
codes = [p.wait() for _, p in smoke]
works = []
for i, ((d, _), rc) in enumerate(zip(smoke, codes)):
    found = glob.glob(d + "/*.json")
    print("card %d: exit %d, result %s" % (i, rc, bool(found)))
    if found:
        works.append(i)
    else:
        print(tail(LOGS + "/smoke_card%d.log" % i))
assert works, "the smoke run failed on every card"
PARALLEL = len(works) >= len(SEEDS)
print("parallel:" if PARALLEL else "only card(s) %s work: seed %d alone" % (works, SEEDS[0]),
      "seeds", SEEDS if PARALLEL else SEEDS[:1])
"""),
        code("""
import glob, time
todo = SEEDS if PARALLEL else SEEDS[:1]
procs = []
for i, s in enumerate(todo):
    gpu = i if PARALLEL else works[0]
    procs.append((s, launch(["benchmarks.run_top_tagging"] + BASE +
                            ["--epochs", str(EPOCHS), "--seed", str(s),
                             "--results-dir", OUT + "/seed%d" % s,
                             "--ckpt-dir", CKPT + "/seed%d" % s,
                             "--resume", "--max-seconds", str(MAX_SECONDS)],
                            gpu, LOGS + "/seed%d.log" % s)))
    print("seed %d on card %d" % (s, gpu), flush=True)
BAD = []
for s, p in procs:
    rc = p.wait()
    print("=== seed %d: exit %d ===" % (s, rc))
    print(tail(LOGS + "/seed%d.log" % s, 2000))
    if rc != 0 or not glob.glob(OUT + "/seed%d/*.json" % s):
        BAD.append(s)
print("failed seeds:", BAD or "none")
print("kernel time %.2f h" % ((time.time() - T0) / 3600))
"""),
        code("""
# Each continued seed against its own epoch 60 (AUC, rejection at 30%%, final
# validation loss), and against the 120-epoch headline.
import json, glob
OWN60 = %r
REF120 = %r
for f in sorted(glob.glob(OUT + "/seed*/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]; env = r["env"]
    h = r["history"]; vl = [e["val_loss"] for e in h]; s = r["seed"]
    assert env["torch"] == "2.11.0+cu128" and "T4" in env.get("gpu", ""), env
    print("seed %%d: %%d epochs (history %%d), %%.2f h cumulative | AUC %%.5f  rej@0.3 %%.1f  final val loss %%.5f"
          %% (s, r["epochs_run"], len(h), r["walltime_sec"] / 3600, t["test_auc"], t["bg_rej_30"], vl[-1]))
    if r["epochs_run"] < EPOCHS:
        print("  INCOMPLETE (epoch %%d of %%d): finish before comparing" %% (r["epochs_run"], EPOCHS))
        continue
    v60, a60, j60 = OWN60[s]
    print("  against its own epoch 60: dAUC %%+.5f, d rej@0.3 %%+.1f, d final val loss %%+.5f"
          %% (t["test_auc"] - a60, t["bg_rej_30"] - j60, vl[-1] - v60))
    print("  against the 120-epoch headline: dAUC %%+.5f, d rej@0.3 %%+.1f"
          %% (t["test_auc"] - REF120[0], t["bg_rej_30"] - REF120[1]))
    if r.get("test_metrics_best_val"):
        tb = r["test_metrics_best_val"]
        print("  best-val weights (epoch %%d): AUC %%.5f, rej@0.3 %%.1f"
              %% (r["best_val_epoch"], tb["test_auc"], tb["bg_rej_30"]))
assert not BAD, "seeds failed (exit code or no result): %%s" %% BAD
""" % (refs, K64_REF120)),
    ]
    stem = "k64_%s_seeds%s" % (tag, "".join(str(s) for s in seeds))
    name = "kaggle_%s.ipynb" % stem
    _finalise(cells, NB_DIR / name)
    slug = "k64-%s-seeds%s" % (tag.replace("_", "-"), "".join(str(s) for s in seeds))
    meta = {
        "id": "nsanya/so3c-" + slug,
        "title": "SO3C K64 %s seeds%s" % (tag, "".join(str(s) for s in seeds)),
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128", src_kernel],
    }
    (NB_DIR / ("kernel-metadata-%s.json" % slug)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


def build_k64_continuation_finish(blob: str) -> str:
    """Finish the 60 -> 120 continuation if it stops on the session cap.

    Prepared, not to be pushed unless the continuation reports INCOMPLETE.
    It mounts that kernel's output, copies its last checkpoint and resumes
    with the same --epochs 120, so the schedule horizon is unchanged and
    resume is bit-exact apart from the GPU's own nondeterminism.
    """
    import hashlib
    bundle_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    base = _k64_headline_base()
    cells = [
        md("""
# Finish: K=64 headline 60 -> 120 continuation

Resumes `nsanya/so3c-k64-headline120-seed0` from its last checkpoint, with
the same 120-epoch horizon. Push only if that run stopped on the session cap.
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib, shutil
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
TAG = %r
OUT = "/kaggle/working/results_k64/" + TAG
CKPT_DIR = "/kaggle/working/checkpoints/" + TAG
BASE = %r
BASE[BASE.index("DATA")] = DATA
prev = glob.glob("/kaggle/input/**/checkpoints/" + TAG + "/*.pt", recursive=True)
assert prev, "attach the output of nsanya/so3c-k64-headline120-seed0"
pathlib.Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
shutil.copy(prev[0], CKPT_DIR)
import torch
st = torch.load(prev[0], map_location="cpu", weights_only=False)
print("resuming from epoch", st["epoch"], "| bundle", %r)
assert st["epoch"] < 120, "the continuation already finished"
""" % (CONT_TAG, base, bundle_sha)),
        code(_CONT_RUN),
        code(_CONT_SUMMARY),
    ]
    name = "kaggle_k64_%s_finish.ipynb" % CONT_TAG
    _finalise(cells, NB_DIR / name)
    meta = {
        "id": "nsanya/so3c-k64-%s-finish" % CONT_TAG,
        "title": "SO3C K64 %s finish" % CONT_TAG,
        "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["nsanya/so3c-data-prep-k-128",
                           "nsanya/so3c-k64-%s-seed0" % CONT_TAG],
    }
    (NB_DIR / ("kernel-metadata-k64-%s-finish.json" % CONT_TAG)).write_text(
        json.dumps(meta, indent=2) + chr(10), encoding="utf-8")
    return name


def build_message(blob: str, seed: int = 0, k: int = 64) -> None:
    """Tier-2 item 5: covariant message passing on the canonical protocol.

    The single-round covariant model puts every constituent through three
    complex numbers and one rotation. `so3c_message_set` runs the same
    covariant connection three times and carries a per-particle scalar
    channel alongside the vector one, which is what LGEB/PELICAN do and
    what our architecture has been missing.

    Dense, not kNN. On the canonical protocol the k=16 graph cost 0.0011
    AUC and 93 rejection and saved 2% of the wall clock (0.57 h against
    0.58 h) -- the edge MLP is not the bottleneck, expm_so3c over
    (B, C, K) generators and the pooled readout are, and neither cares how
    sparse the graph is. The sparsification stays in the model, tested and
    equivariant, but there is no reason to pay for it here.

    Cost measured on the GPU, not extrapolated: message passing is 3.05x
    the covariant model on the same protocol, so K=64 is ~5.1 h/seed
    against its 1.66 h. --max-seconds and a checkpoint dir cover a cut
    session.

    The smoke cell runs 1 epoch on 20k jets first: it costs ~2 minutes and
    catches an OOM or a shape error before the session commits hours.
    """
    cells = [
        md("""
# Covariant message passing

`so3c_covariant_set` scores AUC 0.9746 +- 0.0001 / rejection 312 at K=32 and
0.9772 +- 0.0001 / 638 at K=64 (9078 params). This run asks whether the
architecture is limited by having only ONE round of covariant mixing and no
scalar channel -- the two things every published Lorentz-equivariant tagger
has that we do not. On 40k jets and 20 epochs off the canonical protocol,
three rounds took 0.9679 -> 0.9749 and rejection 132 -> 303.

The model stays exactly equivariant: the graph is ranked by an invariant,
the messages are invariants, the connection is a cross product of covariant
vectors, and the channel mixing is complex-linear.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_message"
CKPT = "/kaggle/working/checkpoints"
print("data:", DATA)
"""),
        code("""
# Smoke: 1 epoch on 20k jets. Catches an OOM or a shape error in ~2 min,
# before the session commits hours to the real run.
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "1", "--normalize", "global",
     "--max-train-samples", "20000",
     "--seed", "0", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "%d",
     "--rounds", "3",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", "/kaggle/working/smoke"])
""" % k),
        code("""
run(["benchmarks.run_top_tagging",
     "--cache-dir", DATA, "--representation", "constituents",
     "--canonical-splits", "--epochs", "30", "--normalize", "global",
     "--seed", "%d", "--device", "cuda", "--dtype", "float32",
     "--batch-size", "256", "--n-constituents", "%d",
     "--rounds", "3",
     "--eval-chunk-size", "1024",
     "--models", "so3c_message_set",
     "--results-dir", OUT, "--ckpt-dir", CKPT,
     "--resume", "--max-seconds", "26000"])
""" % (seed, k)),
        code("""
import json, glob
print("%-22s%9s%9s%10s%8s" % ("model", "params", "AUC", "rej@0.3", "hours"))
print("%-22s%9d%9.4f%10.0f%8.2f"
      % ("covariant K=32", 9078, 0.9746, 312, 0.53))
print("%-22s%9d%9.4f%10.0f%8.2f"
      % ("covariant K=64", 9078, 0.9772, 638, 1.66))
for f in sorted(glob.glob(OUT + "/*.json")):
    r = json.load(open(f)); t = r["test_metrics"]
    print("%-22s%9d%9.4f%10.0f%8.2f"
          % (r["model"], r["n_params"], t["test_auc"], t["bg_rej_30"],
             r["walltime_sec"] / 3600))
"""),
    ]
    _finalise(cells, NB_DIR / ("kaggle_message_k%d_seed%d.ipynb" % (k, seed)))


def build_message_finish(blob: str) -> None:
    """Finish the two K=64 message-passing runs that stopped at epoch 28.

    Both hit --max-seconds 26000 at epoch 28 of 30. The per-epoch cost at
    K=64 is 930 s, 4.7x the covariant model, not the 3.05x measured at K=32
    that the cap was sized from. train.py writes a checkpoint every epoch
    before it checks the time limit, so the epoch-28 state exists, in each
    kernel's output under checkpoints/.

    A new Kaggle session starts with an empty /kaggle/working, so --resume
    on its own would find no checkpoint and train from epoch 1 again. This
    kernel mounts both runs as inputs, copies each checkpoint to the path
    the runner derives (ckpt_dir / "{experiment}__{model}__seed{seed}.pt"),
    and resumes. train.py restores model, optimizer, cosine scheduler and
    RNG state and starts at epoch 29, so epochs 29-30 run as they would have
    in one session.

    Results go to results_message_e30, so they cannot be mistaken for the
    epoch-28 files. The summary cell refuses to report a run that did not
    reach 30 epochs. Wall clock in the result JSON is cumulative across
    sessions, so each resume's own session time is printed separately.

    This relies on train.py capping --max-seconds per session. It used to
    compare the cumulative clock, and these two runs are already past the
    cap (26026 s and 26068 s against 26000), so each would have trained
    epoch 29 and stopped again.

    The first attempt at this kernel lost both seeds in about a minute to a
    second resume bug that only exists on a GPU: checkpoints were loaded
    with map_location=device, which moved the saved RNG states onto CUDA,
    and torch.set_rng_state rejects anything but a CPU ByteTensor. Its
    summary cell also passed silently with zero result files. So this
    version runs the CUDA resume test on the card before the real runs and
    requires it to pass rather than skip, asserts each run succeeded, and
    requires both result files to exist.
    """
    cells = [
        md("""
# Finish K=64 message passing: epochs 29-30

Both anchor-configuration runs stopped at epoch 28 on the time cap, with
AUC 0.98074 / 0.98065 and rejection 893 / 818. The cosine schedule had two
epochs left to anneal. This resumes each from its epoch-28 checkpoint.
"""),
        code("""
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip())
"""),
        code(UNPACK.format(blob=blob)),
        code(RUNNER),
        code(GPU_SETUP),
        code("""
import glob, pathlib, shutil
cands = glob.glob("/kaggle/input/**/top_tagging_train.npz", recursive=True)
assert cands, "attach the K=128 data-prep kernel output"
DATA = str(pathlib.Path(cands[0]).parent)
OUT = "/kaggle/working/results_message_e30"
CKPT = pathlib.Path("/kaggle/working/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)
for seed in (0, 1):
    name = "top_tagging_canonical__so3c_message_set__seed%d.pt" % seed
    found = glob.glob("/kaggle/input/**/checkpoints/" + name, recursive=True)
    assert found, "attach nsanya/so3c-message-k64-seed%d: no checkpoint %s" % (seed, name)
    shutil.copy(found[0], CKPT / name)
    print("seed %d: %s (%d bytes)" % (seed, found[0], (CKPT / name).stat().st_size))
print("data:", DATA)
"""),
        code("""
# Exercise a real CUDA resume before committing the session to the runs.
# A skip would look green, so require an actual pass.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_harness.py",
                    "-k", "cuda_resume"], cwd="/kaggle/working/repo",
                   capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
assert r.returncode == 0 and "1 passed" in r.stdout, "CUDA resume test did not pass; not starting the runs"
"""),
        code("""
import time
for seed in ("0", "1"):
    print("=== seed %s ===" % seed, flush=True)
    t_seed = time.perf_counter()
    ok = run(["benchmarks.run_top_tagging",
         "--cache-dir", DATA, "--representation", "constituents",
         "--canonical-splits", "--epochs", "30", "--normalize", "global",
         "--seed", seed, "--device", "cuda", "--dtype", "float32",
         "--batch-size", "256", "--n-constituents", "64",
         "--rounds", "3",
         "--eval-chunk-size", "1024",
         "--models", "so3c_message_set",
         "--results-dir", OUT, "--ckpt-dir", str(CKPT),
         "--resume", "--max-seconds", "26000"])
    print("seed %s: this session took %.1f min" % (seed, (time.perf_counter() - t_seed) / 60))
    assert ok, "seed %s failed; see the output above" % seed
"""),
        code("""
import json, glob
EPOCH28 = {0: (0.98074, 893.4), 1: (0.98065, 817.5)}
FILES = sorted(glob.glob(OUT + "/*.json"))
assert len(FILES) == 2, "expected 2 result files, found %d" % len(FILES)
print("%-6s%8s%10s%10s%10s%12s%8s" % ("seed", "epochs", "AUC", "rej@0.3", "AUC@28", "rej@28", "hours"))
for f in FILES:
    r = json.load(open(f)); t = r["test_metrics"]
    hours = r.get("walltime_sec", 0) / 3600   # cumulative across sessions
    a28, j28 = EPOCH28[r["seed"]]
    print("%-6d%8d%10.5f%10.1f%10.5f%12.1f%8.2f"
          % (r["seed"], r["epochs_run"], t["test_auc"], t["bg_rej_30"], a28, j28, hours))
    assert r["epochs_run"] == 30, "seed %d stopped at epoch %d" % (r["seed"], r["epochs_run"])
"""),
    ]
    _finalise(cells, NB_DIR / "kaggle_message_k64_finish.ipynb")


def _registry() -> dict:
    """Every notebook this module can build: name -> builder(blob).

    The name is the notebook's file stem without the "kaggle_" prefix, so
    `--only k64_headline120_seed0` writes notebooks/kaggle_k64_headline120_seed0.ipynb
    (and, for kernels that need one, its kernel-metadata file).
    """
    reg = {
        "dataprep_k128": lambda b: build_dataprep(b, k=128),
        "validate": build_validate,
        "scaling": build_scaling,
        "fixed": build_fixed,
        "k64_seed1": lambda b: build_k64_seed(b, seed=1),
        "kappa": build_kappa,
        "message_probe": build_message_probe,
        "message_sweep": build_message_sweep,
        "message_k64_finish": build_message_finish,
        "beams_probe": build_beams_probe,
        "beams_tuning": build_beams_tuning,
        "beams_k64_seed0": lambda b: build_beams_k64(b, seed=0),
        "beams_k64_seed1": lambda b: build_beams_k64(b, seed=1),
        "beams_k64_finish": build_beams_k64_finish,
        "capacity_cost": build_capacity_cost,
        "gap_screen": build_gap_screen,
        "gpu_profile": build_gpu_profile,
        "cuda_diag": build_cuda_diag,
        "width_profile": build_width_profile,
        "knn_profile": build_knn_profile,
        "k64_%s_seed0" % CONT_TAG: build_k64_continuation,
        "k64_%s_finish" % CONT_TAG: build_k64_continuation_finish,
    }
    for tag, flags, seed in LONG_RUNS:
        reg["long_seeds_%s_s%d" % (tag, seed)] = (
            lambda b, t=tag, f=flags, s=seed: build_long_seeds(b, t, f, (s,), part="s%d" % s))
    for tag, flags, seed in LONG_RUNS_PINNED:
        reg["long_seeds_%s_s%d" % (tag, seed)] = (
            lambda b, t=tag, f=flags, s=seed: build_long_seeds(
                b, t, f, (s,), part="s%d" % s, pinned=True))
    for tag, flags, seeds in LONG_DUAL_RUNS:
        reg["long_%s_seeds%s" % (tag, "".join(str(s) for s in seeds))] = (
            lambda b, t=tag, f=flags, s=seeds: build_long_dual(b, t, f, s))
    for tag, flags, seeds in PROBE_SEED_RUNS:
        reg["probe_seeds_%s" % tag] = (
            lambda b, t=tag, f=flags, s=seeds: build_probe_seeds(b, t, f, s))
    for tag, flags, seeds, budget_h in PROBE_SCREEN_RUNS:
        reg["probe_screen_%s" % tag] = (
            lambda b, t=tag, f=flags, s=seeds, h=budget_h: build_probe_screen(b, t, f, s, h))
    for tag, flags, seeds, budget_h in FRESH_CONTROL_RUNS:
        reg["probe_screen_%s" % tag] = (
            lambda b, t=tag, f=flags, s=seeds, h=budget_h: build_probe_screen(
                b, t, f, s, h, reproduce=_check_rerun()))
    for tag, flags, seeds, budget_h, check in REPR_SCREEN_RUNS:
        reg["probe_screen_%s" % tag] = (
            lambda b, t=tag, f=flags, s=seeds, h=budget_h, c=check: build_probe_screen(
                b, t, f, s, h, pairing_check=c, control=_screen_control()))
    for tag, flags, seed, epochs in K64_RUNS:
        reg["k64_%s_seed%d" % (tag, seed)] = (
            lambda b, t=tag, f=flags, s=seed, e=epochs: build_k64_run(b, t, f, s, epochs=e))
    for tag, jobs, budget_h in PROBE_QUEUES:
        reg["probe_queue_%s" % tag.replace("-", "_")] = (
            lambda b, t=tag, j=jobs, h=budget_h: build_probe_queue(
                b, t, j, h, summary=PROBE_QUEUE_SUMMARY.get(t, "ablation")))
    for src, tag, flags, seeds in K64_DUAL_CONTINUATIONS:
        reg["k64_%s_seeds%s" % (tag, "".join(str(s) for s in seeds))] = (
            lambda b, sr=src, t=tag, f=flags, s=seeds: build_k64_continuation_dual(b, sr, t, f, s))
    for tag, flags, seeds, epochs in K64_DUAL_RUNS:
        reg["k64_%s_seeds%s" % (tag, "".join(str(s) for s in seeds))] = (
            lambda b, t=tag, f=flags, s=seeds, e=epochs: build_k64_dual(b, t, f, s, epochs=e))
    reg["k64_%s" % K64_FINISH_QUEUE_NAME] = (
        lambda b: build_k64_finish_queue(b, K64_FINISH_QUEUE_NAME, K64_FINISH_QUEUE_JOBS,
                                         K64_FINISH_QUEUE_SOURCES, K64_FINISH_QUEUE_BUDGET_H))
    reg["k64_%s" % K64_SEED2_NAME] = (
        lambda b: build_k64_jobs_dual(b, K64_SEED2_NAME, K64_SEED2_JOBS))
    reg["k64_%s_finish" % K64_SEED2_NAME] = (
        lambda b: build_k64_jobs_finish(b, K64_SEED2_NAME, K64_SEED2_JOBS))
    for tag, flags, seeds, epochs in K64_FINISH_DUAL:
        reg["k64_%s_finish_seeds%s" % (tag, "".join(str(s) for s in seeds))] = (
            lambda b, t=tag, f=flags, s=seeds, e=epochs: build_k64_finish_dual(
                b, t, f, s, epochs=e, max_seconds=K64_FINISH_MAX_SECONDS.get(t, 39000)))
    for tag, flags, seed, epochs in K64_RUNS_PINNED:
        reg["k64_%s_seed%d" % (tag, seed)] = (
            lambda b, t=tag, f=flags, s=seed, e=epochs: build_k64_run(
                b, t, f, s, epochs=e, pinned=True))
    for k in (32, 64):
        for seed in (0, 1):
            reg["message_k%d_seed%d" % (k, seed)] = (
                lambda b, kk=k, s=seed: build_message(b, seed=s, k=kk))
    return reg


def main(argv: list[str] | None = None) -> int:
    """Build named notebooks; with no arguments, build nothing.

    Every committed notebook embeds the code of the commit it was built from,
    and that bundle is the provenance of its kernel's results. Rebuilding a
    notebook that has already run therefore rewrites its record, so nothing
    is built unless asked for by name, or everything with an explicit --all.
    Plan_2026-10-10 C3: before this, even --help rewrote every notebook.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.build_notebooks",
        description="Build the Kaggle notebooks, each with the repository's "
                    "code embedded. With no arguments nothing is written.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="list the notebook names")
    g.add_argument("--only", nargs="+", metavar="NAME",
                   help="build exactly these notebooks (names from --list)")
    g.add_argument("--all", action="store_true",
                   help="rebuild every notebook, including ones that already ran")
    args = p.parse_args(argv)
    reg = _registry()
    if args.list:
        for name in reg:
            print(name)
        return 0
    if args.only:
        unknown = [n for n in args.only if n not in reg]
        if unknown:
            p.error("unknown notebook name(s): %s (see --list)" % ", ".join(unknown))
        names = list(dict.fromkeys(args.only))
    elif args.all:
        names = list(reg)
    else:
        p.print_usage()
        print("nothing built: name notebooks with --only, or rebuild all with --all")
        return 0
    blob = embed_code()
    print("embedded code: %.0f KB base64" % (len(blob) / 1024))
    for name in names:
        reg[name](blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
