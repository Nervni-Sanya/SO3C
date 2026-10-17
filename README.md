# SO3C — Lorentz-covariant jet tagging with complexified-SO(3) geodesic flows

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Scope.** This repository is SO3C only. The SO(3,3) activation it grew out
> of — the `so33` package, its preprint and its own benchmarks — lives in a
> separate repository, [Nervni-Sanya/so33](https://github.com/Nervni-Sanya/so33).
> What was removed here on 2026-09-18, and where each piece went, is listed in
> [`REMOVED.md`](REMOVED.md).

SO3C builds jet taggers on the complexified rotation algebra
so(3) ⊕ i·so(3) = so(3,ℂ) ≅ so(3,1), which is the Lorentz algebra. Each
particle carries a complex 3-vector z ∈ ℂ³. A learned connection a ∈ ℂ³ moves
it along the flow ż = −[a]× z, which conserves the bilinear invariant
z·z = Σᵢ zᵢ² and is evaluated in closed form with a complex Rodrigues formula,
so no ODE solver is involved. Jet constituents enter ℂ³ through a bivector lift
on which a Lorentz transformation acts as z ↦ Qz with Q ∈ SO(3,ℂ).

## Goal and status

The goal is state of the art on the top-tagging reference dataset of
Kasieczka et al. (arXiv:1902.09914) with its published split: 1,211,000
training, 403,000 validation and 404,000 test jets. It has not been reached.

| Model | Parameters | AUC | 1/ε_B at ε_S = 0.3 |
|---|---:|---|---|
| **SO3C** with lab-frame pair features, `so3c_message_set --beams --channels 8 --pair-lab`, K = 64, 60 epochs continued to 120, mean of 2 seeds | 23,026 | **0.98460 ± 0.00029** | 1302 ± 89 |
| the same, the better seed by test AUC (the other: 0.98439 / 1238.7) | 23,026 | 0.98480 | 1364.3 |
| the same, ensemble of the 2 seeds | 2 × 23,026 | **0.98515** | 1421.9 |
| the same, ensemble of the 2 seeds at 60 and at 120 epochs (four snapshots) | 4 × 23,026 | 0.98513 | **1473.8** |
| the same at 60 epochs, mean of 2 seeds | 23,026 | 0.98442 ± 0.00035 | 1296 ± 71 |
| SO3C without them, `so3c_message_set --beams --channels 8`, K = 64, 60 epochs continued to 120, 1 seed | 22,834 | 0.98392 | 1201.9 |
| the same run at 60 epochs | 22,834 | 0.98363 | 1140.8 |
| the same configuration, mean of 3 seeds at 30 epochs | 22,834 | 0.98326 ± 0.00014 | 1089 ± 75 |
| ensemble: those 3 seeds plus the 60-epoch model | 4 × 22,834 | 0.98396 | 1231.2 |
| ensemble: those 4 plus the 120-epoch model (two snapshots of one run) | 5 × 22,834 | 0.98413 | 1269.9 |
| PELICAN | 208,000 | 0.9870 | 2250 ± 75 |
| LorentzNet | 224,000 | 0.9868 | 2195 ± 173 |
| L-GATr, trained from scratch | 1,100,000 | 0.9870 ± 0.0001 | 2240 ± 70 |
| PELICAN, its own size sweep | 11,000 | 0.9858 | 1879 ± 103 |
| PELICAN, its own size sweep | 3,000 | 0.9850 | 1494 ± 43 |
| PELICAN, its own size sweep | 1,000 | 0.9835 | 1145 ± 74 |

The lab-frame pair features (2026-10-17) give the largest gain since the beams:
+0.00100 AUC at K = 64 over a same-stack control without them (two runs each,
0.98442 against 0.98342; Welch t = 3.5 on about 1.7 degrees of freedom, p ≈ 0.1,
with both pair-feature runs above both controls). On its own that is suggestive;
it repeats the gain measured earlier on a short probe (+0.00051, t = 5.9) and at
K = 32 over 60 epochs (+0.00064, t = 6.8). Training time is about the control's
(0.98–1.08×). The features are ParT's ln Δ, ln k_T, ln z and ln m² on every pair
of distinct constituents (zero on the diagonal and on beam and padding edges), written in Minkowski products
of the constituents with each other and with the beams, so the model stays
exactly covariant when the beams move with the jet. Continuing both runs to 120 epochs added
+0.00018 AUC on the mean. The gap to PELICAN is now 0.0018 AUC and a factor 1.58
in rejection for the two-seed ensemble (0.0024 and 1.73 for the two-seed mean).
PELICAN's 3,000-parameter model (0.9850 / 1494 ± 43) beats the best single run
on both metrics; the ensembles match it in AUC and come within 20–72 of its
rejection.

Without the pair features, training longer appears to have helped: 60 epochs beat the 30-epoch mean by
+0.00037 AUC, 2.2 standard errors with only three 30-epoch seeds to compare
against (t ≈ 2.2 on two degrees of freedom, p ≈ 0.15) — suggestive, not
established. Its final validation loss, 0.16276, is below all three complete
30-epoch runs. Continuing that run to 120 epochs (a warm restart of the cosine
schedule) added +0.00029 AUC and 61 rejection against its own epoch 60. Without
the pair features, the gap to PELICAN from the best ensemble is 0.0029 AUC and a factor of 1.77 in
background rejection; 0.0031 and 1.87 for the best single model; it is 0.0032 and 1.94 for the
three-seed 30-epoch ensemble, and 0.0037 and 2.07 for a single 30-epoch
model. PELICAN's 1,000-parameter model still matches SO3C's 60-epoch model
and is 0.0004 AUC below the 120-epoch one, which points to architecture rather than size, though that rests on
one point of PELICAN's own sweep, trained with its own recipe. Every published
number above this band that we have found comes from fine-tuning after
JetClass pretraining; the highest in the reference table is L-GATr fine-tuned
(0.98793 / 2894). That is a different comparison class: the same architecture
trained from scratch on this dataset lands at 0.9870 / 2240, in line with
PELICAN. The published values were checked against the source
tables ([`figures/literature_reference.csv`](figures/literature_reference.csv),
[`figures/pelican_scaling.csv`](figures/pelican_scaling.csv)). The
working status — candidates under test, questions closed by measurement, the
GPU budget — is in [`SO3C_STATUS.md`](SO3C_STATUS.md).

More parameters help only a little so far. Widening the network
(`--act-hidden`, `--msg-dim`, `--scalar-dim`) is the first capacity change
shown to pay in a paired, multi-seed comparison (the earlier step from 4 to 8
channels gave +0.0004 AUC on one seed, not resolved): at K = 32 on the full
training set for 60 epochs, the 120,202-parameter `w128s32` gains +0.00048 AUC
over the 22,834-parameter base, paired by seed over two seeds — about +0.0002
per doubling of parameters. At K = 64 over 30 epochs the 60,834-parameter
`w64s24` gains only +0.00008 (one seed), and weight decay 0.002 on the wide
model costs −0.00011 (two seeds). Even a full transfer of the K = 32 width gain
would have left about 0.0024 AUC to PELICAN from the beams-only ensemble. The
lab-frame pair features, an input change, are the first of the structural
candidates shown to help at the long schedules: +0.00064 at K = 32 (two runs
against five) and +0.00100 at K = 64 (two runs against two).
The other structural candidates, screened on the 20-epoch probe, moved AUC by
+0.00012 or less and have not been run at the long schedule. Training beyond 60 epochs
was tried on three runs (continued to 120: +0.00029 AUC on the beams-only run,
+0.00022 and +0.00013 on the two pair-feature runs; unresolved on each run alone); width at K = 64 over
60 epochs and other weight-decay strengths are untested
([measurements below](#width-and-weight-decay)).

## How it works

1. **Lift.** A constituent with 4-momentum p_a in a jet with total momentum P
   becomes z_a = (E_a P⃗ − E_P p⃗_a) + i (p⃗_a × P⃗). Transforming every
   4-momentum by a Lorentz matrix Λ maps each z_a to Q z_a, with Q ∈ SO(3,ℂ)
   the matching group element.
2. **A covariant connection.** A connection computed from invariants alone
   conserves z·z but does not turn with the jet, so a model built on it stops
   being invariant as soon as training moves the connection away from zero
   ([theory §5](docs/so3c/theory.md#5-conservation-is-not-equivariance)). SO3C
   uses a_a = z_a × Σ_b φ(invariants) z_b, which transforms as a ↦ Qa, and
   updates z_a ↦ exp(−T [a_a]×) z_a.
3. **Message passing** (`so3c_message_set`): three such rounds over all pairs
   of particles, a real scalar state per particle updated by invariant edge
   messages, and complex channel mixing
   ([theory §10](docs/so3c/theory.md#10-covariant-message-passing-so3c_message_set)).
4. **Beams.** Two beam 4-vectors (1, 0, 0, ±1) join the particle set, as in
   LorentzNet and PELICAN, so the network can use each constituent's lab-frame
   energy and longitudinal momentum
   ([theory §11](docs/so3c/theory.md#11-beams-what-symmetry-survives)).

**Which symmetry the output has.** Without beams, `so3c_message_set`,
`so3c_covariant_set` and `so3c_invariant_set` are Lorentz-invariant by
construction, up to floating-point round-off; `so3c_interaction_set` is
invariant only to solver tolerance, and `so3c_equivariant_set` stops being
invariant once trained ([Models](#models)). With beams the network is exactly
covariant when the jet and the beams are transformed together; its output
changes when only the jet is boosted or tilted, and stays unchanged only under
rotations about the beam axis. That is the intended physics. Measured in
float64 on the beams-only architecture (`--beams --channels 8`) with its
zero-initialised weight heads set to random values, on six random jets at
boost scale 0.5: the logits move by 2×10⁻¹⁶ when the beams move with the jet,
and by 3×10⁻² when they stay fixed
([theory §11](docs/so3c/theory.md#11-beams-what-symmetry-survives)). With `--pair-lab` the same
check holds in the unit tests to below 10⁻⁹ relative.

## Models

| `--models` name | Parameters at the defaults | Output symmetry |
|---|---|---|
| `so3c_message_set` | 13,862; 15,038 with `--beams`; 22,834 with `--beams --channels 8`; 23,026 adding `--pair-lab` | Lorentz-invariant; with beams, covariant together with the beams and invariant under rotations about the beam axis |
| `so3c_covariant_set` | 9,078 | Lorentz-invariant, exactly |
| `so3c_invariant_set` | 5,122 | Lorentz-invariant by construction; no flow |
| `so3c_interaction_set` | 5,652 | Lorentz-invariant to ODE tolerance |
| `so3c_equivariant_set` | 9,056 | **not invariant once trained**; kept to reproduce the results measured on it |
| `so3c`, `so3c_static`, `so3c_multi` | depends on the input width | none: flat tabular models on a learned embedding |

`so3c_message_set` also has six switches, all off by default:
`--no-mass-input`, `--no-self-edges`, `--relnorm-edge`, `--falpha N`,
`--vector-channel` and `--pair-latent N`. Each was screened on a short probe
protocol and none showed a resolved gain; none has been trained on the full data
([models](docs/so3c/models.md#switches-screened-on-the-probe-protocol)). Later switches are in
[models](docs/so3c/models.md#lab-frame-features-added-and-screened-2026-10-10); of them,
`--pair-lab` (ParT's four pair features, 23,026 parameters) passed the probe
screen and gives the current best models on the full data.

Three width flags scale the model: `--act-hidden` (the edge network that feeds
the connection, the messages and the node update, default 16), `--msg-dim` (the edge message, default 8) and `--scalar-dim` (the
per-particle scalar channel, default 8). On `--beams --channels 8`,
`--act-hidden 64 --msg-dim 24 --scalar-dim 24` (`w64s24`) has 60,834
parameters and `--act-hidden 128 --msg-dim 32 --scalar-dim 32` (`w128s32`) has
120,202.

## Results on the published split

| Model | Parameters | K | Seeds | AUC | 1/ε_B at ε_S = 0.3 |
|---|---:|---:|---:|---|---|
| `eta_invariants`, the SO(3,3) invariant readout, as a baseline | 4,802 | 32 | 3 | 0.9478 | 49 |
| Deep Sets MLP with the same parameter count and no geometry | 9,053 | 32 | 1 | 0.7636 | 8.5 |
| `so3c_covariant_set` | 9,078 | 32 | 3 | 0.9746 ± 0.0001 | 312 ± 19 |
| `so3c_covariant_set` | 9,078 | 64 | 2 | 0.9772 ± <0.0001 | 638 ± 1 |
| `so3c_message_set` | 13,862 | 64 | 2 | 0.9807 ± 0.0001 | 850 ± 56 |
| `so3c_message_set --beams --channels 8` | 22,834 | 64 | 3 | 0.98326 ± 0.00014 | 1089 ± 75 |
| `so3c_message_set --beams --channels 8`, 60 epochs | 22,834 | 64 | 1 | 0.98363 | 1140.8 |
| `so3c_message_set --beams --channels 8`, 60 epochs, same-stack control | 22,834 | 64 | 2 | 0.98342 ± 0.00021 | 1144 ± 14 |
| `so3c_message_set --beams --channels 8 --pair-lab`, 60 epochs | 23,026 | 64 | 2 | 0.98442 ± 0.00035 | 1296 ± 71 |
| `so3c_message_set --beams --channels 8 --pair-lab`, continued to 120 epochs | 23,026 | 64 | 2 | 0.98460 ± 0.00029 | 1302 ± 89 |

K is the number of leading constituents per jet; ± is the standard deviation
over seeds; every row trains for 30 epochs except the last four: three train
for 60, and the last continues the pair-feature runs to 120. The `eta_invariants` runs are in the untracked `results/` directory; every
other row has its result files committed.

Symmetry under boosts was tested on smaller models, trained for 8 epochs on
20,000 jets at K = 32 and evaluated in float64 on test jets pushed through
random Lorentz transformations (boost scale up to 3). The mean AUC of the
invariant models moves by at most 0.0001, and so does that of the beams model
when its beams move with the jet (the spread between transformation draws is up
to 0.0002), while `so3c_equivariant_set` falls from 0.9424 to 0.4414 ± 0.3284.

The results with their source directories, and the known limitations, are in
[`docs/so3c/experiments.md`](docs/so3c/experiments.md).

### Width and weight decay

| Model | Parameters | K | Epochs | Seeds | AUC | 1/ε_B at ε_S = 0.3 |
|---|---:|---:|---:|---:|---|---|
| base | 22,834 | 32 | 60 | 3 | 0.98242 ± 0.00020 | 909 ± 32 |
| `w64s24` | 60,834 | 32 | 60 | 1 | 0.98286 | 930 |
| `w128s32` | 120,202 | 32 | 60 | 2 | 0.98289 ± 0.00027 | 989 ± 65 |
| `w128s32`, AdamW with weight decay 0.002 | 120,202 | 32 | 60 | 2 | 0.98278 ± 0.00028 | 955 ± 114 |
| base, control rerun | 22,834 | 64 | 30 | 1 | 0.98311 | 1040.8 |
| `w64s24` | 60,834 | 64 | 30 | 1 | 0.98319 | 1035.5 |

The K = 32 rows use all 1,211,000 training jets. With dropout off and the same
torch build, runs that share a seed see the same batch order
(`train_classifier` reseeds after the model is built), and here the arms
tracked each other as pairs: `w128s32` beats the base by +0.00047 and +0.00049 on seeds 0 and 1,
both arms on torch 2.10. Weight decay costs −0.00010 and −0.00012 if those runs
pair, but they used torch 2.11 against 2.10 for their controls, and whether the
batch order survives that upgrade is unconfirmed (epoch-1 losses cannot tell).
Two pairs per comparison is thin. The two K = 64 rows share one stack (T4,
torch 2.11, compiled) and one seed, one run each; their difference, +0.00008,
is not resolved (whether same-seed runs pair at K = 64 is unmeasured; at the
probe protocol on this stack they did not, `SO3C_STATUS.md`). The control rerun
differs from the original seed-0 headline in torch (2.11 against 2.5.1, as
pinned in the notebook), card (T4 against P100), `torch.compile` (on against
off) and forward-pass implementation (equal in float64, rounding differently in
float32), and scored 0.00014 AUC lower, about one seed spread. Record the torch
version of every run before calling two runs paired.

## Quick start

```bash
git clone https://github.com/Nervni-Sanya/SO3C.git
cd SO3C
pip install -r requirements.txt
pip install -e .
pip install scikit-learn
```

`pip install -e .` installs the `so3c` and `benchmarks` packages; scikit-learn
computes AUC and background rejection. The so33 repository ships a top-level
`benchmarks` package of its own, so do not install both into one environment.

```python
import torch
from benchmarks.models import build_model, count_parameters

# The beams-only configuration; add "pair_lab": True for the current best
# (23026 parameters). Input: (B, K, 5) = (E, px, py, pz, mask).
model = build_model("so3c_message_set", in_features=4, out_features=2,
                    representation="constituents",
                    so3c_kwargs={"beams": True, "channels": 8})
print(count_parameters(model))                # 22834

jets = torch.zeros(4, 32, 5, dtype=torch.float64)
jets[..., 1:4] = torch.randn(4, 32, 3, dtype=torch.float64)
jets[..., 0] = jets[..., 1:4].norm(dim=-1)    # massless constituents: E = |p|
jets[..., 4] = 1.0                            # all 32 constituents are real
logits = model(jets)
print(logits.shape)                           # torch.Size([4, 2])
```

## Reproducing

```bash
# The top-tagging data with its published split. K = 64 runs need files converted
# with at least 64 constituent slots; the default keeps all 200.
pip install huggingface_hub pandas pyarrow tables
python -m benchmarks.download_top_tagging --cache-dir data
```

```bash
# Smoke test: three taggers, one epoch on 1,400 jets, seconds on a CPU
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --models so3c_invariant_set,so3c_covariant_set,so3c_message_set \
    --max-samples 2000 --epochs 1 --results-dir results_smoke
```

```bash
# The headline configuration, one seed: 4.1 hours on a Kaggle T4 with --compile.
# For the 60-epoch model use --epochs 60 (8.6 hours) and raise --max-seconds to
# about 39000; add --pair-lab --keep-best for the pair-feature runs
# (notebooks/kaggle_k64_pairlab60_seeds01.ipynb, two seeds in one two-T4 kernel);
# the current best models continue both to 120 epochs
# (notebooks/kaggle_k64_pairlab120_seeds01.ipynb, --epochs 120 --resume on their checkpoints).
# The best beams-only model continues the 60-epoch run to 120 epochs
# (notebooks/kaggle_k64_headline120_seed0.ipynb: --epochs 120 --resume on its checkpoint). A P100 falls back to eager; its time with the
# rewritten forward pass has not been measured.
# --max-seconds ends a session; rerunning the same command resumes from the checkpoint.
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --epochs 30 --normalize global --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 64 \
    --models so3c_message_set --rounds 3 --beams --channels 8 --eval-chunk-size 1024 \
    --compile --results-dir results_beams_k64 --ckpt-dir ckpt_beams_k64 --resume --max-seconds 30000

# Ensembles from the saved per-jet scores, no retraining: the three 30-epoch seeds
# (0.98378 / 1160.4), then the four-model ensemble of the table (0.98396 / 1231.2),
# whose members include two seed-0 runs and so need --allow-shared-seed, then the
# five-model ensemble with the 120-epoch run (0.98413 / 1269.9).
python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64 results_k64/headline
python -m benchmarks.ensemble_scores --allow-shared-seed \
    --results-dir results_message/beams_k64 results_k64/headline results_k64/headline60
python -m benchmarks.ensemble_scores --allow-shared-seed \
    --results-dir results_message/beams_k64 results_k64/headline results_k64/headline60 \
    results_k64/headline120
```

```bash
# The width screen: K = 32, all training jets, 60 epochs, the 120,202-parameter
# model; 6.4 to 7.0 hours per seed on a T4. Without the three width flags this is
# the base (3.4 to 3.5 hours). Add --keep-best to also test the lowest-validation-
# loss weights (the committed w128s32 runs did not use it).
python -m benchmarks.run_top_tagging --cache-dir data --representation constituents \
    --canonical-splits --normalize global --epochs 60 --seed 0 \
    --device cuda --dtype float32 --batch-size 256 --n-constituents 32 \
    --eval-chunk-size 2048 --models so3c_message_set --rounds 3 --beams \
    --channels 8 --act-hidden 128 --msg-dim 32 --scalar-dim 32 --compile \
    --results-dir results_width --ckpt-dir ckpt_width --resume --max-seconds 40000
```

```bash
# Symmetry on boosted test jets (float64 by default). With --beams the script also
# records the curve with the beams transformed along with the jet.
python -m benchmarks.run_boost_robustness --cache-dir data --models so3c_message_set \
    --beams --channels 8 --tag _beams_c8 --n 20000 --epochs 8
```

The GPU runs were executed as Kaggle notebooks generated by
`benchmarks/build_notebooks.py`, which embed the code. The workflow is in
[`notebooks/README.md`](notebooks/README.md); the pitfalls met along the way are
in the infrastructure table of [`SO3C_STATUS.md`](SO3C_STATUS.md).

## Documentation

| Page | Contents |
|---|---|
| [docs/so3c/theory.md](docs/so3c/theory.md) | The mathematics: realification, the closed-form exponential, why conservation is not equivariance, the covariant connection, the bivector lift, message passing, beams, the optional switches, numerical precision |
| [docs/so3c/models.md](docs/so3c/models.md) | Every tagger: pipeline, parameter counts, switches, and how its symmetry was measured |
| [docs/so3c/api.md](docs/so3c/api.md) | The `so3c` package, `benchmarks.so3c_models` and the harness, with runnable examples |
| [docs/so3c/experiments.md](docs/so3c/experiments.md) | Data, protocols, commands, results with their sources, limitations |
| [SO3C_STATUS.md](SO3C_STATUS.md) | Working status: goal, current best, closed questions, open candidates, compute budget, infrastructure pitfalls |
| [so3c_notes/](so3c_notes/) | Research notes, such as the gap-to-SOTA survey of 2026-09-13 |

## Repository layout

| Path | Contents |
|---|---|
| [`so3c/`](so3c/) | Algebra and closed-form exponential, `HermitianMetric`, `SO3CActivation`, `SO3CInteraction`, the bivector lift |
| [`benchmarks/so3c_models.py`](benchmarks/so3c_models.py) | Taggers, flat models and synthetic-task heads |
| [`benchmarks/`](benchmarks/), the other files | Data loaders, `build_model`, training loop, runners, `ensemble_scores`, figure scripts, Kaggle tooling |
| [`tests/`](tests/) | `test_so3c_*.py` (52 tests) and `test_harness.py` (23) |
| [`notebooks/`](notebooks/) | Generated Kaggle notebooks with the code embedded |
| [`figures/`](figures/) | Figures, their CSV twins and the published reference values |
| [`so3c_notes/`](so3c_notes/) | Research notes and the related-work survey |
| `results_message/`, `results_sweep/`, `results_long_seeds/`, `results_k64/`, `results_cleanup/`, `results_matched/`, `results_matched_canonical/`, `results_60ep/` | Committed results with per-jet test scores |
| `results_kappa/`, `results_fixed/`, `results_boost/`, `results_scaling/`, `results_init/` | Committed result files, no per-jet scores |
| `results_kcheck/` | Constituent-tail diagnostic: what K = 64 leaves out (`benchmarks/diag_constituent_tail.py`) |
| `results_profile/` | GPU timing profiles and kernel logs |
| `results_higgs_ablation/` | HIGGS feature-set ablation (matched bottleneck, three feature sets) |

`results/` and `data/` (the datasets, about 11 GB) are in `.gitignore`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

The suite has 75 tests. The 52 SO3C tests cover the algebra (5), the activation
(7), the interaction layer (5), the lift (4), the models (25, including the
forward-pass rewrite guard) and float32 behaviour (6); the 23 harness tests
cover checkpointing and resume, reproducibility, the training schedules, the
best-validation weights, seed ensembles and the figure scripts. On a CPU-only
machine 72 pass and three CUDA-only tests are skipped. The equivariance tests that exercise the
flow (`test_trained_regime_invariance`, the `*_is_equivariant_when_excited`
tests and the beams and bundle tests) set the zero-initialised weight heads to
random values first: on a freshly built model the flow is the identity, and an
invariance check would pass without testing the flow.
`test_set_lorentz_invariance` and `test_classifier_invariance` use freshly
built models, so for the flow models they check only the readout.

## License and citation

MIT, see [LICENSE](LICENSE). There is no SO3C paper, release or DOI yet;
[`CITATION.cff`](CITATION.cff) describes this software without one. The Zenodo
DOI 10.5281/zenodo.19763339 belongs to the SO(3,3) software in the
[so33 repository](https://github.com/Nervni-Sanya/so33), not to SO3C.

## Acknowledgements

Every GPU result in this repository was trained on **Kaggle**'s free notebook
sessions (P100 and T4), within their quota of 30 GPU-hours per week. The
headline configuration spent 10.6 of those hours per seed on a P100 with the
original code; on a T4 the rewritten forward pass and `torch.compile` take it
from a profiled 10.2 hours to a measured 4.1 (8.6 at 60 epochs). Either way,
this work would not exist without that free compute; the notebooks that ran
there, with the code embedded, are in [`notebooks/`](notebooks/).

The data is the top-tagging reference set of Kasieczka et al.
([arXiv:1902.09914](https://arxiv.org/abs/1902.09914), Zenodo 2603256).

The SO3C construction, benchmark harness and experiments were developed with
the assistance of **Claude (Anthropic)**.
