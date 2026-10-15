# SO3C presentation brief (2026-10-15)

A self-contained starting point for building a talk or poster, so the whole
repository does not have to be read. Every SO3C number here comes from a
committed result file, except the first two rows of the progress table (CPU
runs in the untracked `results/`). Published numbers come from
`figures/literature_reference.csv` and `figures/pelican_scaling.csv`. The
authoritative record is [`SO3C_STATUS.md`](../SO3C_STATUS.md); if this brief
and the status file ever disagree, the status file wins.

## The pitch, in three sentences

SO3C builds Lorentz-covariant jet taggers on the complexified rotation algebra
so(3,ℂ), which is isomorphic to the Lorentz algebra so(3,1). Each constituent
becomes a complex 3-vector relative to the jet, and message passing moves these
vectors along closed-form covariant flows (a complex Rodrigues formula, no ODE
solver) while exchanging invariant scalars. On the standard top-tagging
benchmark a 22.8k-parameter model reaches AUC 0.9839, between ResNeXt and
ParticleNet and below the best Lorentz-symmetric taggers.

## Headline numbers

Kasieczka top-tagging reference dataset, canonical split (1.211M training,
403k validation, 404k test jets). Rejection is 1/ε_B at signal efficiency 0.3.

| SO3C model | params | AUC | rejection | runs | source |
|---|---:|---|---|---|---|
| best single model: `so3c_message_set`, beams, 8 channels, K=64, 60 epochs continued to 120 | 22,834 | 0.98392 | 1202 | 1 | `results_k64/headline120/` |
| the same run at 60 epochs | 22,834 | 0.98363 | 1141 | 1 | `results_k64/headline60/` |
| the same configuration, 30 epochs | 22,834 | 0.98326 ± 0.00014 | 1089 ± 75 | 3 seeds | `results_message/beams_k64/`, `results_k64/headline/` |
| ensemble of five models: the three 30-epoch seeds plus the 60- and 120-epoch snapshots of a fourth run | 5 × 22,834 | 0.98413 | 1270 | — | `results_k64/headline120/ensemble_seeds012_plus_60ep_120ep.json` |

Published taggers on the same benchmark (each checked against its paper's
table):

| model | params | AUC | rejection |
|---|---:|---|---|
| ResNeXt | 1.46M | 0.9837 | 1147 ± 58 |
| ParticleNet | 366k | 0.9858 | 1615 ± 93 |
| ParT | 2.1M | 0.9858 | 1602 ± 81 |
| LorentzNet | 224k | 0.9868 | 2195 ± 173 |
| PELICAN | 208k | 0.9870 | 2250 ± 75 |
| L-GATr (from scratch) | 1.1M | 0.9870 | 2240 ± 70 |
| PELICAN size sweep, 11k | 11k | 0.9858 | 1879 ± 103 |
| PELICAN size sweep, 3k | 3k | 0.9850 | 1494 ± 43 |
| PELICAN size sweep, 1k | 1k | 0.9835 | 1145 ± 74 |
| LGN (an earlier Lorentz-group network) | 4.5k | 0.9640 | 435 ± 95 |

Gap to PELICAN: 0.0031 AUC and a factor 1.87 in rejection for the best single
model; 0.0029 and 1.77 for the ensemble.

## How the model got here (canonical protocol)

| step | params | K | AUC | rejection |
|---|---:|---:|---|---|
| baseline: SO(3,3) invariant readout (`eta_invariants`, not an SO3C model) | 4,802 | 32 | 0.9478 | 49 |
| invariant set, no flow | 5,506 † | 32 | 0.9689 | 183 |
| one covariant round (`so3c_covariant_set`) | 9,078 | 32 | 0.9746 ± 0.0001 | 312 ± 19 |
| the same at K=64 | 9,078 | 64 | 0.9772 ± <0.0001 | 638 ± 1 |
| message passing: 3 rounds + scalar channel | 13,862 | 64 | 0.98072 ± 0.00006 | 850 ± 56 |
| + beams + 8 channels (three seeds, 30 epochs) | 22,834 | 64 | 0.98326 ± 0.00014 | 1089 ± 75 |
| 60 epochs (one run) | 22,834 | 64 | 0.98363 | 1141 |
| 60 continued to 120 epochs (the same run) | 22,834 | 64 | 0.98392 | 1202 |

† As measured; six identically-zero features were removed later, and the
current model has 5,122 parameters.

This table is a ready-made "progress" slide; a waterfall or step chart of the
AUC column shows it well.

## Results worth a slide each

1. **Exact covariance, measured.** The headline configuration (22,834
   parameters), trained for 8 epochs on 14,000 jets at K=32 and evaluated in
   float64: with the beams boosted together with the jet, AUC stays at
   0.9684–0.9685 up to boost scale 3; with the beams pinned to the lab it drops
   to 0.9316 at scale 2 (0.9332 ± 0.0151 at scale 3). The short training is why
   its AUC is 0.968, not 0.984; say so on the slide. Source:
   `results_boost/boost__so3c_message_set_beams_c8__seed0.json`
   (`docs/so3c/experiments.md`, table E; so33 commit `87206e7`). The beams
   reduce the symmetry to rotations about the beam axis, the same residual
   symmetry LorentzNet and PELICAN get from their beam inputs.
2. **Beams: +0.0023 AUC** on the short probe protocol (0.98088 against 0.97859,
   one seed each; `docs/so3c/experiments.md`, table C). The largest gain from
   any change made after the message-passing model; the lab-frame pair features
   are next.
3. **Lab-frame pair features, newest and preliminary.** ParT's pair features
   (ln Δ, ln k_T, ln z, ln m²) written as Lorentz invariants of the momenta and
   the beams, so covariance stays exact:
   - +0.00051 AUC on the probe protocol (3 seeds against 6 control runs,
     t = 5.9);
   - +0.00064 at the K=32 long schedule (2 runs against 5, t = 6.8);
   - +192 parameters; no extra training time at the long schedule (1.00–1.01×
     the control; 1.04–1.13× on the probe protocol); no extra overfitting.
   Not yet run at K=64, the protocol of the headline numbers. Which of the four
   features carries the gain is not known.
4. **A reproducibility lesson** (useful for an ML audience). On Kaggle's T4 with
   torch 2.11 and `torch.compile`, the same seed and the same code in two
   kernels give different training runs: with equal batch order (checked by a
   hash), the epoch-1 validation losses were 4.5e-3 apart; and two full runs of
   the same seed ended up to 0.00018 apart in final AUC (the earlier run
   predates the hash, so its batch order is assumed), as far apart as different
   seeds. The likely cause is per-kernel compilation. Seed-by-seed "paired"
   comparisons can therefore mislead; results are read unpaired.

## Claims to avoid

- **State of the art.** It is not: 0.0027–0.0031 AUC behind PELICAN,
  LorentzNet and L-GATr (0.0029–0.0031 behind PELICAN and L-GATr), and below
  ParticleNet and ParT.
- **Parameter efficiency.** PELICAN's 1k-parameter model (0.9835 / 1145)
  matches the 60-epoch model, and its 3k model (0.9850 / 1494) beats the best
  single model and the ensemble.
- **"First" Lorentz-symmetric or bivector model.** LGN, LorentzNet and L-GATr
  are Lorentz-equivariant and PELICAN is Lorentz-invariant, and L-GATr's
  geometric algebra contains bivectors. The distinctive part is the so(3,ℂ)
  formulation with closed-form flows; check related work before calling it
  novel.
- **Beams as a novelty.** LorentzNet and PELICAN also feed beam particles
  (`docs/so3c/theory.md` §11).
- **The pair-feature gain at K=64.** Not measured yet.
- **Seed-paired statistics** on torch 2.11, or anything resting on one run
  presented as settled: the single-model numbers are one run each.

## Figures

The committed figures in `figures/` (PNG and PDF, with the CSV each was drawn
from) were last drawn on 2026-09-11 (so33 commit `e9b2184`) and moved here on
2026-09-18. None shows message passing with beams or anything later, and
several show the non-invariant `so3c_equivariant_set` ✗ beside
`so3c_invariant_set` and `eta_invariants` (`docs/so3c/experiments.md` §7). Treat
them as templates to redraw, not as current results.

| figure | shows | script |
|---|---|---|
| `pareto_params_vs_performance` | parameters against AUC and against rejection: SO3C's covariant model at K=64 (0.9772) and the matched Deep Sets baseline against the published taggers, including PELICAN's size sweep from `figures/pelican_scaling.csv`; no L-GATr | `benchmarks/figure_pareto.py` |
| `boost_robustness` | AUC against boost scale for the beamless models (message passing 13,862, covariant, invariant set) and the non-invariant `so3c_equivariant_set` ✗, which collapses to 0.44; the beams curves are not drawn | `benchmarks/figure_boost_robustness.py` |
| `k_robustness` | AUC and rejection against K = 4–32 (internal protocol, table G) for the three old models | `benchmarks/figure_k_robustness.py` |
| `convergence_val_acc` | validation accuracy per epoch, internal protocol, the three old models | `benchmarks/figure_convergence.py` |
| `binned_top_tagging_*` | performance binned in jet kinematics, the three old models | `benchmarks/figure_binned.py` |
| (not committed) `scaling_params_vs_performance.pdf` | channel scaling of the non-invariant `so3c_equivariant_set` (table D) | `benchmarks/figure_scaling.py` |

Visuals that do not exist yet and would help:
- the pareto plot redrawn with the current best (0.98392), the ensemble
  (0.98413) and L-GATr;
- the boost plot redrawn for the beams model, moved along and pinned (from
  `results_boost/`);
- the progress table above as a step chart;
- a schematic of the bivector lift and one covariant flow step;
- the pair-feature result (probe and long schedule against their controls);
- the reproducibility finding: `results_message/repr_screen/beame002/pairing_check`
  (two epochs) against `results_message/repr_screen/base1010`, seed 0, for the
  hash-verified divergence; `results_message/probe_screen/base` against
  `base1010` for the final AUC.

## Where the details live

- Mathematics: `docs/so3c/theory.md` (§1–§4 the algebra, the closed-form
  exponential and the flow; §5–§7 why the connection must transform and the
  covariant connection; §9 the bivector lift; §10 message passing; §11 beams;
  §12 the switches including the lab-frame features).
- Model and parameter counts: `docs/so3c/models.md`.
- Every experiment, table by table: `docs/so3c/experiments.md`.
- Status, verdicts and the current plan: `SO3C_STATUS.md`.
