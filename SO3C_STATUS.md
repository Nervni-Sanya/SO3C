# SO3C status

Last updated 2026-10-10. This is the single place to start from. Commit hashes point at the evidence for every number. Hashes of work from before the snapshot (2026-09-16) are commits in the **so33** repository this one was split from; hashes of later work, some marked **(SO3C)**, are commits in this repository; `git cat-file -t <hash>` tells which (see [Provenance](#provenance)).

## Scope and goal

- **Scope.** This repository is SO3C only: the complexified-SO(3) construction in `so3c/` and `benchmarks/so3c_models.py`, plus the harness that measures it. The SO(3,3) preprint and package are a different paper and live in the separate so33 repository; [`REMOVED.md`](REMOVED.md) lists what was taken out of this one. SO3C has no draft yet.
- **Goal.** State of the art on the Kasieczka top-tagging reference dataset, canonical splits (1.211M train / 403k val / 404k test). Not a repositioning to "construction plus methodology": that was proposed after the capacity sweep saturated and was overridden on 2026-09-12.
- **Targets**, verified against the source tables (`figures/literature_reference.csv`):

  | model | params | AUC | 1/ε_B at ε_S=0.3 |
  |---|---|---|---|
  | PELICAN (arXiv:2307.16506, table 1) | 208k | 0.9870 | 2250 ± 75 |
  | LorentzNet (arXiv:2201.08187, tables 1, 5) | 224k | 0.9868 | 2195 ± 173 |
  | PELICAN, size sweep (table 2) | 1k | 0.9835 | 1145 ± 74 |
  | PELICAN, size sweep (table 2) | 11k | 0.9858 | 1879 ± 103 |
  | L-GATr, trained from scratch (arXiv:2411.00446, table 2) | 1.1M | 0.9870 ± 0.0001 | 2240 ± 70 |

  Every published number above that band that we have found comes from fine-tuning after **JetClass pretraining**; the highest in the reference table is L-GATr fine-tuned (0.98793 / 2894 ± 84, same table). A different comparison class, not the target here.

## Current best

`so3c_message_set` with **beams + channels 8**, K=64, canonical protocol, Adam 3e-3 cosine, float32, 22834 parameters. **60 epochs is the best single model** (`f12cd65`); the 30-epoch runs below are the same configuration on the shorter schedule (`b72437f`).

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 | final val loss |
|---|---|---|---|---|
| **60 epochs, seed 0** (T4, `--compile`, 8.59 h) | **0.98363** | **1140.8** | 314.5 | **0.16276** |
| 30 epochs, mean of 3 seeds | 0.98326 ± 0.00014 | 1089 ± 75 | 291 ± 6 | 0.16318 / 0.16449 / 0.16455 |
| ensemble, 3 × 30 epochs | 0.98378 | 1160.4 | 316 | |
| **ensemble, 3 × 30 + 60 epochs** | **0.98396** | **1231.2** | 327.8 | |

+0.00037 AUC over the 30-epoch mean: 2.2 standard errors of one run against a three-seed mean (t on two degrees of freedom, p ≈ 0.15), or 1.8 σ of a difference between two single runs — suggestive, not conclusive from one seed. The validation loss is the cleaner signal: 0.16276 is below all three *complete* 30-epoch schedules. It was still falling at the end (−0.00038 over the last five epochs), but a cosine schedule always flattens there, so that part is weak evidence that more epochs would help. The rejection gain (+52) is inside the ±75 seed spread. Gap to PELICAN from the best ensemble: **0.0030 AUC, factor 1.83 in rejection**.

The 30-epoch seeds, for the record:

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 |
|---|---|---|---|
| seed 0 (P100) | 0.98326 | 1147.2 | 287.6 |
| seed 1 (P100) | 0.98340 | 1115.5 | 297.4 |
| seed 2 (T4, rewritten forward pass) | 0.98312 | 1004.5 | 287.2 |

**The rejection is much noisier than two seeds suggested.** Seeds 0 and 1 alone gave 1131 ± 22; the third seed landed at 1004.5 and the three-seed standard deviation is 75, that is 7%. The AUC is stable to 0.00014. Rejection at 30% signal efficiency is a tail statistic — a few hundred background jets decide it — so **a rejection gain under ~150 on a single seed is not evidence**, while +0.0004 AUC is about two standard deviations of a single-run difference. Seed 2 is not the rewrite's doing: its training curve matches seed 0's (final validation loss 0.16455 against 0.16449) and the two implementations agree to 1e-14 in float64.

- **Gap to PELICAN:** 0.0037 AUC and a factor 2.1 in rejection for a single 30-epoch model (from 0.0063 and 2.6 before beams); 0.0034 and 1.97 for the 60-epoch model; 0.0030 and 1.83 for the four-model ensemble.
- **The sobering comparison (verified):** PELICAN's own 1k-parameter model (0.9835 / 1145) matches this 22.8k-parameter model. The gap is architectural, not a matter of size.
- **Exact equivariance, measured** (float64, `87206e7`): with the beams transformed along with the jet, AUC stays 0.9684 at every boost scale (drop −0.0001); with the beams pinned to the lab it falls to 0.9332. That is exact covariance, the correct physics, and the same property PELICAN has.
- **Reproduce:** `notebooks/kaggle_k64_headline_seed2.ipynb` for a new seed (one session: seed 2 took 5.2 h without `--compile`, which the notebook has passed since, ~4.1 h); seeds 0 and 1 came from `notebooks/kaggle_beams_k64_seed{0,1}.ipynb` plus `kaggle_beams_k64_finish.ipynb` (they stopped at epoch 24 on the session cap and resumed). Seed 2 with its checkpoint and log is in `results_k64/headline/`; results, the epoch-30 checkpoints of seeds 0 and 1 and the full kernel logs are in `results_message/beams_k64/` (weights archived from Kaggle in `a76d837`; most other GPU runs keep their checkpoints in a `checkpoints/` folder beside the results). Ensembles: `python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64 results_k64/headline` for the three 30-epoch seeds (`results_k64/headline/ensemble_seeds012.json`); add `--allow-shared-seed` and `results_k64/headline60` for the four-model ensemble, whose members include two seed-0 runs (`results_k64/headline60/ensemble_seeds012_plus_60ep.json`).

## How we got here (canonical protocol)

| model | params | K | AUC | 1/ε_B@0.3 | commit |
|---|---|---|---|---|---|
| `eta_invariants` anchor | 4802 | 32 | 0.9478 | 49 | — |
| `so3c_invariant_set` (no flow) | 5122 | 32 | 0.9689 | 183 | — |
| `so3c_covariant_set` (1 covariant round) | 9078 | 32 | 0.9746 ± 0.0001 | 312 | `c3a98d0` |
| `so3c_covariant_set` | 9078 | 64 | 0.9772 ± 0.0001 | 638 ± 1 | `6b4e7b5` |
| `so3c_message_set` (3 rounds + scalar channel) | 13862 | 64 | 0.98073 ± 0.00006 | 850 ± 56 | `d77fb9c` |
| + beams + channels 8 | 22834 | 64 | 0.98333 ± 0.00010 | 1131 ± 22 | `b72437f` |

## Closed — measured, do not re-run without a reason

| question | result | commit |
|---|---|---|
| Connection built from invariants (`so3c_equivariant_set`) | **Not equivariant once trained**: 0.9424 falls to 0.4414 (below chance) at boost scale 3. Replaced by the covariant connection `z_a × Σ_b φ z_b` | `c3a98d0`, `f305ae4` |
| ODE solver instead of closed form (`so3c_interaction_set`) | 0.9735, 9.5× slower | `c3a98d0` |
| K for the covariant model | K=32/64/128 → 0.9746/0.9772/0.9781, rejection 312/638/639: saturates at 64 | `6b4e7b5` |
| **K beyond 64, by energy content** (raw 200-slot parquet, 100k test jets) | K=64 keeps 99.75% of jet energy on average (p1 95.3%); 82% of jets are complete and have exact jet mass. K=128 adds only the soft tail. Combined with the K sweep above and 19.3 h per seed even with kNN: **closed** | `1259ff7` |
| **Per-constituent mass m² as an input** | rounding noise: \|m²\|/E² median 4.6e-8, max 2.3e-7, 50% negative, log\|m²\| tracks log E² with slope 0.998. The constituents are massless | `1259ff7` |
| Channels on the broken model | flat, 0.9743–0.9745 across 22× parameters | — |
| kNN(16) graph at K=32 | −0.0011 AUC for 1% wall clock saved (2064 against 2082 s). Note 16 neighbours cover half the jet at K=32 and an eighth at K=128, so the penalty should grow, not shrink, with K | `1f01ddd` |
| Capacity **without** beams (K=32 probe) | flat: scalar_dim 24, hidden 256, channels 8/16, rounds 6 all within 0.0006 AUC of the anchor | `abd5660` |
| Scalar channel | load-bearing: −0.0025 AUC and −40% rejection without it | `abd5660` |
| Published LorentzNet/PELICAN recipe (AdamW wd 0.01, dropout 0.2, lr 1e-3, 35-epoch warm restarts) | **negative**: 0.97769 alone (below anchor 0.9786), and beams + recipe 0.98065 < beams alone 0.98088 | `837c567` |
| Learning rate | 3e-3 beats both 6e-3 (−0.0003) and 1e-3 | `c325950` |
| ~~Longer training~~ **REOPENED 2026-09-20** | The verdict rested on one probe-protocol row (35 vs 20 epochs on 400k jets, K=32: +0.00015 AUC, −14 rejection). Against it: validation loss is lowest at the **final** epoch in all four K=64 runs, and on the older 9k `so3c_equivariant_set` at K=32 on the full training set, 60 epochs beat 30 by +0.00024 AUC (6.4 σ) and +30 rejection (9.5%, 5.2 σ) — `results_60ep` against its three 30-epoch seeds, same batch, same learning rate, only the epoch count differing. Tested at K=64 since: 60 epochs gave 0.98363, +0.00037 over the 30-epoch mean, one seed (`f12cd65`) | `c325950`, `results_60ep` |
| Regularisation | dropout 0.05 + wd 1e-4: −0.0011 AUC, −133 rejection. The model **underfits** | `c325950` |
| Capacity **with** beams | channels 8: +0.0004 AUC, +67 rejection; channels 16: 0.98111 / 811 at 1.5× cost. One seed each, so neither difference is resolved (a difference between two single runs on this protocol spreads by ±0.00024, measured later) | `c325950`, `90f2ead` |
| K=128 wall clock | dense 602 s/epoch on 100k jets (60.8 h full protocol); kNN(16) 191 s (19.3 h), 3.15× faster | `90f2ead` |
| Time-axis reference particle (1,0,0,0), gap-analysis finding 23 | redundant with beams: b+ + b- = 2 (1,0,0,0), so <p, t> = (<p,b+> + <p,b->)/2 is already available from inputs the model has. Closed by argument, not run | 2026-09-13 |
| Two-seed ensemble | +0.0004 AUC; member scores correlate at 0.994 | `49e0c61`, `b72437f` |
| Epochs 25–30 vs 24 (session cap) | +0.00022 AUC | `b72437f` |
| **Jet multiplicity as an input** (true count, count clipped at 64, energy fraction in the leading 64) | Stacked on the two-seed ensemble's test scores, fit on one half of the test set and scored on the other: **+0.00002 AUC**; rejection +65, inside the ±60 spread between the two halves. The model already carries it. Diagnostic only (fits on test jets). **The rejection delta carries no information:** with the logit alone the boosted trees output ~300 distinct scores, which ties the tail, and a pure-noise feature "adds" +125 (`so3c_notes/gap_generic_ml.md` §8.4). Only the AUC delta counts | (SO3C) `benchmarks/diag_multiplicity_stacking.py`, `results_message/beams_k64/multiplicity_stacking.json` |
| **Rest-frame shape invariants (body order)**: tr S³, det S, sphericity eigenvalues, per-node triangles, linear energy-flow tensor of the raw lift | Stacked as the multiplicity row: −0.00001 to 0.00000 AUC on the 60-epoch model, ≤ +0.00003 on the ensemble, the size of a pure-noise control. Jet-level 3- and 4-body shapes add nothing | (SO3C) `so3c_notes/gap_generic_ml.md` §8.4, 2026-10-05 |
| **Flow-head L2 penalty** (1e-3·‖w_head‖² inside the loss, plain Adam) | Not binding: in live channels the data gradient is 47–405× the penalty's and the penalty is ≤ 2.5% of Adam's normaliser; the Hessian curvature implies ≤ 9% shrinkage. It pins only heads of collapsed channels. A paired 300-step fine-tune without it changes nothing. Round 0 is quiet because the generator is quadratic in the (small) state, not because of the penalty | (SO3C) `so3c_notes/gap_generic_ml.md` §2 |
| **Test-time augmentation** over the detector reflections (σ_x, σ_z, beam-slot swap; only 4 views are distinct) | +0.00002 ± 0.00002 AUC on 100k test jets, against +0.00034 for the 4-seed ensemble on the same jets | (SO3C) `so3c_notes/gap_generic_ml.md` §4 |
| **Width at K=64** (w64s24, seed 0, 30 epochs, paired with a control rerun on the same software) | 0.98319 / 1035.5 against 0.98311 / 1040.8: **+0.00008 AUC** for 1.64× the time (+0.00025 at the K=32 long schedule, one seed) | (SO3C) `results_k64/w64s24`, `results_k64/headline_t4` |
| **Edge LayerNorm** (`--edge-norm`, per-pair LayerNorm before the edge tanh; probe protocol, 3 seeds paired with a fresh torch-2.11 control) | **ΔAUC −0.00004 ± 0.00010** (t = −0.6); 1/ε_B@0.3 **−41.8 ± 7.8** on every seed; same GPU time. It halves the tanh saturation of rounds 1–2 (34/61% → 15/31% at seed 0) without moving AUC, so saturation is not the limit on this protocol | (SO3C) `results_message/probe_screen/`, `so3c_notes/gap_generic_ml.md` §9.2 |
| **Log knee** (`--log-knee`, asinh(e^κ x) with learnable κ on the bilinear invariants; same screen) | **ΔAUC +0.00000 ± 0.00008** (t = 0.08); 1/ε_B@0.3 +39.0 ± 30.1, positive on all three seeds (p ≈ 0.08 one-sided), suggestive only; κ barely moves from its initial 4 (3.2–4.4). Free (+7 parameters), so a possible passenger, not an arm | (SO3C) `results_message/probe_screen/`, `so3c_notes/gap_generic_ml.md` §9.3 |
| Best-validation checkpoint | Validation loss is lowest at the **final** epoch in both headline seeds and both beamless K=64 seeds (cosine schedule), so the best-validation checkpoint is the last one | (SO3C) 2026-09-19, from the stored histories |

**Lesson worth keeping:** a closed question can re-open when the inputs change. Capacity was flat without beams and looked as if it paid once beams supplied the information (one seed, not resolved); width did pay later, paired, at the long schedule. kNN was useless at K=32 and 3.15× faster at K=128.

## Open — candidates

Ranked literature and generic-ML candidates, every literature number checked against its paper's table: [`so3c_notes/gap_thesis.md`](so3c_notes/gap_thesis.md) and [`so3c_notes/gap_generic_ml.md`](so3c_notes/gap_generic_ml.md), with the 2026-10-06 audit's corrections applied. Their two measured leads were conditioning problems — round-0 inputs in asinh's linear range (about 95% of pair invariants below 0.1) and edge-tanh saturation in rounds 1–2 (61–84% of pre-activations beyond |x| > 2), with no normalisation layer in the network. Both were screened on 2026-10-05 and **neither moves AUC** (Closed table). The plan for the window opening 2026-10-10 is [`so3c_notes/plan_2026-10-10.md`](so3c_notes/plan_2026-10-10.md).

### Screened on 2026-09-20, none better than the base

The two defect fixes, the vector channel, the relative-norm edge, the pair latent, their combination, depth 6 and f_α were all measured on the probe protocol (see the screen table below). None improved on the base in that screen; the seed spread measured since (`3ca06bd`) shows the screen was noise (see the GPU-window sections below).

### Verified defects in the current model (free to fix)

Both are now switchable and default to the old behaviour: `--no-mass-input --no-self-edges` (22634 params on beams + channels 8, CPU cost 1.00x the headline, `6771158`). Screened on GPU on 2026-09-20: one seed, no resolved gain.

- **m² fed as noise.** `h_init` takes `asinh(m²)` per node, and three of the seven `_minkowski_stats` readout features are per-particle m² moments. All of it is rounding noise (see Closed).
- **Self-edges in the dense graph.** The dense path includes `a = b` in the scalar message sum and counts N, not N−1, in the denominator. The flow itself is unaffected, since `z_a × w_aa z_a = 0`. LorentzNet masks the self-edge; our sparse path already excludes it.

### Best-motivated architectural candidates

- **Rank-2 pair latent** with a reduced Eq2→2 aggregator basis (7 of PELICAN's 15). PELICAN carries a [B,N,N,C] state through its blocks; we collapse pairs to nodes every round. The verified table-2 comparison — 1k PELICAN parameters match our 22.8k — is consistent with this being the dominant carrier. Outcome likely bimodal: a real step toward 1400–1800 rejection, or under +0.0005 if our flow already supplies it. **Implemented** as `--pair-latent N`: 3680 parameters at C_p = 8 on beams + channels 8, CPU cost 1.56x the headline, 1.64x combined with the vector channel and both fixes; on a T4 1.55x per step, an estimated ~7.7 h per 30-epoch seed at K=64 (~5.8 h compiled). Exactly Lorentz covariant (1.7e-16) and permutation invariant (0.0), with the 7 aggregator maps checked for permutation equivariance on their own. Screened on GPU on 2026-09-20 and replicated over three seeds: +0.00012 AUC paired, not resolved. Agent estimate: screen at 20% data × 3 seeds ≈ 11 GPU-h.
- **A vector channel alongside the bivector.** Two reasons, one measured elsewhere and one structural:
  - arXiv:2606.21790 (abstract, verified) finds that in L-GATr "bivector channels are negligible for top-quark tagging while vector-like channels are dominant". The LLoCa (2505.20280) and slim L-GATr (2512.17011) abstracts do not address grades; their bodies are unchecked.
  - Our lift `z_a = bivec(p_a, P)` is **unchanged under p_a → p_a + λP**, so it discards each constituent's component along the jet axis (**correction, 2026-10-05:** for the massless constituents of this dataset p_a² = 0 fixes that component, and z_a·z_a = ⟨p_a,P⟩², M² = Σ⟨p_a,P⟩ and Re z_a·z_b recover every ⟨p_a,p_b⟩, so the lift loses no information; `so3c_notes/gap_thesis.md` §3). The scalar channel recovers ⟨p_a, P⟩ only as an invariant. A covariant vector v_a = p_a keeps it — the same kind of missing-information problem that beams fixed. **Implemented** as `--vector-channel`: 531 parameters on beams + channels 8, CPU cost 1.10x the headline, exactly covariant to 7e-15 with and without beams, and the vector update moves the logits by 7e-2 to 9e-2. Screened on GPU on 2026-09-20 and replicated over three seeds: −0.00002 AUC paired, bounded near zero.

### Cheap candidates (unverified gain)

| candidate | mechanism | note |
|---|---|---|
| N^α / N̄^α aggregation rescaling | sum-vs-mean semantics; our flow angle scales with Σ_b w_b z_b | ~0 runtime |
| Time-axis reference particle (1,0,0,0) | a third symmetry-breaking reference, as in L-GATr -- **redundant with beams** (b+ + b- = 2t; see Closed) | 1.03× cost |
| Softmax attention over w_b | normalised aggregation; needed for depth | ~1.05× |
| Lion optimiser, lr 3e-4, wd 0.2 | a different optimiser, not the ruled-out AdamW bundle | a training-limit control |
| JetClass pretraining | every published result above PELICAN that we have found is pretrained; the highest in the reference table is **L-GATr fine-tuned after JetClass pretraining**: AUC 0.98793 ± 0.00001, 1/ε_B 2894 ± 84 (arXiv:2411.00446 table 2, verified). That is a **different comparison class** and not our target — the same L-GATr *trained from scratch* on this dataset scores 0.9870 ± 0.0001 / 2240 ± 70 at 1.1M parameters, matching PELICAN rather than beating it | ≥ 25 GPU-h truncated; changes the comparison class |

Full agent text: `so3c_notes/gap_to_sota_2026-09-13.md`. Three of the workflow's six agents failed on the usage limit, so the findings there are unranked.

## GPU window 2026-10-10 → 2026-10-17 (in progress)

Plan: [`so3c_notes/plan_2026-10-10.md`](so3c_notes/plan_2026-10-10.md). Its code steps C0–C6 were not done before the window opened; C0 (integrating the gap-notes branch: results, notes with the audit corrections, and this status) is done, and the first run is the 60 → 120-epoch continuation of the K=64 headline (plan §2a).

## GPU window 2026-10-03 → 2026-10-10 (closed): regularising the wide model

**Run:** w128s32 with decoupled weight decay (`--optimizer adamw --weight-decay 0.002`) and `--keep-best`, seeds 0 and 1, long-schedule protocol (`c762616`). Verified before pushing: the notebooks equal the control notebooks plus exactly those flags; same initial weights and same batch order as the control under each seed; AdamW with zero decay reproduces Adam exactly, so the decay is the only knob. Pushed 03:10, 6.30 h and 7.41 h. Results in `results_long_seeds/w128s32wd2`.

**Result: not shown to help** (corrected by the 2026-10-06 audit; it was headed "a small, consistent negative"). Against the unregularised w128s32 runs of the same seeds — paired only if the batch order survived the torch 2.10 → 2.11 upgrade, see below:

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| control (Adam), final AUC | 0.98308 | 0.98270 | 0.98289 |
| wd 0.002, final AUC | 0.98298 (−0.00010) | 0.98259 (−0.00012) | **−0.00011** |
| wd 0.002, best-validation weights | 0.98304 (−0.00004), epoch 50 | 0.98260 (−0.00010), epoch 52 | −0.00007 |
| 1/ε_B@0.3, final | 1035.5 → 1035.5 | 943.5 → 874.1 | −35 |
| train − val gap at epoch 60 | 0.0120 → 0.0133 | 0.0103 → 0.0104 | not reduced |
| best validation loss | 0.16532 → 0.16537 | 0.16632 → 0.16724 | not improved |

Decoupled decay on every parameter (torch's AdamW also decays the identity-initialised channel mixing and the channel weights) does not regularise this model: the generalisation gap is unchanged and the training loss ends at least as low.

**What the `--keep-best` numbers show, which matters more than the decay itself:** the best-validation weights beat the final weights by only +0.00006 and +0.00001 AUC. The rise in validation *loss* after epoch 45–50 is therefore mostly calibration — the model growing overconfident — not lost ranking power. Calling the wide models "overfit-limited" (previous window) overstated it: their discrimination saturates rather than degrades, which is why a regulariser had little to recover. Rejection at 30% efficiency is also coarse at this level: ~195 of ~202k background jets pass the cut, so it moves in steps of about one jet (~0.5%).

### Width at K=64 over 30 epochs: no gain (`82235c4`, `8e76d33`, `4b91517`)

w64s24 and a control rerun of the headline configuration, both seed 0, K=64, 30 epochs, pushed together on the same software (T4, **torch 2.11**, compiled, `--keep-best`).

| K=64, 30 epochs, seed 0 | params | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 | best val loss (epoch) | train − val gap | h |
|---|---|---|---|---|---|---|---|
| control (`results_k64/headline_t4`) | 22834 | 0.98311 | 1040.8 | 287.6 | 0.16483 (30) | 0.0030 | 4.10 |
| w64s24 (`results_k64/w64s24`) | 60834 | 0.98319 | 1035.5 | 284.4 | 0.16448 (30) | 0.0039 | 6.72 |
| Δ, paired | 2.7× | **+0.00008** | −5 | −3 | −0.00035 | | |

Neither model overfits at 30 epochs (validation loss lowest at the final epoch, so `--keep-best` returns the final weights). At K=32 the width gain appeared on the **60-epoch** schedule; at K=64 on 30 epochs it is +0.00008. That is consistent with capacity needing the long schedule, so width at K=64 on 60 epochs is still untested — not refuted.

**Cross-software reproducibility.** The control reruns the original seed-0 headline (P100, torch 2.5.1, old forward pass, eager: 0.98326 / 1147.2) and lands at 0.98311 / 1040.8: −0.00014 AUC, −106 rejection, about one seed spread. Across that software jump two seed-0 runs behave like independent seeds.

**Kaggle moved from torch 2.10 to 2.11** between 2026-09-27 and 2026-10-03, unannounced. Every log was checked: the long screen (base, w64s24, w128s32), the probe replicates and headline seed 2 ran on 2.10; the weight-decay runs and today's K=64 pair on 2.11. So the weight-decay comparison crossed the upgrade. Whether a shared seed still gives the same batch order across 2.10 → 2.11 cannot be checked directly here (no CUDA locally), and the epoch-1 losses do not settle it: same-seed pairs across the upgrade differ by 0.0008–0.0015 in epoch-1 validation loss, but base seeds 0 and 2, different seeds on one version, differ by only 0.00008. So the weight-decay verdict is paired only if the batch order survived the upgrade; unpaired, −0.00011 is inside the ±0.00027 seed spread. **Trap:** record the torch version of every run (the GPU_SETUP line in each log) and check it before calling two runs paired.

### Conditioning screen (2026-10-05)

The two conditioning switches of the gap notes, on the probe protocol (K=32, 400k training jets, 20 epochs, `--compile`), three seeds per arm, paired seed by seed with a **fresh** control on the same build (T4, torch 2.11.0+cu128, pinned and asserted by the new setup cell; every result JSON now records `env`). Code `2303568`; results `results_message/probe_screen/` (9 runs, 4.16 GPU-h).

| arm | params | ΔAUC vs control | Δ 1/ε_B@0.3 |
|---|---|---|---|
| control | 22834 | 0.98112 ± 0.00002 (absolute) | 817.6 ± 11.4 (absolute) |
| `--edge-norm` | 22930 | −0.00004 ± 0.00010 | −41.8 ± 7.8 |
| `--log-knee` | 22841 | +0.00000 ± 0.00008 | +39.0 ± 30.1 |

Conditioning, the top-ranked cheap lead, is not where the gap is. A fresh control was the safe design; whether the torch-2.10 baseline would have paired is **not shown** — epoch-1 losses cannot tell same-seed from different-seed runs (audit #6).

**Budget:** ~13.9 h (regularised wide model) + ~11.2 h (the two K=64 runs) + 4.16 h (conditioning screen) ≈ **29.3 of 30 GPU-h** for this window (resets Saturday 2026-10-10).

**Where this leaves the SOTA question.** Best single model is still the 60-epoch K=64 headline, 0.98363; best ensemble 0.98396; PELICAN 0.9870. Measured levers so far: training length (+0.00037 at K=64), width (+0.00048 at K=32 on 60 epochs, +0.00008 at K=64 on 30 epochs), weight decay (−0.00011), early stopping (~0). The one width test left is K=64 on 60 epochs: w64s24 needs ~13.4 h (two sessions, so a finish kernel), w128s32 ~22 h, and it would need a same-stack 60-epoch control (~8.6 h): the 60-epoch headline ran on torch 2.10, and pairing across 2.10 → 2.11 is not established (audit #1). Even a full transfer of the K=32 width effect would leave about 0.0026 from the best ensemble (0.0029 from the best single model), so this is the last tuning-scale lever; closing the gap needs a structural change, not more of these.

## GPU window 2026-09-26 → 2026-10-03 (closed): width at the long schedule

**Question.** The model was undertrained at 30 epochs, so every capacity candidate was screened where it could not pay. Does capacity pay once the schedule is long enough — and specifically the edge width `act_hidden`, the one capacity axis never moved, through which every pairwise interaction passes? PELICAN's own size sweep goes 1k → 11k → 208k parameters for 0.9835 → 0.9858 → 0.9870; our 22.8k model sits at its 1k point, so a working capacity axis is the only lever in sight the size of the remaining 0.0030.

**Protocol: the long-schedule screen** (`build_long_seeds`, `4b3cba6`). K=32 on **all 1.211M training jets, 60 epochs**, compiled — it keeps the two things capacity needs and cuts only K, and it is the regime of the `results_60ep` evidence. Several multi-hour seeds can share a 12 h session: each run is given what remains of an 11.2 h budget as `--max-seconds`, so an overrun stops with a checkpoint instead of being killed. The on-card golden guard runs first.

**Measured cost** (`results_profile/t4_width_1007eaa`, T4, compiled, per epoch):

| config | params | K=32 | 60 epochs at K=32 | K=64 |
|---|---|---|---|---|
| base | 22834 | 168 s | ~3.0 h | 485 s |
| w32 (act_hidden 32) | 27922 | 178 s | ~3.1 h | 584 s |
| w64s24 (act_hidden 64, msg/scalar 24) | 60834 | 243 s | ~4.3 h | 815 s |
| w128s32 (act_hidden 128, msg/scalar 32) | 120202 | 361 s | ~6.4 h | 1270 s |

The same profile's rows for c16, pair8 and pair16w32 are eager, not compiled: dynamo's recompile cache filled after four configurations in one process and it fell back silently (compile time 0 s, logit difference exactly 0). Fixed in `c3478d1` with `torch._dynamo.reset()` per configuration; training runs build one model per process and are unaffected.

**Design** (`89e05b3`, `LONG_RUNS`): base, w64s24 and w128s32, two seeds each, one seed per kernel so the order can adapt — ≈27.4 GPU-h of the ~29.5 h left after the profile. base seed 0 and w128s32 seed 0 run first, in the two slots, so within ~6.5 h it is known whether width pays at the long schedule at all; if it clearly does, the remaining budget moves to the larger-K confirmation instead of more K=32 seeds.

### Final table of the window (`results_long_seeds/`)

| arm | params | seeds | AUC | 1/ε_B@0.3 | best val epoch | train − val gap at 60 | h per seed |
|---|---|---|---|---|---|---|---|
| base | 22834 | 0, 1, 2 | 0.98242 ± 0.00020 | 909 ± 32 | 56–60 | 0.005 | 3.4–3.5 |
| w64s24 | 60834 | 0 | 0.98286 | 930 | 50 | 0.009 | 4.4 |
| **w128s32** | 120202 | 0, 1 | **0.98289 ± 0.00027** | **989 ± 65** | 45–47 | 0.010–0.012 | 6.4–7.0 |

**w128s32 − base = +0.00048 AUC, +81 rejection (+9%).** Paired by seed (same batch order, see below): +0.00047 and +0.00049. Unpaired against three base seeds: t = 2.34, df 3, p ≈ 0.05 one-sided. The midpoint (w64s24, one seed) is +0.00025 over base seed 0, so AUC rises monotonically with width at roughly +0.0002 per doubling of parameters, while overfitting rises just as steadily.

**What it means for the SOTA gap.** Width is the first capacity axis that pays for this model, but at ~+0.0002 per doubling it cannot close 0.0030 on its own, and the wide models are **overfit-limited** — the next lever is regularisation for them, not more width. Also measured here: this protocol's base spread is ±0.00020 AUC, ±32 rejection over three seeds.

### First result: width pays, and the wide model overfits (`a43d4d7`)

| arm | params | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 | best val loss (epoch) | train − val gap at 60 | cost |
|---|---|---|---|---|---|---|---|
| base, seed 0 | 22834 | 0.98261 | 943.5 | 256.2 | 0.16650 (60) | 0.0049 | 3.52 h |
| **w128s32, seed 0** | 120202 | **0.98308** | **1035.5** | **277.7** | 0.16541 (45) | **0.0120** | 6.40 h |

+0.00047 AUC, +92 rejection (+10%), +21.5 at ε_S = 0.5. One seed each — roughly 2.4 σ if this protocol's spread matches K=64's — so suggestive, not established, but all three metrics agree, and it is the **first time added capacity has helped this model**.

The curves change the picture more than the numbers do. The wide model **overfits**: its validation loss bottoms at epoch 45 and rises to epoch 60 while its training loss keeps falling, and its train − val gap is 2.5× the base's. The small model underfit, which is exactly why dropout and weight decay only ever hurt it (`c325950`); the wide model is the first to reach the regime where regularisation can help. The earlier flat capacity sweeps were training-starved, not parameter-starved.

**Replicated, and the runs are paired** (`7d8aafe`):

| seed | base | w128s32 | paired Δ |
|---|---|---|---|
| 0 | 0.98261 | 0.98308 | +0.00047 |
| 1 | 0.98221 | 0.98270 | +0.00049 |
| mean ± sd | 0.98241 ± 0.00028 | 0.98289 ± 0.00027 | **+0.00048 ± 0.00001** |
| 1/ε_B@0.3 | 913 ± 44 | 989 ± 65 | +77 |

Unpaired, +0.00048 is only t = 1.76 (df 2). But **runs with the same seed see the same batch order across architectures**: `train_classifier` calls `torch.manual_seed(cfg.seed)` after the model is built, so model initialisation, whatever RNG it consumes, cannot shift the shuffle; checked directly by capturing the permutations of two different architectures under seed 1 (identical). The seed-to-seed swing of ~0.0004 is therefore data-order noise **shared** between the arms, and the paired differences agree to ±0.00001 (paired t ≈ 48, df 1, p ≈ 0.007 one-sided). Two pairs make that variance estimate fragile, so the honest reading is *strong, pending more pairs* — but the unpaired test badly understates it. The commit message of the w128s32 seed-1 result said a shared seed does not pair runs; that was wrong.

**Methodological consequence:** every comparison in this harness should be designed and analysed paired by seed. At this protocol the data-order component (~0.0004) is an order of magnitude larger than the architecture effect's own scatter, so pairing is worth far more than extra seeds.

**Cost correction:** real runs are 1.06× (w128s32) to 1.26× (base) the synthetic profile — fixed per-epoch overheads weigh more on the small model. The full two-seed curve no longer fits; w128s32 seed 1 runs now to replicate, base seed 1 follows w64s24 seed 0, and w64s24 seed 1 is the likely cut.

**Next window (opens Saturday 2026-10-03, 30 GPU-h).**

1. **Regularise the wide model, paired against the runs already made.** Because a shared seed gives the same batch order, w128s32 with a regulariser at seeds 0 and 1 is compared *paired* with the w128s32 seed 0 and 1 runs of this window, which cost nothing more. Two runs ≈ 13.5 h. The regulariser: decoupled weight decay (`--optimizer adamw --weight-decay`); PELICAN uses 0.005, and with lr 3e-3 over 60 cosine epochs that shrinks an untrained weight by exp(−0.005 × 426) ≈ 0.12, so 0.002 (≈ 0.43) is the milder first value. Every run carries **`--keep-best`** (`59817e1`), so early stopping on validation loss is measured on the same runs for free.
2. **Then K=64.** The wide model at K=64 costs 1270 s per epoch compiled — ~22 h for 60 epochs, two sessions — and w64s24 815 s (~14.4 h). With the overfitting at epoch 45–50, a 45-epoch schedule with `--keep-best` is the cheaper first K=64 run. Worth doing only once step 1 says how to regularise it.

## GPU window 2026-09-19 → 2026-09-26 (closed)

### Forward pass is 2× faster (SO3C `2a3b344`, `47ec4ad`)

A profile on the card (`benchmarks/profile_gpu.py`, notebook `kaggle_gpu_profile.ipynb`) showed a third of the step was bookkeeping, not model: the concatenation of the (B, K, K, 64) edge input from broadcast node features (13%), batched 3×3 complex matmuls forming and applying exp(−T[a]×) (11%; cuBLAS pads each to a 32×32 tile), and permute copies and fills of complex pair tensors. Rewritten as the **same function**: Rodrigues applied to the vector (`so3c.algebra.expm_so3c_apply`), the first edge layer and `pair_mix` evaluated block by block with per-node shares broadcast, pair invariants in real arithmetic, readout batched over channels.

| K=64, batch 256, Tesla T4, per-epoch estimate | before | after | |
|---|---|---|---|
| beams + channels 8 (headline) | 1219 s | **589 s** | 2.07× |
| + vector channel | 1457 s | 738 s | 1.97× |
| + pair latent 8 | 1894 s | 918 s | 2.06× |
| headline at K=32 | 389 s | 208 s | 1.87× |

Peak memory −38% (headline). Profiles: `results_profile/t4_16302b1`, `t4_2a3b344`, `t4_47ec4ad`.

**Guard.** Two tests, 16 switch combinations with platform-stable weights and every zero-init head excited: `test_message_set_matches_reference` runs the pre-rewrite implementation (16302b1, shipped as `tests/data/ref_so3c_models_16302b1.py`) beside the current one on the same weights and device; `test_message_set_matches_golden` compares with logits and gradient fingerprints stored from that implementation. Both hold to <5e-14 in float64, and mutations fail them wherever they can act (edge rewrite 15/16, pair_mix 4/4, flow time 16/16). On the trained headline checkpoint and 4000 real test jets, old and new agree to 2e-10 in float64; in float32 the new path is closer to float64 than the old one (max logit error 0.019 vs 0.128). K=64 notebooks now run `test_message_set_matches_golden_cuda` on the card first and require "1 passed".

### Gap screen: every candidate at or below the base

Seven rows, K=32 probe protocol (400k training jets, 20 epochs, seed 0, beams + channels 8), T4, old forward pass, against the base 0.98128 / 834.4 (`c325950`). Results in `results_message/gap_screen`, ~9.9 GPU-h.

| row | params | AUC | ΔAUC | 1/ε_B@0.3 | Δ |
|---|---|---|---|---|---|
| base (c8) | 22834 | 0.98128 | — | 834.4 | — |
| pair latent 8 | 26514 | 0.98124 | −0.00004 | 827.5 | −7 |
| combo (vector + pair8 + fixes) | 26861 | 0.98115 | −0.00013 | 814.2 | −20 |
| vector channel | 23365 | 0.98109 | −0.00019 | 817.5 | −17 |
| f_α, n=3 | 24373 | 0.98105 | −0.00023 | 782.6 | −52 |
| fixes (no m², no self-edges) | 22634 | 0.98097 | −0.00031 | 739.6 | −95 |
| rounds 6 | 28378 | 0.98092 | −0.00036 | 721.1 | −113 |
| relative-norm edge | 23602 | 0.98084 | −0.00044 | 788.7 | −46 |

Not one improves on the base, and the whole table spans 0.0004 AUC. Note that even removing verified noise (`fixes`: the per-constituent m², which is rounding noise, and the self-edge) costs 0.0003 — a sign that differences this small are dominated by optimisation variance rather than by the inputs.

**The regime limits what this screen could ever have shown.** It ran at 20 epochs on 400k jets. Candidates that *add capacity* — the pair latent, a wider edge — cannot pay there: more parameters need more data and more training before they earn their keep. A null result in that regime does not transfer to 30–60 epochs on 1.2M jets, so none of these candidates is closed on the strength of this table. What the screen does establish is that none of them is a large, regime-independent win.

**Measured (`3ca06bd`): the table above is noise.** Three seeds of the base on exactly this protocol give **0.98106 ± 0.00017 AUC and 806 ± 58 rejection** (`results_message/probe_seeds/base`, new forward pass with `--compile`; seeds score 0.98125 / 0.98092 / 0.98100).

- The screen's reference was a **high seed**: its base row, 0.98128, sits +0.00022 above that three-seed mean, about 1.3 σ.
- A single-run difference has spread √2 × 0.00017 = 0.00024, so every candidate delta is inside noise: pair8 0.2 σ, combo 0.5 σ, vector 0.8 σ, f_α 1.0 σ, fixes 1.3 σ, rounds 6 1.5 σ, relnorm 1.8 σ. **The screen neither closes nor supports any candidate**; it only rules out a large, regime-independent win.
- The rewrite and `--compile` do not shift results: base seed 0 scores 0.98125 against 0.98128 for the old eager code, a difference of 0.00003.

**Three arms re-measured properly** (`fb8d80a`, `9d3a123`): three seeds each, paired seed by seed, same code.

| arm | AUC | 1/ε_B@0.3 | paired Δ AUC vs base | h/seed |
|---|---|---|---|---|
| base | 0.98106 ± 0.00017 | 806 ± 58 | — | 0.45 |
| pair latent 8 | **0.98118 ± 0.00004** | 835 ± 27 | **+0.00012** (t = 1.42) | 0.59 |
| vector channel | 0.98104 ± 0.00012 | 805 ± 24 | −0.00002 (t = −0.59) | 0.48 |

- **Pair latent:** the screen's −0.00004 becomes +0.00012 once it is paired against three base seeds instead of one high one. Not significant (p ≈ 0.14 one-sided), but the sign is reversed and its run-to-run spread is four times smaller than the base's. The one candidate worth carrying into the long-schedule regime.
- **Vector channel:** paired differences with sd 0.00006, so it is bounded to |Δ| < ~0.00015 at 95% — a real negative, and much sharper than the screen's single seed. Feeding ⟨p_a, p_b⟩ directly, which the bivector lift exposes only through combinations with ⟨p_a, P⟩ and ⟨P, P⟩, does not pay. The regime excuse is weak here: it adds information, not parameters.

**Rule for future screens on this protocol:** one seed per row resolves nothing below about 0.0005 AUC. Compare against a multi-seed base, not a single row, and budget three seeds per arm (about 1.4 GPU-h at 0.45 h per row) — or screen where the effect is expected to be larger.

### The capacity axis never tested: the edge width

Every pairwise interaction in the model passes through `act_hidden = 16` tanh units: `edge[r] = Linear(edge_in → 16) + Tanh`, and the flow weights, the messages and the pair-latent update are all read out of those 16 numbers. The capacity sweep varied the complex channels, `scalar_dim` and `hidden` — the **readout** width — but never `act_hidden`. PELICAN's pair state is wide (tens of channels carried through every block), which is exactly the axis we never moved.

| configuration | params | × base |
|---|---|---|
| base (act_hidden 16, msg 8, scalar 8) | 22834 | 1.00 |
| act_hidden 32 | 27922 | 1.22 |
| act_hidden 64, msg 24, scalar 24 | 60834 | 2.66 |
| pair latent 16 + act_hidden 32 | 39506 | 1.73 |
| pair latent 16 + act_hidden 64 + msg/scalar 24 | 76002 | 3.33 |

Still far below PELICAN's 208k, and the first capacity axis with a mechanism behind it rather than a general "more parameters". The three probe kernels are **built and held** (`8b3cde6`): running them at 20 epochs on 400k jets would repeat exactly the regime that cannot show a capacity gain. They belong at the long schedule on the full training set, after the 60-epoch baseline below says whether that regime is where the model actually lives.

### 60 epochs at K=64: the model was undertrained (`f12cd65`)

Done, 8.59 h on a T4 at 515 s per epoch with `--compile` (predicted ~495). Result in Current best: **0.98363 / 1140.8**, +0.00037 AUC over the 30-epoch mean, final validation loss below all three complete 30-epoch schedules and still falling. The epoch-by-epoch curve is in `results_k64/headline60`.

The evidence that motivated it (`results_60ep`, the older 9k model at K=32 on the full training set) paid 6 σ in AUC and 5 σ in rejection. That comparison is between two *complete* schedules, not a mid-run snapshot: the 60-epoch run is behind at epoch 30 (validation loss 0.19814, its learning rate still high) and finishes at 0.19639, against 0.19705 for the complete 30-epoch schedule, with validation accuracy +0.00027 over three 30-epoch seeds. Worth stating plainly: "lowest validation loss at the final epoch" is weak evidence on its own, because a cosine schedule anneals to zero and always flattens at the end — the 60-epoch run is the actual test. It was: every capacity candidate has therefore been screened in a regime where it could not pay, and the screen has to be redone at the longer schedule.

**Next window (opens Saturday 2026-09-26, 30 GPU-h).** Recommended order: a second and third 60-epoch seed of the headline (8.6 h each) to turn +0.00037 into a measured mean, and the pair latent at 60 epochs (~11.7 h at 702 s per epoch), which is the one candidate whose paired probe came out positive. That is 29 h for three runs, so it is one or the other, not both — or a cheaper long-schedule screening protocol first: K=32 on the **full** training set at 60 epochs costs ~2.8 h per seed, the same regime as the `results_60ep` evidence, which would let candidates be screened three-seeded for ~8.4 h per arm.

### Sparse graph now carries beams (`829c4c0`)

The dense-only restriction on beams was conservative, not structural; it is lifted, with the two beam nodes forced into every node's neighbourhood (a fixed, frame-independent index set) so a ranked list cannot discard the only lab-frame information the model has. Cost: k + 2 edges per node. This unlocks K=128 with kNN and beams at roughly 7.7 h per 30-epoch seed, but that experiment is **not** scheduled: its case rests on +0.0009 AUC with rejection flat on the 9k covariant model, and the kNN penalty should grow with K.

### Budget

GPU used this window: ~9.2 h (screen) + 5.2 h (headline seed 2) + ~0.7 h (profiles, diagnostics, two guard failures) + ~3.1 h (base and pair8 replicates) + 8.6 h (60 epochs) ≈ 26.8 h of 30. The last ~1.5 h went to the vector-channel replicates (`nsanya/so3c-probe-seeds-vector`). The width probes are built and held for the next window. A 30-epoch K=64 seed with `--compile` was estimated at ~3.9 h and a 60-epoch one at ~8.3 h (measured since: 4.10 h and 8.59 h); a probe row costs 0.4–0.9 h.

## Infrastructure traps

| trap | what happened | fix |
|---|---|---|
| Kaggle derives the kernel slug from the **title** | polling the wrong slug returns HTTP 403 "Permission kernels.get denied", which reads like auth | `push` prints the real slug |
| Duplicate kernel titles | HTTP 409 | unique titles |
| P100 is sm_60 | preinstalled torch lacks the arch; every CUDA op fails | `GPU_SETUP` cell reinstalls torch 2.5.1 (~150 s) |
| Proxy `127.0.0.1:10808` is dead | every API call refused | run the Kaggle client with `HTTPS_PROXY` / `HTTP_PROXY` unset |
| Weekly GPU quota resets **Saturday** | a Monday-based running total said "spent" when ~24 h were available | test by pushing, not by arithmetic |
| Kernel output (including the log) is unavailable until the kernel finishes | cannot read per-epoch cost mid-run | budget from completed runs |
| A new session starts with an empty `/kaggle/working` | `--resume` alone restarts from epoch 1 | finish kernels mount the earlier run via `kernel_sources` and copy the checkpoint (`build_*_finish`) |
| `max_seconds` compared the cumulative clock | a run resumed past the cap trained one epoch per session, forever | per-session cap (`c030d17`) with a regression test |
| Code bundle ships only `CODE_PACKAGES` | the on-card pytest guard found no tests and stopped the session | `tests` added (`a4a0c61`); replay the bundle locally before pushing |
| CUDA-only resume bug (RNG state on GPU) | invisible to CPU tests | on-card `pytest -k cuda_resume` guard must report "1 passed" |
| Eval OOM at large K | the **eval chunk** (B, K, K, C), not the training batch | `--eval-chunk-size` 2048 at K=32, 1024 at K=64, 256 at K=128 |
| Result filenames carry no K or variant | K-sweep runs overwrote each other; a beams boost run would have overwritten the beamless one | one results directory per variant; `--tag` in the boost script; the figure groups by model **and** tag |
| Session cap vs cost | beams + channels 8 at K=64 is 1272 s/epoch = 10.6 h; runs stop at epoch 24 with `--max-seconds 30000` | plan a finish kernel for any run over ~8 h |
| Cost extrapolated from a cheaper setting | K=32 ratio 3.05× became 4.7× at K=64; the kNN saving and the K slope also missed | measure at the target setting before committing a window |
| float32 in the boost diagnostic | fakes a 0.023 AUC symmetry break at boost scale 2 | `run_boost_robustness.py` defaults to float64 |
| Beams change what "invariant" means | with fixed beams the output must change | measure with the beams moved along with the jet |
| Zero-initialised connection heads | equivariance tests pass vacuously on the identity flow | tests excite the heads and assert the logits move |
| Local npz cache stores only K=32 | local CPU checks cannot see past 32 constituents (verified: shape `(n, 32, 4)`) | K=64/128 data exists only on Kaggle (`nsanya/so3c-data-prep-k-128`); the raw 200-slot parquet is local in `data/toptagging/` |
| Bash tool collapses `\\` to `\` | `\times` became TAB + "imes" in a LaTeX edit | write files containing backslashes with the Write tool, or build them with `chr(92)` |
| Large subagent workflows | 45 agents: 38 failed on the usage limit; 6 agents: 3 failed | keep workflows small; verify bounded claims directly |
| Unverified agent output | a findings file said "zero runtime" for f_α and cited abstracts that do not make the claim | verify before building on it; record the verification |
| Separate commands in a Bash script keep running after one fails | a status-doc edit raised and exited non-zero, and the `git commit` on the next line ran anyway (`6771158` has the code but not the doc update it was meant to carry) | chain every dependent step with `&&`, or `set -e` |
| Kaggle replaces a kernel's output when it is pushed again | trained weights and full logs of ~40 GPU-h of runs existed only in kernel outputs until `a76d837` | fetch checkpoints and logs into the results directory as soon as a run finishes |
| Kaggle may hand out a **Tesla T4** (Python 3.12; torch 2.10 preinstalled, 2.11 by 2026-10-03) instead of the P100 | all three profile kernels on 2026-09-19 got a T4; timings do not transfer between cards blindly | record the device with every timing (the nvidia-smi cell, `summary.json`) |
| Kaggle's API has **no cancel** (the status enum has CANCEL_REQUESTED, set only from the web UI) | the gap screen was already running on the old, 2× slower forward pass when the speedup landed | push expensive kernels only once the code they depend on is final; a cancel needs the user in the web UI |
| Bash collapses backslashes inside notebook-builder cells too | an escaped newline in a generated `print` became a real line break, and the profile kernel's last cell died with a SyntaxError (its data was already saved) | write builder fragments with the Write tool or use `chr(10)`; `ast.parse` every generated code cell before pushing |
| A root-level ignore written as `data/` also matches `tests/data/` | the golden fixture could not be added | anchor root-only ignores: `/data/` (SO3C `2a3b344`) |
| **Exactly two** Kaggle GPU sessions at once | a third push returns HTTP 200 with "Maximum batch GPU session count of 2 reached." and is refused, not queued; the client printed it as "pushed" | use both slots, push the next kernel when one frees; `kaggle_client push` exits 2 on a refusal (SO3C `8d7dff7`) |
| A numerical fixture written on one platform, checked on another | the on-card guard failed at 4e-8 on every case; a diagnostic kernel showed even the **old** code on Kaggle's CPU was 4e-8 off, while new vs old on the T4 agreed to 3e-15 (`results_profile/t4_cuda_diag_3a3b10b.txt`). `nn.Linear` initialises in float32 before `.to(float64)`, and float32 init differs at the ulp level between torch versions | draw fixture parameters and inputs from numpy's legacy RandomState straight into float64, and compare against the shipped reference implementation on the same device (SO3C `89d67f3`) |
| Kaggle's T4 image moved from torch 2.10 to 2.11 between windows (seen 2026-10-04) | seed pairing assumes the same CUDA `randperm` stream, unverified across versions; the probe and long-schedule baselines are on 2.10 | screen notebooks pin and assert the build (`GPU_SETUP_T4_PINNED`, `2303568`); every result JSON records `env`; rerun controls on the same stack rather than reuse old ones |
| `kaggle_client push` asks for confirmation on stdin | in a non-interactive shell it died with EOFError after printing "about to push" | pass `--yes` once the push is approved |
| The embedded bundle's sha256 changes on every build | gzip stamps the compression time, so the hash names a blob, not a code state | to tie a notebook to a commit, extract its bundle and compare file contents with the tree |
| `.double()` on a complex tensor | silently drops the imaginary part (a warning only); a read-only diagnostic briefly reported nonsense jet-frame norms | cast complex tensors with `.to(torch.complex128)` |
| A Windows Python given a Git Bash `/tmp/...` path | resolves it to `C:\tmp\...`, not the scratch directory | hand Python Windows paths (`cygpath -w`) |
| A local smoke run with `--canonical-splits` | evaluates all 807k validation and test jets on CPU; it did not finish in 15 min | replay on the internal split with `--max-samples`; the canonical loader is proven on Kaggle |
| Kaggle's image has a regular top-level `tests` package in site-packages | while our `tests/` was a namespace package, `tests.message_golden` failed to import on the card and the golden guard stopped headline seed 2 after ~5 min (log in `results_k64/headline/`) | `tests/__init__.py` (SO3C `4f397f4`); replay bundles with a fake shadowing `tests` package on PYTHONPATH |

## Compute and quota facts

| configuration | seconds per epoch, full train set (P100 unless marked) | 30 epochs |
|---|---|---|
| `so3c_covariant_set`, K=64 | ~199 | 1.66 h |
| `so3c_message_set`, K=64 | ~930 | 7.75 h |
| + beams + channels 8, K=64 | 1272 | 10.6 h |
| + beams + channels 8, K=64, **rewritten forward pass, T4** (measured, seed 2) | 621 | 5.2 h |
| + beams + channels 8, K=64, **rewritten forward pass, `--compile`, T4** (measured, control rerun) | 492 | 4.1 h |
| K=128 dense (from 100k jets) | ~7290 | 60.8 h |
| K=128 kNN(16) (from 100k jets) | ~2313 | 19.3 h |

- The K=32 probe protocol (400k train jets, 20 epochs) costs 0.6–1.2 h per row.
- Quota is 30 GPU-h per week, resetting on Saturday. The current window is [2026-10-03 → 2026-10-10](#gpu-window-2026-10-03--2026-10-10-in-progress-regularising-the-wide-model).
- Rows not marked T4 are P100 wall clock on the old forward pass. Since `47ec4ad` the per-epoch cost is about half; the T4 profiles are in the 2026-09-19 GPU-window section and in `docs/so3c/experiments.md` (table O).

## Provenance

- This repository is a **snapshot** of the so33 repository's branch `feature/so3c-complexification`, taken at its tip `d360acc` (2026-09-13) and pushed as a single root commit, `6f79e70`, tagged `pre-so33-removal`. It shares no history with so33.
- **Hashes of commits made before the snapshot refer to the so33 repository**, not to a commit reachable from here: every hash in the Closed table, and those in the commit messages inherited from before the snapshot. Hashes of later work are commits in this repository; `git cat-file -t <hash>` tells which.
- Before the bench is deleted from so33, **tag `d360acc` there** (for example `so3c-snapshot-2026-09-13`), so those hashes stay reachable after that repository is pruned.
- SO(3,3)-only content was removed here on 2026-09-18 in four commits: `e322d9e`, `f3f1568`, `09aa178` and the packaging/documentation commit. [`REMOVED.md`](REMOVED.md) lists what went, what was kept on purpose and what changed as a result.
- so33 keeps its copy of this bench for now; it remains the home of the SO(3,3) paper, its Zenodo DOI and its `CITATION.cff`.
