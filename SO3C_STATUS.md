# SO3C status

Last updated 2026-10-11. This is the single place to start from. Commit hashes point at the evidence for every number. Hashes of work from before the snapshot (2026-09-16) are commits in the **so33** repository this one was split from; hashes of later work, some marked **(SO3C)**, are commits in this repository; `git cat-file -t <hash>` tells which (see [Provenance](#provenance)).

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

`so3c_message_set` with **beams + channels 8**, K=64, canonical protocol, Adam 3e-3 cosine, float32, 22834 parameters. **The best single model is the 60-epoch run continued to 120 epochs** (2026-10-10, `results_k64/headline120`). The 60-epoch run (`f12cd65`) and the 30-epoch runs below are the same configuration on shorter schedules (`b72437f`).

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 | final val loss |
|---|---|---|---|---|
| **120 epochs, seed 0** (the 60-epoch run continued, cosine re-anchored; T4, torch 2.11, `--compile`, +7.91 h) | **0.98392** | **1201.9** | 321.5 | **0.16184** |
| 60 epochs, seed 0 (T4, `--compile`, 8.59 h) | 0.98363 | 1140.8 | 314.5 | 0.16276 |
| 30 epochs, mean of 3 seeds | 0.98326 ± 0.00014 | 1089 ± 75 | 291 ± 6 | 0.16318 / 0.16449 / 0.16455 |
| ensemble, 3 × 30 epochs | 0.98378 | 1160.4 | 316 | |
| ensemble, 3 × 30 + 60 epochs | 0.98396 | 1231.2 | 327.8 | |
| ensemble, 3 × 30 + 120 epochs | 0.98409 | 1254.1 | 339.4 | |
| **ensemble, 3 × 30 + 60 + 120 epochs** (the 60 and the 120 are two snapshots of one run) | **0.98413** | **1269.9** | 339.9 | |

The 60-epoch run: +0.00037 AUC over the 30-epoch mean: 2.2 standard errors of one run against a three-seed mean (t on two degrees of freedom, p ≈ 0.15), or 1.8 σ of a difference between two single runs — suggestive, not conclusive from one seed. The validation loss is the cleaner signal: 0.16276 is below all three *complete* 30-epoch schedules. It was still falling at the end (−0.00038 over the last five epochs), but a cosine schedule always flattens there, so that part is weak evidence that more epochs would help. The rejection gain (+52) is inside the ±75 seed spread. The 120-epoch continuation is +0.00066 over the 30-epoch mean (0.98392 against 0.98326). Gap to PELICAN from the best ensemble (3 × 30 + 60 + 120 epochs): **0.0029 AUC, factor 1.77 in rejection** (0.0030 and 1.83 for the four-model one).

The 30-epoch seeds, for the record:

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 |
|---|---|---|---|
| seed 0 (P100) | 0.98326 | 1147.2 | 287.6 |
| seed 1 (P100) | 0.98340 | 1115.5 | 297.4 |
| seed 2 (T4, rewritten forward pass) | 0.98312 | 1004.5 | 287.2 |

**The rejection is much noisier than two seeds suggested.** Seeds 0 and 1 alone gave 1131 ± 22; the third seed landed at 1004.5 and the three-seed standard deviation is 75, that is 7%. The AUC is stable to 0.00014. Rejection at 30% signal efficiency is a tail statistic — a few hundred background jets decide it — so **a rejection gain under ~150 on a single seed is not evidence**, while +0.0004 AUC is about two standard deviations of a single-run difference. Seed 2 is not the rewrite's doing: its training curve matches seed 0's (final validation loss 0.16455 against 0.16449) and the two implementations agree to 1e-14 in float64.

- **Gap to PELICAN:** 0.0037 AUC and a factor 2.1 in rejection for a single 30-epoch model (from 0.0063 and 2.6 before beams); 0.0034 and 1.97 for the 60-epoch model; 0.0031 and 1.87 for the 120-epoch model; 0.0029 and 1.77 for the five-model ensemble.
- **The sobering comparison (verified):** PELICAN's own 1k-parameter model (0.9835 / 1145) (± 74) matches the 60-epoch model and is only 0.0004 AUC below the 120-epoch continuation (0.98392 / 1202), at 1/23 of the parameters. The gap is architectural, not a matter of size.
- **Exact equivariance, measured** (float64, `87206e7`): with the beams transformed along with the jet, AUC stays 0.9684 at every boost scale (drop −0.0001); with the beams pinned to the lab it falls to 0.9332. That is exact covariance, the correct physics, and the same property PELICAN has.
- **Reproduce:** `notebooks/kaggle_k64_headline_seed2.ipynb` for a new seed (one session: seed 2 took 5.2 h without `--compile`, which the notebook has passed since, ~4.1 h); seeds 0 and 1 came from `notebooks/kaggle_beams_k64_seed{0,1}.ipynb` plus `kaggle_beams_k64_finish.ipynb` (they stopped at epoch 24 on the session cap and resumed). Seed 2 with its checkpoint and log is in `results_k64/headline/`; results, the epoch-30 checkpoints of seeds 0 and 1 and the full kernel logs are in `results_message/beams_k64/` (weights archived from Kaggle in `a76d837`; most other GPU runs keep their checkpoints in a `checkpoints/` folder beside the results). Ensembles: `python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64 results_k64/headline` for the three 30-epoch seeds (`results_k64/headline/ensemble_seeds012.json`); add `--allow-shared-seed` and `results_k64/headline60` for the four-model ensemble, whose members include two seed-0 runs (`results_k64/headline60/ensemble_seeds012_plus_60ep.json`), and `results_k64/headline120` as well for the five-model one (`results_k64/headline120/ensemble_seeds012_plus_60ep_120ep.json`; without the 60: `ensemble_seeds012_plus_120ep.json`). The 120-epoch run: `notebooks/kaggle_k64_headline120_seed0.ipynb` (`acfb88e`), which embeds the epoch-60 checkpoint; result, checkpoint and log in `results_k64/headline120/`.

## How we got here (canonical protocol)

| model | params | K | AUC | 1/ε_B@0.3 | commit |
|---|---|---|---|---|---|
| `eta_invariants` anchor | 4802 | 32 | 0.9478 | 49 | — |
| `so3c_invariant_set` (no flow) | 5506 (5122 since six identically-zero features were removed) | 32 | 0.9689 | 183 | — |
| `so3c_covariant_set` (1 covariant round) | 9078 | 32 | 0.9746 ± 0.0001 | 312 | `c3a98d0` |
| `so3c_covariant_set` | 9078 | 64 | 0.9772 ± 0.0001 | 638 ± 1 | `6b4e7b5` |
| `so3c_message_set` (3 rounds + scalar channel) | 13862 | 64 | 0.98072 ± 0.00006 | 850 ± 56 | `d77fb9c` |
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
| ~~Longer training~~ **REOPENED 2026-09-20** | The verdict rested on one probe-protocol row (35 vs 20 epochs on 400k jets, K=32: +0.00015 AUC, −14 rejection). Against it: validation loss is lowest at the **final** epoch in all four K=64 runs, and on the older 9k `so3c_equivariant_set` at K=32 on the full training set, 60 epochs beat 30 by +0.00024 AUC (6.4 σ) and +30 rejection (9.5%, 5.2 σ) — `results_60ep` against its three 30-epoch seeds, same batch, same learning rate, only the epoch count differing. Tested at K=64 since: 60 epochs gave 0.98363, +0.00037 over the 30-epoch mean, one seed (`f12cd65`); continued to 120 epochs (2026-10-10, warm restart at lr 1.5e-3): +0.00029 AUC, +61 rejection, final val loss −0.00092 against its own epoch 60, unresolved by plan §2's rule (needs ≥ +0.0003), the new best single model (`results_k64/headline120`) | `c325950`, `results_60ep` |
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
| **Width at K=64** (w64s24, seed 0, 30 epochs, against a same-seed control rerun on the same software; one run each, unpaired) | 0.98319 / 1035.5 against 0.98311 / 1040.8: **+0.00008 AUC** for 1.64× the time (+0.00025 at the K=32 long schedule, one seed) | (SO3C) `results_k64/w64s24`, `results_k64/headline_t4` |
| **Edge LayerNorm** (`--edge-norm`, per-pair LayerNorm before the edge tanh; probe protocol, 3 seeds by seed against a fresh torch-2.11 control; same-seed runs on this stack turned out not to pair, 2026-10-10) | **ΔAUC −0.00004 ± 0.00010** (t = −0.6), +0.00000 unpaired against the pooled six-run control; 1/ε_B@0.3 **−41.8 ± 7.8** by seed, lower on every seed, and −33 read unpaired against the pooled six-run control (t = −2.6); same GPU time. It halves the tanh saturation of rounds 1–2 (34/61% → 15/31% at seed 0) without moving AUC, so saturation is not the limit on this protocol | (SO3C) `results_message/probe_screen/`, `so3c_notes/gap_generic_ml.md` §9.2 |
| **Log knee** (`--log-knee`, asinh(e^κ x) with learnable κ on the bilinear invariants; same screen) | **ΔAUC +0.00000 ± 0.00008** (t = 0.08), +0.00004 unpaired against the pooled six-run control (t = 0.7); 1/ε_B@0.3 +39.0 ± 30.1 by seed, positive on all three (p ≈ 0.08 one-sided), +48 read unpaired against the pooled control (t = 2.1), suggestive only; κ barely moves from its initial 4 (3.2–4.4). Free (+7 parameters), so a possible passenger, not an arm | (SO3C) `results_message/probe_screen/`, `so3c_notes/gap_generic_ml.md` §9.3 |
| **Beam energy 0.02** (`--beam-energy 0.02`, R1 of the representation screen; probe protocol, 3 seeds against a fresh same-stack control) | **ΔAUC −0.00025 ± 0.00018** by seed, every seed negative; −0.00028 against the pooled six-run control (t = −2.8); all three runs below all six control runs; 1/ε_B@0.3 765.9 against 808.9; best validation loss worse on every seed. Unresolved by the plan's rule; closed by judgment | (SO3C) `results_message/repr_screen/` |
| **Lab-frame particle features** (`--lab-scalars`, R2: ParT's log E, log pT, their ratios to the jet's, y − y_J, \|Δφ_J\|, ΔR_J in the scalar initialisation, written in invariants of {p, P, b±}; probe protocol, 3 seeds against a fresh same-stack control) | **ΔAUC +0.00004 ± 0.00009** by seed, mixed signs; +0.00000 against the pooled six-run control (t = 0.05); 1/ε_B@0.3 796.2 against 808.9; free (+56 parameters, 0.98× the time). Closed by the plan's rule | (SO3C) `results_message/repr_screen/labscalars` |
| Best-validation checkpoint | Validation loss is lowest at the **final** epoch in both headline seeds and both beamless K=64 seeds (cosine schedule), so the best-validation checkpoint is the last one. Not after the 60 → 120 warm restart: lowest at epoch 110 (0.16170 against 0.16184 at 120), with the same AUC (0.98392) and rejection 1194.8 against 1201.9, so the best-validation weights still add nothing (2026-10-10, `results_k64/headline120`) | (SO3C) 2026-09-19, from the stored histories |

**Lesson worth keeping:** a closed question can re-open when the inputs change. Capacity was flat without beams and looked as if it paid once beams supplied the information (one seed, not resolved); width did pay later, paired, at the long schedule. kNN was useless at K=32 and 3.15× faster at K=128.

## Open — candidates

**Lead (2026-10-11): lab-frame pair features (`--pair-lab`) at K=64.** Promoted on the probe protocol (+0.00051 against the pooled control, t = 5.9) and confirmed at the K=32 long schedule (+0.00064, t = 6.8, free). Not yet run at K=64, the protocol the current best is measured on. Its four features (ln Δ, ln k_T, ln z, ln m²) have not been separated.

Ranked literature and generic-ML candidates, every literature number checked against its paper's table: [`so3c_notes/gap_thesis.md`](so3c_notes/gap_thesis.md) and [`so3c_notes/gap_generic_ml.md`](so3c_notes/gap_generic_ml.md), with the 2026-10-06 audit's corrections applied. Their two measured leads were conditioning problems — round-0 inputs in asinh's linear range (about 95% of pair invariants below 0.1) and edge-tanh saturation in rounds 1–2 (61–84% of pre-activations beyond |x| > 2), with no normalisation layer in the network. Both were screened on 2026-10-05 and **neither moves AUC** (Closed table). The plan for the window opening 2026-10-10 is [`so3c_notes/plan_2026-10-10.md`](so3c_notes/plan_2026-10-10.md).

### Screened on 2026-09-20, none better than the base

The two defect fixes, the vector channel, the relative-norm edge, the pair latent, their combination, depth 6 and f_α were all measured on the probe protocol (see the screen table below). None improved on the base in that screen; the seed spread measured since (`3ca06bd`) shows the screen was noise (see the GPU-window sections below).

### Verified defects in the current model (free to fix)

Both are now switchable and default to the old behaviour: `--no-mass-input --no-self-edges` (22634 params on beams + channels 8, CPU cost 1.00x the headline, `6771158`). Screened on GPU on 2026-09-20: one seed, no resolved gain.

- **m² fed as noise.** `h_init` takes `asinh(m²)` per node, and three of the seven `_minkowski_stats` readout features are per-particle m² moments. All of it is rounding noise (see Closed).
- **Self-edges in the dense graph.** The dense path includes `a = b` in the scalar message sum and counts N, not N−1, in the denominator. The flow itself is unaffected, since `z_a × w_aa z_a = 0`. LorentzNet masks the self-edge; our sparse path already excludes it.

### Best-motivated architectural candidates

- **Rank-2 pair latent** with a reduced Eq2→2 aggregator basis (7 of PELICAN's 15). PELICAN carries a [B,N,N,C] state through its blocks; we collapse pairs to nodes every round. The verified table-2 comparison — 1k PELICAN parameters match our 22.8k — is consistent with this being the dominant carrier. Outcome likely bimodal: a real step toward 1400–1800 rejection, or under +0.0005 if our flow already supplies it. **Implemented** as `--pair-latent N`: 3680 parameters at C_p = 8 on beams + channels 8, CPU cost 1.56x the headline, 1.64x combined with the vector channel and both fixes; on a T4 1.55x per step, an estimated ~7.7 h per 30-epoch seed at K=64 (~5.8 h compiled). Exactly Lorentz covariant (1.7e-16) and permutation invariant (0.0), with the 7 aggregator maps checked for permutation equivariance on their own. Screened on GPU on 2026-09-20 and replicated over three seeds: +0.00012 AUC by seed (paired t = 1.42; unpaired t = 1.2), not resolved. Agent estimate: screen at 20% data × 3 seeds ≈ 11 GPU-h.
- **A vector channel alongside the bivector.** Two reasons, one measured elsewhere and one structural:
  - arXiv:2606.21790 (abstract, verified) finds that in L-GATr "bivector channels are negligible for top-quark tagging while vector-like channels are dominant". The LLoCa (2505.20280) and slim L-GATr (2512.17011) abstracts do not address grades; their bodies are unchecked.
  - Our lift `z_a = bivec(p_a, P)` is **unchanged under p_a → p_a + λP**, so it discards each constituent's component along the jet axis (**correction, 2026-10-05:** for the massless constituents of this dataset p_a² = 0 fixes that component, and z_a·z_a = ⟨p_a,P⟩², M² = Σ⟨p_a,P⟩ and Re z_a·z_b recover every ⟨p_a,p_b⟩, so the lift loses no information; `so3c_notes/gap_thesis.md` §3). The scalar channel recovers ⟨p_a, P⟩ only as an invariant. A covariant vector v_a = p_a keeps it — the same kind of missing-information problem that beams fixed. **Implemented** as `--vector-channel`: 531 parameters on beams + channels 8, CPU cost 1.10x the headline, exactly covariant to 7e-15 with and without beams, and the vector update moves the logits by 7e-2 to 9e-2. Screened on GPU on 2026-09-20 and replicated over three seeds: −0.00002 AUC by seed, bounded near zero if those runs pair (paired differences sd 0.00006; read unpaired, the 95% bound is about ±0.00035).

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

Plan: [`so3c_notes/plan_2026-10-10.md`](so3c_notes/plan_2026-10-10.md). Its code steps C0–C6 were not done before the window opened; they were done on 10-10, in this order:

| step | commit | what |
|---|---|---|
| C0 | `43c2242`, `3599b68`, `83dc8a8`, `f4aba0f`, `9938676` | the gap-notes branch integrated: the 10-05 screen results, the notes with the audit corrections, its status and docs content ported into these files |
| continuation | `acfb88e` | slot A: `headline60` continued 60 → 120 epochs (plan §2a), pushed 06:23 as `nsanya/so3c-k64-headline120-seed0`; still running after 20 min, so past the guard, the checkpoint check and the smoke run, any of which would have stopped it (the log is readable only at the end). A finish kernel is prepared, to push only if the run stops early |
| C1 | `7c274e7` | `--beam-energy` (R1) |
| C2 | `86ca67a` | batch-order fingerprint in every result JSON (`batch_order`: epoch 1's permutation and a running hash, carried through checkpoints) |
| C3 | `c9dd235` | `build_notebooks` builds nothing unless named: `--list`, `--only NAME...`, `--all` |
| C4, C5 | `06c68bc` | `--lab-scalars` (R2) and `--pair-lab` (R3), ParT's lab-frame particle and pair features written in invariants of {p, P, b±}; CPU cost 0.98× and 1.50× |
| C6 | `227bf53` | slot B notebooks: `probe_screen_{beame002,labscalars,pairlab}`, each the 10-05 control's command plus one flag (tested against the committed control notebook), seeds 0–2, paired summary with the plan §3 decision rule. R1 starts with the pairing check: control seed 0 rerun for two epochs of its 20-epoch schedule (two one-epoch sessions), compared with the stored history; pass within 1e-5, otherwise a fresh control |

Slot B, one kernel at a time:
- R1 (`nsanya/so3c-probe-screen-beame002`), pushed 06:54 after a direct review of the diff and the notebooks;
- the fresh control `base1010`, pushed 08:43, because R1's pairing check failed;
- R2 (`labscalars`), pushed 11:20;
- R3 (`pairlab`), pushed 12:57.

Phase 2 (the branch for a promoted arm, plan §4): the same-stack long controls `base211` s0 and s1 were pushed at 17:15 into both slots, and the R3 long-schedule arm `pairlab211` s0 and s1 at 22:47 (the session was paused while the controls finished).

### Phase 2: long schedule (K=32, all 1.211M training jets, 60 epochs)

**Same-stack controls `base211`** (seeds 0 and 1; T4, torch 2.11, `--keep-best`; `results_long_seeds/base211`):

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| `base211`, AUC / 1/ε_B@0.3 | 0.98225 / 909.5 | 0.98254 / 975.4 | 0.98240 |
| base, torch 2.10 (`results_long_seeds/base`) | 0.98261 / 943.5 | 0.98221 / 881.7 | 0.98242 ± 0.00020 (3 seeds) |
| `w128s32` + weight decay, torch 2.11 (`w128s32wd2`) | 0.98298 | 0.98259 | |

- **The base level agrees across builds:** 0.98240 on 2.11 (two seeds), 0.98242 ± 0.00020 on 2.10 (three seeds). Pooled, the five runs give 0.98241 ± 0.00017 AUC and 922.3 ± 37.1 rejection.
- **The seed order does not carry over.** On 2.10, seed 0 beat seed 1 by 0.0004 in both the base and `w128s32`; that was the evidence that the long schedule pairs. On 2.11 the base reverses it (−0.00036 and +0.00033 against the same 2.10 seeds). The 2.11 weight-decay runs kept the 2.10 order, so on 2.11 two architectures did not track by seed: `w128s32wd2` − `base211` is +0.00073 and +0.00004.
- **So the R3 long runs are read unpaired:** two arm runs against the five pooled base runs, standard error ≈ 0.00014. The plan's by-seed reading against `base211` is reported alongside.
- Best-validation weights: epochs 58 and 56. AUC 0.98225 and 0.98253 (−0.00000 and −0.00002 against the final weights); rejection 909.5 and 1004.5 (+0 and +29).
- Batch orders (epoch 1): `4956a97c…` and `99fdd635…`. The R3 long runs should match them.
- GPU time: training 3.25 h and 3.35 h; kernels 3.33 h and 3.42 h (log span).

**R3 at the long schedule (`pairlab211`, seeds 0 and 1, pushed 22:47): the gain holds** (+0.00064, against +0.00051 on the probe protocol; the difference is about one standard error).

| K=32 long schedule | seed 0 | seed 1 | mean ± sd |
|---|---|---|---|
| R3 (`--pair-lab`), AUC / 1/ε_B@0.3 | 0.98310 / 1057.1 | 0.98300 / 934.8 | 0.98305 ± 0.00007 / 996.0 |
| `base211`, AUC / 1/ε_B@0.3 | 0.98225 / 909.5 | 0.98254 / 975.4 | 0.98240 / 942.5 |
| Δ by seed (the plan's reading) | +0.00085 | +0.00045 | +0.00065 |

- Unpaired, the primary reading: +0.00064 against the five pooled base runs (0.98241 ± 0.00017; Welch t = 6.8). Both R3 runs lie above all five base runs (worst R3 0.98300, best base 0.98261).
- Best validation loss: 0.16442 and 0.16465, against 0.16745 ± 0.00090 for the pooled base (−0.0029, t = −7.0).
- Rejection: 996.0 against 922.3 ± 37.1 (+74, t = 1.2, not resolved).
- Lowest validation loss at epoch 58 in both runs, and the best-validation weights give the same AUC. The validation − train gap at epoch 60 is 0.0046–0.0049, against 0.0045–0.0050 for the control: the features do not add overfitting.
- Cost: free at this schedule (training 3.30 h and 3.35 h, 1.01× and 1.00× the control's). Batch orders equal the controls'.
- For scale: `w128s32` (120,202 parameters, about twice the time, torch 2.10) averaged 0.98289. R3 gets more, +0.00064 against +0.00048 over the same base level, with +192 parameters. That comparison crosses torch builds.

R3 pays on both schedules it has been run on, and unlike width (whose validation − train gap doubled at this schedule) without adding overfitting. Width's edge axis was never screened on the probe protocol; at K=64 over 30 epochs it gave +0.00008, a schedule R3 has not been run on. It is the largest gain from a change to the model since the beams, and about as large as the whole training-length lever: width gave +0.00048 at this schedule (torch 2.10), while at K=64 training longer gave +0.00037 at 60 epochs and +0.00066 at 120 epochs over the 30-epoch mean (one run each against a three-seed mean; the 120-epoch run is +0.00029 over its own epoch 60). Unlike either, it costs neither time nor overfitting.

Budget: 27.3 GPU-h used (13.76 h phase 1, 6.75 h controls, 6.77 h R3 long); about 2.7 h left until the quota resets on Saturday 2026-10-17.

### 60 → 120 continuation (slot A, K=64, canonical protocol)

**Unresolved by the plan's rule, and the new best single model** (8.06 h). Against its own epoch 60: AUC 0.98392 against 0.98363 (+0.00029, just under the +0.0003 the rule asks for to call it paying), rejection 1201.9 against 1140.8 (+61), final validation loss 0.16184 against 0.16276 (−0.00092). The restart at epoch 61 (learning rate back to 1.5e-3) lifted the validation loss to 0.16802 at epoch 61 and 0.16865 at epoch 62; it fell below the epoch-60 value by epoch 90 and reached 0.16170 at epoch 110 (the best-validation weights: AUC 0.98392, rejection 1194.8). It is one run compared with its own earlier snapshot, so the gain is not a seed-to-seed comparison. Against the four-model ensemble (0.98396) it adds +0.00013 in place of the 60-epoch member (0.98409) and +0.00016 beside it (0.98413); added to the three 30-epoch seeds alone, +0.00031 (Current best).

### Representation screen (probe protocol, slot B)

**Same-seed runs in different kernels do not reproduce (measured 2026-10-10).** Control seed 0 ran three times with the same default-path code on the same stack (T4, torch 2.11.0+cu128, compiled), each in its own Kaggle kernel:

| run of control seed 0 | batch order, epoch 1 | epoch-1 val loss | final AUC |
|---|---|---|---|
| 10-05 control | not recorded (before C2) | 0.20259 | 0.98115 |
| R1's pairing check (two epochs) | `a36497dc…` | 0.19830 | — |
| fresh control `base1010` | `a36497dc…`, the same | 0.20283 | 0.98097 |

- The check and the fresh control have equal batch-order fingerprints and, by the code (`torch.manual_seed` before the model is built), the same initial weights. Their epoch-1 losses still differ by up to 4.5e-3.
- **The likely cause is per-kernel compilation.** Each run prints a compiled-vs-eager check on its first batch, before any optimiser step. For seed 0 it gives 8.1e-5 (10-05), 5.3e-5 (the check) and 4.8e-5 (`base1010`), so the compiled forward pass already differs between kernels at step 0. The check's two sessions ran in one kernel, sharing its compile cache, and agree to the printed precision (5.3e-5 and 2.9e-7 in both). Inductor choosing among kernel configurations by timing would do this. Atomic adds would not explain it: the step-0 numbers quoted are the logit difference, computed from forward passes before the check compares gradients, and the only indexed operation on the default path is the readout's `T_cc[:, pair_rows, pair_cols]`, whose upper-triangle indices are unique, so its accumulating backward has no colliding writes. No run was repeated inside one kernel, so reproducibility within a kernel is untested.
- **A shared seed still shares part of the trajectory.** Seed 1's epoch-1 validation loss is the outlier in all five runs (0.23–0.27, against 0.19–0.20 for seeds 0 and 2). Over epochs 1–20, same-seed control runs differ in validation loss by 0.0009–0.0015 on average, against 0.0017–0.0052 for different seeds. Their per-jet test scores correlate at 0.9955–0.9959, above every different-seed control pair (0.9944–0.9953).
- **It does not pair the final AUC at this protocol.** The same seed run twice (`base1010` − 10-05, seeds 0–2) gives final AUCs −0.00018, −0.00003 and −0.00002 apart, and rejections up to 62 apart. Different seeds within one control differ by up to 0.00004 (10-05) and 0.00011 (`base1010`) in AUC, and by up to 20 and 72 in rejection, so a same-seed rerun lands as far off as a different seed. R1's per-jet scores are no closer to the same-seed control run (0.9929–0.9937) than to other seeds' runs (0.9927–0.9935).

So at the probe protocol on this stack **arms are read as unpaired**, against the controls pooled. The six control runs pool to 0.98108 ± 0.00006 AUC and 808.9 ± 25.9 rejection. The 10-05 screen's "paired sd 0.00008–0.00010" is the size of this run-to-run noise: same-seed control reruns differ with sd 0.00009, and √2 × 0.00006 ≈ 0.00008. Its deltas are consistent with unpaired draws, and its AUC verdicts stand when read unpaired: against the pooled six-run control the edge norm is ΔAUC +0.00000 (t = 0.0) and the log knee +0.00004 (t = 0.7). Other settings behaved differently, and not consistently by torch version. At the K=32 long schedule the two `w128s32` − base differences (torch 2.10) agreed to ±0.00001 while the seeds swung 0.0004, and the torch-2.11 weight-decay runs tracked their 2.10 counterparts as closely (−0.00010 and −0.00012). The torch-2.11 base at the long schedule, though, reversed the 2.10 seed order (phase 2 above). At the probe protocol on torch 2.10 the vector channel tracked the base seed by seed (paired differences sd 0.00006, against a base seed spread of 0.00017), but the pair latent did not (paired sd 0.00015). On torch 2.11 the probe seeds barely differ (sd 0.00002–0.00006) and a same-seed rerun moves as much, so pairing has nothing to remove. Whether a shared seed pairs the final metric depends on the stack and the protocol, and has to be measured, not assumed.

**R1 (`--beam-energy 0.02`): unresolved by the plan's rule and not a candidate; treated as closed, a judgment beyond the rule.**

| | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| R1, AUC | 0.98081 | 0.98097 | 0.98063 | 0.98080 ± 0.00017 |
| fresh control, AUC | 0.98097 | 0.98108 | 0.98109 | 0.98105 ± 0.00006 |
| Δ by seed (the plan's rule) | −0.00017 | −0.00012 | −0.00046 | −0.00025 ± 0.00018 |
| R1, 1/ε_B@0.3 | 761.9 | 779.6 | 756.2 | 765.9 (pooled control 808.9 ± 25.9) |
| best validation loss, R1 / fresh control | 0.17408 / 0.17363 | 0.17360 / 0.17345 | 0.17477 / 0.17299 | worse on every seed |

- By the letter of plan §3 this is "unresolved": |mean| ≥ 0.0001 and the signs agree, and the rule has no row for a consistent loss. The kernel printed the same verdict.
- Against the pooled six-run control it is −0.00028 (Welch t = −2.8, df ≈ 2.2).
- All three R1 runs lie below all six control runs (best R1 0.980966, worst control 0.980974). If the beam energy made no difference, that ordering would have probability 1/84.
- Each seed's batch order equals the fresh control's.

The cost cannot be pinned on one mechanism. At E_b = 0.02, asinh⟨p,b±⟩ is entirely linear (scaled from the plan's measurement at E_b = 1), and the beam nodes' soft bound 1+⟨b,P⟩, saturated at E_b = 1, no longer is; one arm cannot say which change costs. Kernel time (log span): R1 1.50 h including the check; fresh control 1.37 h.

**R2 (`--lab-scalars`, ParT's seven particle features in the scalar initialisation): closed at the probe level.**

| | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| R2, AUC | 0.98105 | 0.98102 | 0.98119 | 0.98109 ± 0.00009 |
| fresh control, AUC | 0.98097 | 0.98108 | 0.98109 | 0.98105 ± 0.00006 |
| Δ by seed (the plan's rule) | +0.00007 | −0.00006 | +0.00011 | +0.00004 ± 0.00009 |
| R2, 1/ε_B@0.3 | 779.6 | 810.9 | 798.1 | 796.2 (pooled control 808.9 ± 25.9) |
| best validation loss, R2 / fresh control | 0.17335 / 0.17363 | 0.17288 / 0.17345 | 0.17236 / 0.17299 | |

- The plan's rule closes it: the signs are mixed and |mean| < 0.0001.
- Against the pooled six-run control it is +0.00000 (Welch t = 0.05).
- The best validation loss is lower than the fresh control's on every seed, but against the pooled six-run control the difference is unresolved (−0.00029, Welch t = −0.9), and R2's seed 0 (0.17335) lies above all three 10-05 control runs (0.17287–0.17307).
- Each seed's batch order equals the fresh control's.
- Cost: 22,890 parameters (+56); training time 0.97–0.98× the fresh control's per seed (mean 0.98×); kernel 1.34 h (log span).

This agrees with L-GATr's Table 3: lab-frame particle features add nothing once the network has reference vectors, which the beams already are here.

**R3 (`--pair-lab`, ParT's four pair features ln Δ, ln k_T, ln z, ln m² on the edges): promoted to the long schedule.**

| | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| R3, AUC | 0.98144 | 0.98164 | 0.98172 | 0.98160 ± 0.00015 |
| fresh control, AUC | 0.98097 | 0.98108 | 0.98109 | 0.98105 ± 0.00006 |
| Δ by seed (the plan's rule) | +0.00046 | +0.00055 | +0.00063 | **+0.00055 ± 0.00009** |
| R3, 1/ε_B@0.3 | 817.5 | 837.8 | 834.4 | 829.9 (pooled control 808.9 ± 25.9) |
| best validation loss, R3 / fresh control | 0.17136 / 0.17363 | 0.17100 / 0.17345 | 0.17073 / 0.17299 | |

- The plan's rule promotes it: mean +0.00055 ≥ +0.00015 and every seed positive.
- Read unpaired, which is the primary reading on this stack, it is +0.00051 against the pooled six-run control (Welch t = 5.9), and all three R3 runs lie above all six control runs.
- Best validation loss −0.0021 against the pooled control (t = −9.7); rejection +21 (t = 1.7, not resolved).
- Each seed's batch order equals the fresh control's.
- Cost: 23,026 parameters (+192); training time 1.04–1.13× the fresh control's per seed on the T4 (`walltime_sec`), against 1.50× per step on the CPU; kernel 1.49 h (log span).

This is the largest gain on the probe protocol since the beams (+0.0023). Our edges already see the pair invariants s_ab, compressed by asinh. R3 adds four log-scaled features: three lab-frame ones (ln Δ, ln k_T, ln z), which need the beams to define y, φ and p_T, and ln m²_ab, a direct ⟨p_a,p_b⟩ that the edges otherwise see only inside Re z_a·z_b = ⟨p_a,P⟩⟨p_b,P⟩ − ⟨p_a,p_b⟩M². One arm cannot say which carries the gain. The log knee (log-compressed s_ab) and the vector channel (asinh⟨p_a,p_b⟩ on the edges) were both null, which points to the lab-frame three, but that is not measured. Whether the gain survives the long schedule is phase 2.

**Reading R2 and R3.** Their notebooks print deltas against the stored 10-05 control by seed. The plan's letter reads them against `base1010` by seed, and the fingerprint confirms the batch order. On this stack that is an unpaired comparison too, so the pooled six-run control is read alongside it.

Also prepared, for phase 2 if no arm is promoted: `long_seeds_pairlat211_s{0,1}`, the same command plus `--pair-latent 8` (the pair latent cost 1.34× the base on the probe protocol, so ~4.7 h per seed).

Phase 2 needs a K=32 long-schedule control on its own stack in either branch (plan §4; the `results_long_seeds` bases ran on torch 2.10). Prepared: `long_seeds_base211_s{0,1}`, the committed long base's command plus `--keep-best`, T4 / torch 2.11 pinned, each result's stack asserted and its batch order printed (the 2.10 base took 3.37–3.52 h per seed).

## GPU window 2026-10-03 → 2026-10-10 (closed): regularising the wide model

**Run:** w128s32 with decoupled weight decay (`--optimizer adamw --weight-decay 0.002`) and `--keep-best`, seeds 0 and 1, long-schedule protocol (`c762616`). Verified before pushing: the notebooks equal the control notebooks plus exactly those flags; same initial weights and same batch order as the control under each seed, checked on one torch build (the controls later turned out to have run on 2.10 and the decay runs on 2.11; see below); AdamW with zero decay reproduces Adam exactly, so the decay is the only knob. Pushed 03:10, 6.30 h and 7.41 h. Results in `results_long_seeds/w128s32wd2`.

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
| Δ (one run each, unpaired) | 2.7× | **+0.00008** | −5 | −3 | −0.00035 | | |

Neither model overfits at 30 epochs (validation loss lowest at the final epoch, so `--keep-best` returns the final weights). At K=32 the width gain appeared on the **60-epoch** schedule; at K=64 on 30 epochs it is +0.00008. That is consistent with capacity needing the long schedule, so width at K=64 on 60 epochs is still untested — not refuted.

**Cross-software reproducibility.** The control reruns the original seed-0 headline (P100, torch 2.5.1, old forward pass, eager: 0.98326 / 1147.2) and lands at 0.98311 / 1040.8: −0.00014 AUC, −106 rejection, about one seed spread. Across that software jump two seed-0 runs behave like independent seeds.

**Kaggle moved from torch 2.10 to 2.11** between 2026-09-27 and 2026-10-03, unannounced. Every log was checked: the long screen (base, w64s24, w128s32), the probe replicates and headline seed 2 ran on 2.10; the weight-decay runs and today's K=64 pair on 2.11. So the weight-decay comparison crossed the upgrade. Whether a shared seed still gives the same batch order across 2.10 → 2.11 cannot be checked directly here (no CUDA locally), and the epoch-1 losses do not settle it: same-seed pairs across the upgrade differ by 0.0008–0.0015 in epoch-1 validation loss, but base seeds 0 and 2, different seeds on one version, differ by only 0.00008. So the weight-decay verdict is paired only if the batch order survived the upgrade; unpaired, −0.00011 is inside the ±0.00027 seed spread. **Trap:** record the torch version of every run (the GPU_SETUP line in each log) and check it before calling two runs paired.

### Conditioning screen (2026-10-05)

The two conditioning switches of the gap notes, on the probe protocol (K=32, 400k training jets, 20 epochs, `--compile`), three seeds per arm, read seed by seed against a **fresh** control on the same build (T4, torch 2.11.0+cu128, pinned and asserted by the new setup cell; every result JSON now records `env`). Code `2303568`; results `results_message/probe_screen/` (9 runs, 4.16 GPU-h). On 2026-10-10 same-seed runs on this stack turned out not to pair; read unpaired, the AUC verdicts below stand (representation screen section).

| arm | params | ΔAUC vs control | Δ 1/ε_B@0.3 |
|---|---|---|---|
| control | 22834 | 0.98112 ± 0.00002 (absolute) | 817.6 ± 11.4 (absolute) |
| `--edge-norm` | 22930 | −0.00004 ± 0.00010 | −41.8 ± 7.8 |
| `--log-knee` | 22841 | +0.00000 ± 0.00008 | +39.0 ± 30.1 |

Conditioning, the top-ranked cheap lead, is not where the gap is. A fresh control was the safe design; whether the torch-2.10 baseline would have paired is **not shown** — epoch-1 losses cannot tell same-seed from different-seed runs (audit #6).

**Budget:** ~13.9 h (regularised wide model) + ~11.2 h (the two K=64 runs) + 4.16 h (conditioning screen) ≈ **29.3 of 30 GPU-h** for this window (resets Saturday 2026-10-10).

**Where this leaves the SOTA question.** Best single model is still the 60-epoch K=64 headline, 0.98363; best ensemble 0.98396; PELICAN 0.9870. (Superseded 2026-10-10: the 60-epoch run continued to 120 epochs scores 0.98392, and the five-model ensemble 0.98413; a full width transfer would then leave about 0.0024 and 0.0026.) Measured levers so far: training length (+0.00037 at K=64, one seed: suggestive), width (+0.00048 at K=32 on 60 epochs, +0.00008 at K=64 on 30 epochs), weight decay (−0.00011, not shown to help: inside the ±0.00027 spread unpaired), early stopping (~0). The one width test left is K=64 on 60 epochs: w64s24 needs ~13.4 h (two sessions, so a finish kernel), w128s32 ~22 h, and it would need a same-stack 60-epoch control (~8.6 h): the 60-epoch headline ran on torch 2.10, and pairing across 2.10 → 2.11 is not established (audit #1). Even a full transfer of the K=32 width effect would leave about 0.0026 from the best ensemble (0.0029 from the best single model), so this is the last tuning-scale lever; closing the gap needs a structural change, not more of these.

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

**w128s32 − base = +0.00048 AUC, +81 rejection (+9%).** By seed (same batch order; here the pairs tracked, see below): +0.00047 and +0.00049. Unpaired against three base seeds: t = 2.34, df 3, p ≈ 0.05 one-sided. The midpoint (w64s24, one seed) is +0.00025 over base seed 0, so AUC rises monotonically with width at roughly +0.0002 per doubling of parameters, while overfitting rises just as steadily.

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

Unpaired, +0.00048 is only t = 1.76 (df 2). But **runs with the same seed see the same batch order across architectures**: `train_classifier` calls `torch.manual_seed(cfg.seed)` after the model is built, so model initialisation, whatever RNG it consumes, cannot shift the shuffle; checked directly by capturing the permutations of two different architectures under seed 1 (identical). The paired differences agree to ±0.00001 (paired t ≈ 48, df 1, p ≈ 0.007 one-sided) while the seeds swing ~0.0004, so here that swing behaved as noise **shared** between the arms. A shared batch order alone does not guarantee this: at the probe protocol on torch 2.11 it did not, nor at this long schedule on torch 2.11 (`w128s32wd2` − `base211` +0.00073 and +0.00004; phase 2, 2026-10-10). Two pairs make that variance estimate fragile, so the honest reading is *strong, pending more pairs* — but the unpaired test badly understates it. The commit message of the w128s32 seed-1 result said a shared seed does not pair runs; at this protocol on torch 2.10 that was wrong.

**Methodological consequence:** design every comparison paired by seed, which costs nothing, but analyse it paired only where runs that share a seed are shown to track each other better than different seeds do. Here, at the long schedule on torch 2.10, the two `w128s32` − base differences agreed to ±0.00001 while the seeds swung 0.0004; there was no same-seed rerun. At the probe protocol on torch 2.11, same-seed reruns did not track (representation screen, 2026-10-10), and at the long schedule on torch 2.11 the base and the weight-decay runs did not track each other either (phase 2, 2026-10-10). At the long schedule on torch 2.10 the seed-to-seed swing (~0.0004) was shared and an order of magnitude larger than the architecture effect's own scatter, so there pairing was worth far more than extra seeds; on torch 2.11 it was not shared (`w128s32wd2` − `base211` +0.00073 and +0.00004), so that gain has to be re-measured, not assumed.

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
- **Vector channel:** paired differences with sd 0.00006, so if the runs pair it is bounded to |Δ| < ~0.00015 at 95% (they tracked the base by seed; read unpaired the bound is about ±0.00035) — a real negative at that level, and much sharper than the screen's single seed. Feeding ⟨p_a, p_b⟩ directly, which the bivector lift exposes only through combinations with ⟨p_a, P⟩ and ⟨P, P⟩, does not pay. The regime excuse is weak here: it adds information, not parameters.

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
| Kaggle's T4 image moved from torch 2.10 to 2.11 between windows (seen 2026-10-04) | seed pairing assumes the same CUDA `randperm` stream, unverified across versions; the probe and long-schedule baselines are on 2.10 | screen notebooks pin and assert the build (`GPU_SETUP_T4_PINNED`, `2303568`); every result JSON records `env`; rerun controls on the same stack rather than reuse old ones (a same-stack control fixes the batch order, not the trajectory: see the trap on same-seed training below) |
| `kaggle_client push` asks for confirmation on stdin | in a non-interactive shell it died with EOFError after printing "about to push" | pass `--yes` once the push is approved |
| The embedded bundle's sha256 changes on every build | gzip stamps the compression time, so the hash names a blob, not a code state | to tie a notebook to a commit, extract its bundle and compare file contents with the tree |
| `.double()` on a complex tensor | silently drops the imaginary part (a warning only); a read-only diagnostic briefly reported nonsense jet-frame norms | cast complex tensors with `.to(torch.complex128)` |
| A Windows Python given a Git Bash `/tmp/...` path | resolves it to `C:\tmp\...`, not the scratch directory | hand Python Windows paths (`cygpath -w`) |
| A local smoke run with `--canonical-splits` | evaluates all 807k validation and test jets on CPU; it did not finish in 15 min | replay on the internal split with `--max-samples`; the canonical loader is proven on Kaggle |
| Same-seed training in different Kaggle kernels (T4 / torch 2.11, compiled) is not bitwise reproducible | with the same default-path code: epoch-1 losses 4.5e-3 apart at equal batch-order fingerprints (R1's check against `base1010`); final AUC up to 0.00018 and rejection up to 62 apart between the 10-05 and 10-10 controls (the 10-05 batch order is assumed: it predates the fingerprint), as large as the differences between seeds. The compiled-vs-eager check on the first batch already differs between kernels | read same-seed comparisons as unpaired unless same-seed reruns are shown to agree better than different seeds; pool the controls; the fingerprint verifies the batch order, not the trajectory |
| `--resume` skips a run whose result JSON already exists, even one written by a run that stopped early | a second session pointed at the same results directory does nothing | a finish or second session writes to its own results directory and finds the run through the checkpoint (the pairing check uses `session1/`, `session2/`) |
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
- Quota is 30 GPU-h per week, resetting on Saturday. The current window is [2026-10-10 → 2026-10-17](#gpu-window-2026-10-10--2026-10-17-in-progress).
- Rows not marked T4 are P100 wall clock on the old forward pass. Since `47ec4ad` the per-epoch cost is about half; the T4 profiles are in the 2026-09-19 GPU-window section and in `docs/so3c/experiments.md` (table O).

## Provenance

- This repository is a **snapshot** of the so33 repository's branch `feature/so3c-complexification`, taken at its tip `d360acc` (2026-09-13) and pushed as a single root commit, `6f79e70`, tagged `pre-so33-removal`. It shares no history with so33.
- **Hashes of commits made before the snapshot refer to the so33 repository**, not to a commit reachable from here: every hash in the Closed table, and those in the commit messages inherited from before the snapshot. Hashes of later work are commits in this repository; `git cat-file -t <hash>` tells which.
- Before the bench is deleted from so33, **tag `d360acc` there** (for example `so3c-snapshot-2026-09-13`), so those hashes stay reachable after that repository is pruned.
- SO(3,3)-only content was removed here on 2026-09-18 in four commits: `e322d9e`, `f3f1568`, `09aa178` and the packaging/documentation commit. [`REMOVED.md`](REMOVED.md) lists what went, what was kept on purpose and what changed as a result.
- so33 keeps its copy of this bench for now; it remains the home of the SO(3,3) paper, its Zenodo DOI and its `CITATION.cff`.
