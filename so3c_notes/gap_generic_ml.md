# Generic ML and optimisation levers: diagnostics on the trained headline

A follow-up to [`gap_thesis.md`](gap_thesis.md), written 2026-10-05 in the same chat with no agent fan-out. It covers optimisation and training tricks that have nothing to do with physics. Rules for this pass:

- Every diagnostic ran read-only on CPU, from scratch scripts outside the repository.
- Nothing was pushed to Kaggle, and nothing in `benchmarks/` or `so3c/` changed.
- Numbers marked **measured** come from those runs. The core of each script is in appendix A.
- Gain estimates not backed by a measurement are marked **prior**.

**Checkpoints**

| file | epochs | role | test AUC |
|---|---|---|---|
| `results_k64/headline60/checkpoints/...seed0.pt` | 60 | best single model | 0.98363 |
| `results_k64/headline/checkpoints/...seed2.pt` | 30 | — | — |

Both are float32, K=64, beams, 8 channels. Both store Adam's moment estimates, which several diagnostics below read directly.

**Input pipeline (verified first)**

- The scripts rebuild the training inputs from `data/toptagging/*.parquet` using the loader's rule: the 64 leading constituents by pT, divided by one global RMS.
- That scale, computed over all 1,211,000 training jets, is **37.90 GeV**. `gap_thesis.md` §2 quoted 38.0 GeV from 30k jets; the difference changes none of its conclusions.
- With 37.90, both checkpoints reproduce their stored test scores on the first 3,000 test jets: median |Δscore| 2.4e-7, max 2.0e-4. The residual is float32 CPU against compiled CUDA.

---

## 1. Bottom line

1. **The flow-head penalty is not what keeps the flow quiet.**
   - In channels whose state is alive, the data gradient is 47–405× the penalty gradient (§2.2).
   - The penalty is 0.07–2.5% of Adam's per-element normaliser (§2.3).
   - The loss curvature along each head implies the penalty shrinks the round-0 heads by about 2–9% and the later heads by ≲ 0.2% (§2.3).
   - It pins only the heads of channels whose state has already collapsed to about zero, where a head does nothing.
   - Removing it for 300 paired CPU steps changes nothing visible, and nothing breaks even at peak learning rate (§2.4).

   Round 0 is quiet because the rotation generator is quadratic in the state and round-0 states are small (§2.5). **No coefficient-0 arm is proposed.**
2. **The default that does bind is tanh saturation in the edge layer.**
   - In rounds 1–2, 61–84% of edge pre-activations have |x| > 2, where tanh′ ≤ 0.07. In round 0 the share is 14–40%.
   - At initialisation it is zero, so saturation grows with training (§3.2).
   - It is the measured mechanism behind `gap_thesis.md` idea 7.1(b): normalise the edge pre-activation.
3. **Gradient clipping is not a lever.**
   - It binds on 29% of the first 150 steps from a fresh initialisation and on 0–12% of batches at the trained weights (§3.1).
   - Under Adam it only down-weights outlier batches.
4. **Test-time augmentation over the detector's reflections is negligible: ΔAUC = +0.00002 ± 0.00002** (§4). On the same 100k jets the 4-seed ensemble gains +0.00034 ± 0.00006.
5. **fp16 on the edge Linear is harmless in the forward pass** (ΔAUC −3e-8), but the speed it could buy is small (§5).
6. **Ranking** (§6):
   1. Normalise the edge pre-activation, with readout standardisation bundled in.
   2. A longer schedule, or warmup–stable–decay.
   3. Distil the ensemble.

   EMA and dropping the penalty are passengers, not arms.
7. **Measured on GPU (§9), 2026-10-05.** Ranked first, edge normalisation gives ΔAUC −0.00004 ± 0.00010 and lower tail rejection. The log-knee gives ΔAUC +0.00000 ± 0.00008, with a suggestive rejection gain of +39 ± 30. Both are paired over three seeds on the probe protocol. **Conditioning does not move AUC.**

---

## 2. The coupled L2 penalty on the flow heads

### 2.1 What the code does

- `SO3CMessageSetClassifier.regularization_loss()` (`benchmarks/so3c_models.py:1100`) returns 1e-3 · Σ_r (‖W_r‖² + ‖b_r‖²) over the three `w_head` layers.
- `train.py:300` adds it to the cross-entropy. It is therefore an L2 term **inside** the loss, preconditioned by plain Adam.

**The suspicion** (Loshchilov & Hutter): Adam divides every gradient element by √v. For a parameter whose data gradient is tiny, the penalty gradient then becomes a full-size step toward zero, which pins the parameter.

### 2.2 Data gradient against penalty gradient, per round

**Measured** on 50 training batches of 256 jets (train rows 0–12,799, shuffled), float32 as trained. Each head has 136 parameters: 8 × 16 weights plus 8 biases.

| checkpoint | round | head RMS | ‖penalty grad‖ | ‖data grad‖ per batch | ‖mean data grad‖, 50 batches | sampling noise of that mean | cos(mean data grad, penalty grad) |
|---|---|---|---|---|---|---|---|
| headline60 s0 | 0 | 6.6e-3 | 1.6e-4 | 1.5e-2 | 4.4e-3 | 2.3e-3 | +0.10 |
| | 1 | 3.6e-2 | 8.4e-4 | 0.22 | 4.9e-2 | 3.4e-2 | +0.25 |
| | 2 | 4.7e-2 | 1.1e-3 | 0.45 | 6.0e-2 | 6.8e-2 | +0.11 |
| headline s2 | 0 | 5.4e-3 | 1.3e-4 | 5.9e-3 | 5.7e-4 | 9.6e-4 | +0.54 |
| | 1 | 3.6e-2 | 8.5e-4 | 0.18 | 2.8e-2 | 2.8e-2 | +0.26 |
| | 2 | 5.2e-2 | 1.2e-3 | 0.33 | 5.1e-2 | 5.1e-2 | −0.06 |

- The per-batch data gradient is **47–405×** the penalty gradient.
- The 50-batch mean data gradient is about as large as its own sampling noise. The heads therefore sit at a stationary point, to the precision 50 batches allow, and the penalty is 4–60× below that noise floor.
- The cosine is never close to −1. **The two are not equal and opposite.**

### 2.3 Per channel, Adam's normaliser, and the loss curvature

The rows aggregate two very different populations of output channels. The table below covers 20 batches (**measured**).

"Penalty / √v" is the median, over a head row, of |2λw| divided by the square root of Adam's stored second moment (`exp_avg_sq`). It is the fraction of each Adam step that the penalty contributes.

| channel state entering the round (median Hermitian \|z\|) | head row norm | penalty / √v | reading |
|---|---|---|---|
| alive: 0.2–8 | 2e-4 to 0.35 | **0.0007–0.025** | data dominates; the penalty is a 0.1–2.5% nudge per step |
| collapsed: ≤ 0.07 (s0: 4 channels in round 0, 5 in round 1, 1 in round 2; s2: 0 / 4 / 1) | 1e-13 to 2e-5 | **0.1–1.4** | the Loshchilov–Hutter regime: the penalty pins these rows to zero |

**What the collapsed channels mean.** Their states were already near zero, either because of the channel weights (round 0) or the channel mixing (later rounds). Neither of those is penalised. With z ≈ 0 the flow does nothing, whatever the head. The penalty pins heads that are irrelevant; it does not create dead channels.

**How far the penalty can move the optimum.** For a penalty λ‖w‖², the optimum moves by about |Δw|/|w| ≈ 2λ/h. Here h = uᵀHu is the curvature of the training cross-entropy along the head's own direction u = w/‖w‖. It was computed exactly with a Hessian–vector product, averaged over 20 batches (**measured**):

| round | h, headline60 s0 | h, headline s2 | implied shrinkage 2λ/h |
|---|---|---|---|
| 0 | 0.083 ± 0.041 | 0.024 ± 0.019 | ≈ 2–9% |
| 1 | 10.6 ± 4.7 | 1.1 ± 1.7 | ≲ 0.2% |
| 2 | −0.5 ± 2.8 | 1.1 ± 1.2 | not resolved (flat or indefinite) |

The round-0 heads sit in a nearly flat direction of the loss. Even so, 2λ = 2e-3 is 10–40× below its curvature, so the penalty trims them by a few percent and does not pin them.

### 2.4 Paired fine-tune with the penalty removed

**Measured.** Each arm starts from the 60-epoch checkpoint with Adam's stored state restored, and sees the same 300 training batches (rows 20,000 onward). Gradient clipping is at 1.0, as in training. Validation cross-entropy is measured on 5,000 validation jets.

| arm | λ | lr | steps | head RMS r0 / r1 / r2 (before: 0.0066 / 0.0358 / 0.0470) | round-0 flow p90, largest channel | val CE (before: 0.16527) | non-finite |
|---|---|---|---|---|---|---|---|
| penalty | 1e-3 | 3e-4 | 300 | 0.0070 / 0.0361 / 0.0468 | 0.0030 | 0.16691 | none |
| no penalty | 0 | 3e-4 | 300 | 0.0079 / 0.0362 / 0.0471 | 0.0033 | 0.16723 | none |
| no penalty, peak lr | 0 | 3e-3 | 150 | 0.055 / 0.121 / 0.075 | 0.0091 | 0.17400 | none |

- **Nothing breaks.** There is no NaN or Inf in any arm. The pre-clip gradient norm has median 0.61–0.80.
- **Removing the penalty barely moves the heads.** At lr 3e-4 the round-0 head RMS rises 19% instead of 6%; rounds 1–2 are unchanged; the round-0 flow is unchanged.
- **The leverage is weak.** At peak lr the round-0 heads grow 8×, yet the round-0 flow only goes from 0.003 to 0.009. That is the leverage problem of §2.5, measured directly.
- **Validation CE rises in every arm.** That is expected when restarting a non-annealed learning rate from an annealed minimum. The no-penalty arm ends 0.0003 above the penalty arm. That is one paired comparison with no noise estimate, so all it shows is that removing the penalty gives no visible improvement.

### 2.5 Why round 0 is quiet

The rotation generator is a = z_a × (1/n) Σ_b w_ab z_b, so |a| scales as |w| · |z|². The states entering round 0 are the soft-bounded lift times the channel weight: measured median Hermitian |z| 0.2–1.0 in live channels. Channel mixing then grows them to 1–8 entering rounds 1–2 (`gap_thesis.md` §2.3). Per unit of head weight, round 0 therefore rotates one to two orders of magnitude less, and the loss curvature along the round-0 heads is correspondingly small (§2.3).

Making round 0 rotate is an architectural question, not a regularisation one. Options include a per-round flow time, or applying the first channel mixing before the first flow. The flat curvature says the data does not currently reward it.

### 2.6 Verdict on question 1

- **Not binding.** A coefficient-0 arm has a **prior of about 0** (≤ +0.0001). It is not worth a GPU slot of its own.
- If another arm is being built anyway, setting the coefficient to 0 there is free.
- One caveat: without the penalty, nothing restrains the heads of collapsed channels. Under Adam a zero-signal parameter takes noise-driven steps of order lr. That is harmless while the channel stays empty, and it was seen as extra head-RMS growth in the no-penalty arms with no effect on the flow.

---

## 3. Do the other training defaults bind?

### 3.1 Gradient clipping at 1.0 (`train.py:29`)

All values **measured**: the global norm of the training-loss gradient, before clipping.

| where | batches | median | p90 | max | fraction > 1.0 |
|---|---|---|---|---|---|
| headline60 s0, trained weights | 50 | 0.62 | 1.01 | 1.24 | 12% |
| headline s2, trained weights | 50 | 0.52 | 0.76 | 0.95 | 0% |
| fresh initialisation, the first 150 steps at lr 3e-3 | 150 | 0.67 | — | 5.18 | 29% (28% of the first 50 steps, 22% of the last 50) |

The penalty's share of the norm is negligible: the data-only median is the same to 4 digits. Clipping binds mostly early, and then only on about a quarter of steps. Adam's update is nearly invariant to a global rescale, so clipping acts as an outlier down-weighting of single batches. **Not a lever.**

### 3.2 Tanh saturation in the edge layers

**Measured**: the fraction of edge-layer pre-activations, over real pairs including the beams and the self-edges as in training, with |x| > 2 (tanh′ ≤ 0.071) and |x| > 3 (tanh′ ≤ 0.0099).

| | round 0: \|x\|>2 / \|x\|>3 / mean \|x\| | round 1 | round 2 |
|---|---|---|---|
| fresh init, step 0 | 0 / 0 / — | 0 | 0 |
| fresh init, step 150 | 0.05% | 0.4% | 3.6% |
| headline s2, 30 epochs | 14% / 8% / 1.07 | 61% / 47% / 4.6 | 76% / 67% / 8.9 |
| headline60 s0, 60 epochs | 40% / 25% / 2.8 | 67% / 55% / 7.7 | 84% / 76% / 12.8 |

**Mechanism.** The state norms grow about 10× across rounds through the channel mixing (§2.5). The pair invariants grow with |z|², asinh compresses only logarithmically, and nothing normalises the pre-activation. As a result, in rounds 1–2 the 16 edge units act mostly as sign detectors on their inputs. Gradient through those units, and so to the edge weights and everything upstream of them, is suppressed 15–100×.

This is consistent with, though it does not prove, two things already in STATUS:
- the model is slow to train (60 epochs beat 30);
- adding width helps (w128s32, which widens `act_hidden` from 16 to 128, gave +0.00048): more units means more of them in the linear range.

The fix is the same change as `gap_thesis.md` idea 7.1(b): a per-pair LayerNorm over the 16 hidden units, or a learnable gain, before the tanh. It acts on invariants only, so exact covariance is kept.

### 3.3 Readout input scales (the 201 features into the readout MLP)

**Measured** over 12,800 training jets. The 10/50/90% quantiles of the per-feature std are 0.41 / 2.35 / 4.94 (s0) and 0.21 / 2.24 / 4.14 (s2). The max std is 8.1–8.3, and |mean|/std reaches 5.4–7.0.

| group | features | std range | notes |
|---|---|---|---|
| per-channel bivector invariants | 88 | 0.03–5.9 | 0.03 for collapsed channels |
| cross-channel z_tot products | 72 | 2.2–5.9 | |
| scalar-state mean and max | 16 | 1.0–8.3 | means up to 16.5 |
| Minkowski statistics | 7 | **1e-10 to 0.99** | the 3 m² moments are rounding noise (std 1e-10); mean ⟨p_a,p_b⟩ and its square sit in asinh's linear range (std < 1e-2) |
| beam scalar states | 16 | 0.6–2.7 | largest \|mean\|/std, 7.0 |
| asinh⟨P,b±⟩ | 2 | 0.91 | |

The spread is about 250× once the noise features are excluded. A fixed standardisation from training statistics, or BatchNorm1d, is cheap. But it **must** be combined with `--no-mass-input`: standardising the three m² moments would blow rounding noise of std 1e-10 up to unit variance.

The same readout produces extreme logits. On 100k test jets the logits run from −2,182 to +12. 2.4% of jets, all background, are below −50, and 3 jets are below −709, where e^{-logit} overflows float64. This does not affect the ranking: no signal jet scores below −50. It is the over-confidence behind the late rise in validation loss that STATUS attributes to calibration.

### 3.4 Scheduler (note only, not tested)

- `CosineAnnealingLR(T_max = epochs)` is stepped once per epoch (`train.py:309`). The learning rate is therefore a staircase: 60 plateaus of 4,731 steps at K=64 and batch 256.
- Epoch 1 runs at the full 3e-3 with no warmup.
- §3.1 shows the early gradient norms reach 5.2, and 29% of early steps are clipped. A short warmup would mostly remove that clipping.
- **Prior about 0.** Per-step cosine and warmup are hygiene, not levers.

---

## 4. Test-time augmentation over the detector's reflections

### 4.1 Which views are distinct

With the beams fixed, the model is exactly invariant only to rotations about the beam axis. The detector is statistically symmetric under three reflections:
- σ_x: p_x → −p_x, through a plane containing the beam;
- σ_z: p_z → −p_z;
- a relabelling of the two beam slots, S. The beams are both protons.

Two exact relations collapse that group:
- A rotation by π about y, applied to the jet *and* the beams, is an exact symmetry of the model and maps b₊ ↔ b₋. So **S ≡ σ_xσ_z**, and **σ_z + beam swap ≡ σ_x**. The latter is the "z → −z plus a beam-slot swap" view, and also the parity of the whole event.
- Parity of the jet alone is R_z(π)·σ_z, so **P ≡ σ_z**.

All four equivalences were checked on 2,000 jets with **max |Δlogit| = 0.0** exactly. The distinct views are therefore {id, σ_x, σ_z, σ_xσ_z}.

### 4.2 Result

**Measured** on the first 100,000 canonical test jets, 60-epoch model, with a paired bootstrap over jets (300 resamples):

| view or combination | AUC | Δ vs id | 1/ε_B @ 0.3 |
|---|---|---|---|
| id | 0.983207 | — | 1233 |
| σ_x alone; Δlogit std 0.015, corr 0.9999997 | 0.983207 | ≈ 0 | |
| σ_z alone; Δlogit std 5.0 on a logit scale of 17, corr 0.972 | 0.983208 | ≈ 0 | |
| id + σ_z (probabilities averaged) | 0.983227 | **+0.000021 ± 0.000015** | 1204 |
| all 4 views (probabilities) | 0.983227 | +0.000021 ± 0.000015 | 1204 |
| all 4 views (logits) | 0.983214 | +0.000008 ± 0.000009 | 1204 |
| *4-seed ensemble on the same jets (stored scores)* | 0.983545 | *+0.000338 ± 0.000056* | |

On 100k jets, rejection at 30% signal efficiency is decided by about 41 background jets, so the 1233 → 1204 change is one or two jets.

**Reading.**
- The model is almost exactly invariant to the improper reflection σ_x, even though the complex construction does not enforce it.
- It is **strongly** non-invariant to the z-mirror, which is the same as swapping the beam slots: the logit shift has std 5. Yet the AUC does not change, because the shifts sit in the extreme-logit tail of §3.3.
- **Verdict:** TTA costs nothing to train and about 2.5 GPU-minutes on the full test set, but it buys 6% of what the seed ensemble buys. **Not worth carrying.**

---

## 5. fp16 autocast on the edge Linear

**Precision, measured.** The test is forward-only, on 20,000 test jets, with the first edge Linear computed under autocast semantics: fp16 operands, fp32 accumulation, fp16 result.
- Δlogit std 0.013, p99 0.055, max 0.35. **ΔAUC −3e-8 ± 9e-7.**
- 25% of the edge layer's inputs are fp16 subnormals (< 6.1e-5) and 40% are exactly zero. That is the input-scale problem of `gap_thesis.md` §2.1 again: a quarter of the inputs are too small to affect the pre-activation at any precision.
- Training dynamics under fp16 (gradient underflow, loss scaling) were not tested.

**Speed (prior).**
- In the eager T4 profile (`results_profile/t4_47ec4ad`, base, K=64), all matrix kernels together (mm, addmm, bmm) are about 31% of CUDA time. The edge Linear is one part of that.
- Its inner dimension is 16, so it is bandwidth-bound, and fp16 at best halves its memory traffic.
- Expected step saving ≤ 5–10%, i.e. 3–6 more epochs per 60 at equal wall clock. Through the 30 → 60 epoch slope (itself only suggestive, audit #3), that is worth about +0.00003–0.00005 (prior).

**Cost and risk.** It needs a GradScaler, and a non-default flag excluded from the float64 golden guard. **Low value.** Revisit only if the input scale is fixed first (`gap_thesis.md` 7.1).

---

## 6. Generic candidates, ranked

**Protocol.** Costs are on the K=32 long-schedule screen: all 1.211M training jets, 60 epochs, compiled T4, about 3.4–3.5 h per base seed. ~~Arms are compared seed by seed against `results_long_seeds/base` seeds 0–2.~~ **Corrected (audit, plan_2026-10-10 §1 #1):** those bases are torch 2.10 and the card is 2.11; pairing across that change is not established, so every arm needs a **same-stack control** (K=32 long: about +3.5 h per seed on every cost here).

**Gains do not add up.** Ideas 1, 4 and `gap_thesis.md` 7.1 overlap.

| # | candidate | mechanism and fit | expected gain | cost |
|---|---|---|---|---|
| 1 | **Normalise the edge pre-activation** (per-pair LayerNorm or learnable gain before tanh), **with readout standardisation and `--no-mass-input`** | Measured: 61–84% saturation in rounds 1–2 (§3.2); readout feature scales spread 250× (§3.3). Acts on invariants only, so exact covariance is kept | **prior** 0 to +0.001 (shared with `gap_thesis.md` 7.1, not additive) | about 1.05–1.10× (estimate), about **7.5 GPU-h** for seeds 0 and 1 |
| 2 | **Longer schedule, or warmup–stable–decay with branched decays** | In-house: 30 → 60 epochs gave +0.00037 at K=64 — 2.2 standard errors (t on 2 df, p ≈ 0.15), **suggestive** (audit #3); a falling validation loss is weak evidence because cosine flattens at the end; the K=32 base's best validation epoch is 56–60 and it does not overfit. WSD measures the epoch curve in one run | **prior** +0.0001 to +0.0003 for 60 → 120 epochs | 120 epochs is about 7 GPU-h per seed at K=32; WSD with two branch decays about 9 h. Needs a constant-then-decay schedule and branch-from-checkpoint in the harness |
| 3 | **Distil the 4-model ensemble into one student** | Soft targets carry the ensemble's +0.00034 (measured on 100k jets) into a single model. Fits a small, under-fitting model | **prior** +0.0001 to +0.0002 | Teacher scoring of 1.21M training jets is about 8 GPU-min (0.095 s per 1,024 jets per model). The student must be K=64 because the teachers are: 5.2 h for 30 epochs, paired with seed 2 or the running `headline_t4` seed 0. Needs a soft-target loss in the harness |
| 4 | **EMA of weights** (SWA with a constant-learning-rate tail is similar and overlaps with #2) | Averages along one trajectory. `--keep-best` found the trajectory's end flat (best-validation weights beat the final ones by only +0.00001–0.00006). Trajectory averages are far more correlated than the independent seeds behind the +0.00034 | **prior** 0 to +0.0001 | about 0: a 22.8k-parameter copy per step. A **passenger** on every run, like `--keep-best`, not an arm |
| 5 | Drop the flow-head penalty | Measured not binding (§2) | **prior** about 0 | free; passenger only |
| 6 | Test-time augmentation | Measured (§4) | **measured** +0.00002 ± 0.00002 | no training; not worth carrying |
| 7 | fp16 on the edge Linear | Forward measured harmless; ≤ 5–10% faster (§5) | **prior** +0.00003–0.00005 via more epochs | flag, GradScaler, guard exclusion; low value |
| 8 | Warmup, or a different clip | Clipping binds on 29% of early steps, irrelevant later (§3.1) | **prior** about 0 | free |

**Update after §9.** #1 is measured null on AUC, and so is its companion, the log-knee. The order below is kept as written, for the record.

**Proposed order.**
1. Idea #1 as a single arm on seeds 0 and 1, with EMA and penalty 0 as passengers if the harness gets those switches anyway (about 7.5 GPU-h).
2. If #1 is positive, #2 on the winner. Normalisation may also change how many epochs the model needs, so #2 should be measured after #1, not before.
3. #3 when there is a K=64 slot.

---

## 7. Not recommended

| idea | why not |
|---|---|
| Label smoothing | Our metrics are ranking and tail metrics; smoothing compresses exactly the confident tail where 1/ε_B @ 0.3 is decided. It is also a regulariser, and the base model underfits (dropout and weight decay measured negative). |
| Focal loss | Not a proper scoring rule. It moves capacity toward the 50% decision boundary, while rejection at 30% signal efficiency is set among the most signal-like jets. No support on this benchmark in any paper read for `gap_thesis.md`. |
| AUC or pairwise ranking losses | Cross-entropy is proper, and its minimiser (the likelihood ratio) is already AUC-optimal. At AUC 0.983 the misordered pairs are few, so a pairwise loss trains on a noisy handful. |
| Mixup | Interpolating the 4-momentum sets of two jets is not a jet, nor covariant unless done on invariants. And it regularises a model that underfits. |
| SAM | About 2× the cost per step for a flatness regulariser. The model is under-trained (60 epochs beat 30), so the compute is better spent on epochs. |
| Weight decay, dropout | Measured negative: dropout 0.05 + weight decay 1e-4 gave −0.0011; AdamW 0.002 on w128s32 is **not shown to help** — −0.00011, but decay runs on torch 2.11 against controls on 2.10, so unpaired and inside the ±0.00027 spread (audit #2); the LorentzNet recipe also lost (SO3C_STATUS.md). |
| Temperature scaling for the over-confident logits (§3.3) | It does not change ranking, so it cannot change AUC or rejection. Calibration only. |

---

## 8. Follow-up, 2026-10-05 afternoon: the K=64 width result, the flow formalised, and ideas A–D

This section folds in a side-chat summary and tests its two concrete proposals on CPU:
- **A**, a jet-frame Hermitian norm;
- **B**, moment tensors for higher body order.

Same rules as above. Nothing was built and nothing was pushed in this part.

### 8.0 Does width transfer to K=64? (the kernels from commit 82235c4)

Both kernels finished on a Tesla T4 and passed the on-card guard ("1 passed"). Their outputs are archived in `results_k64/w64s24/` and `results_k64/headline_t4/` (JSON, scores, best-validation scores, checkpoint, kernel log).

| seed 0, K=64, 30 epochs, T4 | params | AUC | 1/ε_B @ 0.3 | 1/ε_B @ 0.5 | final val loss | s / epoch |
|---|---|---|---|---|---|---|
| headline_t4 (control) | 22,834 | 0.98311 | 1040.8 | 287.6 | 0.16483 | 492 |
| w64s24 | 60,834 | 0.98319 | 1035.5 | 284.4 | 0.16448 | 806 |
| **paired Δ** | 2.7× | **+0.00008** | −5 | −3 | −0.00035 | 1.64× |

- **Width does not transfer.** At the K=32 long schedule, w64s24 gained +0.00025 over its paired base seed. At K=64 and 30 epochs it gains +0.00008, which is one pair and inside the noise, for 1.64× the time.
- **Validation loss** is lowest at the final epoch in both runs, so `--keep-best` scores equal the final ones.
- **Reproducibility across software.** The rerun control scores 0.98311, against 0.98326 for the original seed-0 30-epoch run (P100, torch 2.5.1). The gap is −0.00014 AUC and −106 rejection, unrounded (audit #5): about one standard deviation. Across that stack change a shared seed behaved like an independent seed.
- **Software change that matters for pairing.** Kaggle's T4 image now ships **torch 2.11.0+cu128**; the probe and long-schedule baselines ran on 2.10. Whether CUDA `randperm` gives the same permutation across that change is unverified, so seed-pairing against `results_message/probe_seeds/base` needs torch 2.10 pinned on the card, or fresh controls (§8.7).

### 8.1 What our flow is (for the record)

- **Within a round**, the generator is a_a = (z_a × r_a)/(1 + |·|^½), with r_a = (1/n) Σ_b w_ab z_b. It is frozen and applied once in closed form, z ← exp(−T[a]_×) z with T = 1 (Rodrigues; `expm_so3c_apply`).
  - so(3,ℂ) acts on ℂ³ through its adjoint representation ([a]_× z = a × z = ad_a z), so the update is z ← Ad_{exp(−Ta)} z.
  - z·z, the Casimir, is conserved exactly (theory.md §4).
- **Across rounds**, a is recomputed from the current states of all particles after channel mixing, with separate weights per round. It transforms as a → Qa, which is what makes the model equivariant; an invariant-fed generator is not (theory.md §5, measured relative logit change 0.35–0.81).
  - Formally the three rounds are Lie–Euler steps of z′ = −a × z = z × (z × r) = [z, [z, r]], Brockett's double-bracket form.
  - But it is a discrete-depth network: three rounds with different weights, not a neural ODE.
  - The true continuous version, `SO3CInteraction`, is closed: 0.9735 at 9.5× the cost (STATUS).
- **Measured** (§2.5, §8.3): the round-0 flow is about the identity. The flow conserves z·z, and the trained states stay nearly simple (|z*|/|z·z|^½ = 1.00–1.2). So **the 4–8× growth of the invariant state size across rounds comes from the channel mixing**, not from the flow.

### 8.2 The professor's suggestions

The suggestions were Lie–Poisson networks, port-Hamiltonian neural ODEs, generators learned from trajectories, neural CDEs, Magnus expansions, L-conv, Casimirs/CLPNets, moment maps, automorphisms and Lie-group kernels. They mostly do not apply. Our "time" is depth, the data are static snapshots, and the measured bottlenecks are conditioning, saturation and possibly body order, not the flow's structural prior. Specifics, with what was checked:

| claim | status |
|---|---|
| A single-particle Hamiltonian H(z) on so(3,ℂ)* must be invariant, so H = h(z·z), ∇H ∝ z, and the Lie–Poisson flow z × ∇H vanishes | **verified** against theory.md §6 (the only covariant one-state vectors are f(invariants)·z) |
| The multi-particle Lie–Poisson variant rotates z_a about r_a ("precession") instead of about z_a × r_a | r_a → Q r_a, so it is covariant; a cheap generator variant. **Prior about 0**; an optional passenger, given that the current generator is barely used (§2.5) |
| Clebsch–Gordan: (1,0) ⊗ (1,0) = (0,0) ⊕ (1,0) ⊕ (2,0). We use z·w and z × w; only spin-2 is missing | correct. The spin-2 part, the traceless symmetric z wᵀ + w zᵀ, is exactly what idea B's moment tensors use. Their jet-level invariants add nothing (§8.4) |
| The coadjoint orbit is the complex quadric {z·z = c}, not S² | correct (two complex dimensions) |
| The only outer automorphism is parity | correct for the real form so(1,3) ≅ sl(2,ℂ)_ℝ: complex conjugation, i.e. parity. **Measured**: the model is nearly parity-invariant; the azimuthal mirror σ_x changes the logits with std 0.015 (§4) |
| LPNets, CLPNets, "Network of coadjoint orbits I", MoLiegroup, graded representation theory, Euler–Poincaré neural dynamics | **not verified**; no existence check was made |

### 8.3 Idea A: the jet-frame Hermitian norm

**The claim to correct.** The model's "Bounding" docstring and theory.md §10 ("Why the bound") say z "cannot be normalised away" because the only scalars are bilinear invariants. That is too strong. P is timelike and covariant, its stabiliser is the compact SO(3), and the Hermitian norm in the jet rest frame is an exact Lorentz invariant of (z, P). It is real, ≥ 0, and zero only for z = 0, so it is a genuine norm, null states included.

**Formula.** For z = E + iB and u = P/M = (γ, **b**):

  N(z, P) = |z*|² = 2 T^{μν}(F) u_μ u_ν = (1 + 2|**b**|²)|z|² − 2|**b**·z|² − 2iγ **b**·(z × z̄)

Here T is the Maxwell stress tensor, so N is twice the field energy density seen by the jet. z × z̄ is purely imaginary, so the last term is real. The sign of the Poynting term was fixed by test (below).

**Checks, measured in float64:**

| check | result |
|---|---|
| (i) invariance under random Lorentz transformations with P moving along (boost scales 0.5 and 2.0, 40 transformations, 400 states) | sign −: max relative change 3e-6 (generic), 6e-5 (null), at float64 conditioning for rapidities up to about 6. Sign +: fails by about 1e11 |
| (i) null states (z·z = 0 to 3e-14) | N > 0: min N/\|z\|² = 0.023 |
| (i) generic states | N ≥ \|z·z\| (min (N − \|z·z\|)/N = 0.09) |
| (i) raw lift z = bivec(p, P), massless p | N = \|z·z\| = ⟨p,P⟩² to 1.0e-10 |
| (ii) cross-check: boost to the rest frame with `expm_so3c(i·β)`, then the Hermitian norm | agrees to **2.4e-14** |
| (iii) float32, closed form in the lab frame, real jets (γ_jet: QCD 4.7 / 10.0 / 25.9 at 10/50/90%; top 3.4 / 4.5 / 7.9) | QCD: median 5e-4, **p99 0.30, max 63** relative error; QCD with γ > 15: median 1.6%, p99 124%. Top: p99 0.005 |
| (iii) float32 after boosting the inputs to the jet rest frame once, in float64 | **≤ 3.6e-7** for QCD and top alike |

The lab-frame closed form cancels terms of order γ²|z|², which costs about γ²(|z|_lab/|z·z|^½)² digits; the rest-frame Hermitian norm cancels nothing.

**The rest-frame route keeps exact invariance.** If B(P) is the standard boost to the rest frame, then B(ΛP)Λ = W·B(P) with W a Wigner rotation. The network is exactly rotation-covariant, so its output is unchanged. **A side benefit:** every invariant the model computes, including its own float32 z·z, becomes well conditioned. Today those carry about γ²-amplified rounding for boosted QCD jets.

**(iv) On both checkpoints**, the states entering each round, measured on 600 test jets:

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| median \|z*\| / \|z·z\|^½ over channels | 1.00 | 1.00 | 1.00–1.20 |
| median \|z*\|, live channels (s0 / s2) | 0.06–0.18 / 0.04–0.15 | 0.19–0.74 / 0.15–0.71 | 0.12–1.40 / 0.18–1.27 |
| Spearman(pair's mean \|pre-activation\|, Σ_c \|z*_a\|\|z*_b\|) (s0 / s2) | 0.39 / 0.26 | 0.80 / 0.73 | 0.68 / 0.76 |
| share of pre-activations with \|x\| > 2, by quartile of that product, Q1 → Q4 (s0) | 34% → 58% | 44% → 89% | 76% → 91% |

**Reading.**
1. On the trained states the jet-frame norm equals |z·z|^½ to within 0–20%: the states stay nearly simple. **Replacing the 1 + |z·z|^½ soft bound with 1 + |z*| would be close to a no-op.**
2. The invariant state size grows 4–8× from round 0 to rounds 1–2, through the channel mixing (§8.1). In rounds 1–2 that size tracks edge saturation closely.
3. The version of A that would act is therefore a **per-channel normalisation of the states by their jet-frame RMS in every round**. It targets the same saturation as the per-pair LayerNorm, at its source, so the two share one gain. It needs the rest-frame preprocessing for float32 accuracy.

**Prior:** it shares the gain of `gap_thesis.md` 7.1 and §3.2, so it is not additive. **Cost:** a float64 boost of B × K × 4 inputs once per batch, then O(B·C·N) per round; about 1.0× (estimate).

### 8.4 Idea B: moment tensors and body order, closed by a stacking test

**The idea.** Our messages are 2-body, and the near-idle flow adds little beyond that. MACE-style moment tensors would add 3- and 4-body information. At jet level, S = Σ_a z_a z_aᵀ on the raw lift.

**Verified.** On the raw lift, tr S^k equals M^{2k} times the trace of the rest-frame quadratic momentum tensor Σ p*p*ᵀ, to relative 1.5e-12, 1.1e-10 and 3.4e-8 for k = 1, 2, 3 (200 jets, float64). The characteristic-polynomial invariants are the rest-frame sphericity.

**Stacking test.** The same protocol as `benchmarks/diag_multiplicity_stacking.py`: the fixed halves from rng 12345, HistGradientBoosting (200 iterations, lr 0.05, 15 leaves), fit on one half and scored on the other. Features come from the leading-64 constituents, boosted to the jet rest frame in float64. 8 of 404,000 jets have no usable rest frame and get NaN.

| features stacked on the logit | ΔAUC, headline60 (halves) | ΔAUC, 4-seed ensemble (halves) |
|---|---|---|
| c₃ = tr S³/(tr S)³, d = det S/(tr S)³ | −0.00001 (−0.00001, −0.00001) | +0.00001 (−0.00000, +0.00001) |
| sphericity eigenvalues ν₁…ν₃ | −0.00001 (−0.00002, +0.00000) | +0.00001 (+0.00000, +0.00001) |
| per-node triangles t_a = p*_aᵀS²p*_a/(\|p*_a\|²(tr S)²): mean and max | −0.00001 (+0.00000, −0.00002) | +0.00001 (+0.00001, +0.00002) |
| linear (energy-flow) tensor Σ p*p*ᵀ/\|p*\|, eigenvalues | +0.00000 (−0.00000, +0.00001) | +0.00003 (+0.00003, +0.00004) |
| all of the above | +0.00000 (−0.00000, +0.00001) | +0.00002 (+0.00001, +0.00003) |
| *control: one pure-noise feature* | *−0.00003* | |

On their own these features reach only AUC 0.51–0.69. **The jet-level 3- and 4-body shape invariants of the raw lift carry nothing the model lacks.** The ensemble's +0.00003 is the size of the noise control.

**Correction, which also applies to STATUS.** The rejection deltas of this protocol are an **artifact**:
- With only the logit as input, the boosted trees output 299 distinct values, because features are binned into at most 255 levels. That ties the tail: the stacked score-only rejection is 940, against the model's own 1141.
- Any added feature breaks the ties. A pure-noise feature "raises" rejection by +125.

So this protocol's rejection gains (+52 to +237 here, and the "+65 rejection" of STATUS's multiplicity row) carry no information. Only its AUC deltas do.

**Literature.** MACE (arXiv:2206.07697) supports body order only in a figure: Fig. 2 (left), learning-curve slopes s = −0.33 / −0.51 / −0.64 for correlation order ν = 1 / 2 / 3, aspirin forces from rMD17. Its tables compare whole models; none isolates body order. That evidence is for molecular force fields with a local cutoff, and the transfer to jets is unverified. Since the CPU test closed B, no jet-tagging event-shape paper was searched.

**What stays open.** This tests jet-level invariants of the raw lift, not learned, weight-dependent per-node tensors T_a = Σ_b w_ab z_b z_bᵀ inside the network. Those remain untested. The null above lowers their **prior to about 0 to +0.0002**, and they would cost O(K²C·6) per round.

### 8.5 Idea C: invariant theory, true with one qualifier

- For several z under SO(3,ℂ), the polynomial invariants are generated by the bilinears z_a·z_b and the triple products det[z_a z_b z_c] (first fundamental theorem).
- For the real Lorentz group acting on (z, z̄), the invariants are the holomorphic ones together with their conjugates. So without P, Re and Im of the bilinears plus the triple products are complete.
- Triple products are parity-odd, and their squares are Gram determinants, i.e. polynomials in the bilinears. So the parity-even invariants we use are complete.
- **Qualifier:** with the jet vector P, which lives in the (½,½) representation, mixed non-holomorphic invariants z̄ᵀA(P)z appear. A's |z*|² and the jet-frame Hermitian products z̄*_a·z*_b are examples. They are new only once the states are non-simple, and on the trained states they nearly coincide with the holomorphic ones (§8.3).

The conclusion stands: **stop hunting for new invariants; improve how they are conditioned (A, §3, `gap_thesis.md` 7.1) and combined (B, which measured null at jet level).**

### 8.6 Idea D: constrained channel mixing

The growth comes from M_r (§8.1), so constraining M_r is the direct alternative to A.

- **Correction.** A complex-orthogonal M (Mᵀ M = I, e.g. M = exp of a complex antisymmetric matrix) conserves only the complex bilinear Σ_c z_c·z_c. That group is non-compact, so individual jet-frame norms can still grow hyperbolically.
- The constraint that bounds them is a **unitary** M (M = exp(iH), H Hermitian on the channel index). It conserves Σ_c |z_c|² in every frame, so Σ_c |z*_c|² as well.
- Any constant channel matrix commutes with the SO(3,ℂ) action, so covariance is kept either way.
- It also cannot make a channel collapse to zero without moving its norm into the other channels.

**Prior:** an alternative to A, not an addition; it shares the same gain. **Cost:** a matrix exponential of an 8×8 per round per step, negligible.

### 8.7 What this means for the screen

| candidate for arm 2 | CPU verdict |
|---|---|
| B: moment tensors | **closed** at jet level (§8.4) |
| A as a soft-bound swap | **about a no-op** on trained states (§8.3) |
| A as per-round state normalisation, or D | supported as a mechanism, but aimed at the same rounds-1–2 saturation as arm 1 (LayerNorm); the overlap would muddy attribution |
| **log-knee** (learnable asinh(e^κ·x) on the pair and node invariants, `gap_thesis.md` 7.1(a)) | aimed at **round 0**, where the inputs are tiny (95–99% in asinh's linear range) and saturation is low (14–40%), i.e. the part that LayerNorm on the pre-activation does not fix. **Complementary to arm 1** |

**Recommendation:** arm 1 = per-pair LayerNorm before tanh; arm 2 = log-knee.

**Pairing caveat.** The probe baseline `results_message/probe_seeds/base` ran on torch 2.10; the card now has 2.11. Either pin torch==2.10.0 (cu128) in the screen notebooks, about 2–3 minutes of install per kernel, and assert the version on the card; or add three fresh base seeds (about 1.4 GPU-h). Pinning fits the 4.85 h left; fresh controls make it tight (2 arms × 3 seeds × about 0.5 h + 1.4 h ≈ 4.4 h).

*Decided 2026-10-05: fresh controls on torch 2.11 — the safe design; whether reuse would have paired is not shown (audit #6).*

---

## 9. The conditioning screen (GPU, 2026-10-05)

**Setup.**
- Code: `2303568`.
- Notebooks: `notebooks/kaggle_probe_screen_{base,edgenorm,logknee}.ipynb`.
- Protocol: the probe protocol (K=32, 400k training jets, 20 epochs, batch 256, float32, `--compile`), seeds 0–2 per arm.
- Software, every kernel: Tesla T4, torch 2.11.0+cu128 pinned and recorded in each result's `env`, on-card golden guard "1 passed".
- Each arm is paired seed by seed with a **fresh** base control on the same build.
- Results, checkpoints and kernel logs: `results_message/probe_screen/`.

| kernel | session | per seed (h) |
|---|---|---|
| base | 1.41 h | 0.424–0.462 |
| edgenorm | 1.40 h | 0.425–0.448 |
| logknee | 1.35 h | 0.410–0.435 |

On the GPU, the LayerNorm costs nothing measurable.

### 9.1 The fresh control, and why the old baseline could not be reused

| control | AUC | 1/ε_B @ 0.3 |
|---|---|---|
| base, torch 2.11 (this screen) | 0.98112 ± 0.00002 | 817.6 ± 11.4 |
| base, torch 2.10 (`probe_seeds/base`, 2026-09-20) | 0.98106 ± 0.00017 | 806 ± 58 |

For the same seed, the epoch-1 and epoch-2 training losses differ by up to 1.6e-3 between the two builds. **Corrected (audit #6):** epoch-1 losses cannot tell same-seed from different-seed runs — base seeds 0 and 2 agree to 0.00008 in epoch-1 validation loss — so this does not show that the old runs would not have paired. The fresh control was the safe design; "an old run is not a valid pair" is **not shown**. From now on a batch-order fingerprint in every result (plan C2) settles pairing directly.

### 9.2 Edge LayerNorm (`--edge-norm`): no AUC gain, slightly worse tail rejection

| seed | ΔAUC | Δ 1/ε_B @ 0.3 | Δ 1/ε_B @ 0.5 | Δ final val loss |
|---|---|---|---|---|
| 0 | −0.00001 | −39.6 | +3.0 | −0.00007 |
| 1 | −0.00015 | −50.5 | −6.1 | +0.00028 |
| 2 | +0.00006 | −35.4 | −3.3 | −0.00009 |
| **mean ± sd** | **−0.00004 ± 0.00010** (t = −0.6) | **−41.8 ± 7.8** (t = −9.3) | −2.1 | +0.00004 |

**Mechanism check** (measured on CPU from the seed-0 checkpoints, 2,000 test jets; they reproduce the stored scores to 1.5e-4). The share of tanh inputs with |x| > 2:

| round | base | edge-norm |
|---|---|---|
| 0 | 4% | 12% |
| 1 | 34% | 15% |
| 2 | 61% | 31% |

The LayerNorm halves the saturation in rounds 1–2. It does not remove it: the learned gains scale part of it back up, and round 0 rises.

**Reading.**
- Halving the saturation leaves AUC unchanged. **Saturation is not what limits the model on this protocol.**
- The rejection at 30% signal efficiency drops on all three seeds by 35–51, i.e. about 5%. The tail gets slightly worse.
- **Caveat:** at 20 epochs on 400k jets the base saturates less (4/34/61%) than the 60-epoch K=64 headline (40/67/84%), so a long-schedule effect is not excluded. But a null in AUC and a consistent loss in rejection are no case for spending a long-schedule slot on it.
- **Verdict: closed** on the probe protocol. It also lowers the prior for state normalisation (§8.3 A) and unitary mixing (§8.6 D), which target the same saturation through the state size.

### 9.3 Log knee (`--log-knee`): no AUC gain; rejection +39 ± 30, suggestive only

| seed | ΔAUC | Δ 1/ε_B @ 0.3 | Δ 1/ε_B @ 0.5 | Δ final val loss |
|---|---|---|---|---|
| 0 | +0.00004 | +26.5 | +9.7 | +0.00022 |
| 1 | +0.00006 | +17.2 | −7.3 | −0.00026 |
| 2 | −0.00009 | +73.3 | −5.4 | +0.00073 |
| **mean ± sd** | **+0.00000 ± 0.00008** (t = 0.08) | **+39.0 ± 30.1** (t = 2.25) | −1.0 | +0.00023 ± 0.00050 |

The arm scores 0.98113 ± 0.00009 AUC and 856.5 ± 35.8 rejection, against the control's 0.98112 ± 0.00002 and 817.6 ± 11.4. It ran slightly faster than the control (0.410–0.435 h per seed).

**Learned knees** (initialised at 4.0), read from the three checkpoints:

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| κ_pair | 4.24–4.29 | 4.24–4.43 | 3.98–4.03 |
| κ_node | 3.88–3.98 | 3.41–3.66 | 3.21–3.36 |

κ_q ends at 3.90–4.01. **The knees barely moved:** training is close to indifferent to where they sit.

**Reading.**
- Moving the compression knee into the data's range does **not move AUC**.
- The rejection at 30% signal efficiency rises on every seed: +17 to +73, about 5% on average. With t = 2.25 on two degrees of freedom (p ≈ 0.08 one-sided) and a tail statistic decided by about a dozen background jets, this is **suggestive, not established**.
- It costs nothing (+7 parameters, no measurable time), so it can ride along in a later run. It does not justify a slot of its own, since the target is AUC.

### 9.4 What the screen says about the gap

**Both conditioning hypotheses are measured null on AUC on the probe protocol:**
- the input knee: `gap_thesis.md` 7.1(a), measured in its §2.1;
- the edge saturation: 7.1(b), and §3.2.

Each was the mechanism behind the top-ranked cheap idea, and each is real: the knee sits where the data are, and the LayerNorm halves the saturation. Neither changes the ranking power of the model. The two switches also move the high-purity tail in opposite directions (−42 and +39), each about 5%, while AUC stays within ±0.00004. On this protocol, at this size of effect, AUC is insensitive to how the invariants are conditioned.

The remaining candidates from `gap_thesis.md` §7 are capacity and structure, not conditioning:
- the pair latent at the long schedule (probe +0.00012);
- depth;
- Lion as a check on whether training is the limit;
- a longer schedule;
- ensemble distillation (§6 here).

**Budget.** About 1.41 + 1.40 + 1.35 = 4.16 GPU-h of the 4.85 h left this window. About 0.7 h remain until the reset on Saturday 2026-10-10.

---

## Appendix A. Diagnostic snippets

The scripts lived in the session scratch folder. This is their core, enough to rerun them from the worktree root. `common.py` holds the shared pieces:

```python
# common.py: rebuild the training inputs exactly as benchmarks/datasets.py does
scale = sqrt(mean of squares over all 4 components of real constituents, leading 64, all train jets)  # 37.8997 GeV
def raw_jets(split, start, n):    # parquet -> (n, 64, 4) leading-64-by-pT, as select_leading_constituents
    ...                           # pT = hypot(px, py); keep pT > 0; argsort(-pT)[:64]; zero-pad
def model_input(p4, scale):       # (n, 64, 5): p4 / scale, mask in column 4
def load_model(name):             # SO3CMessageSetClassifier(channels=8, beams=True, dtype=float32) + state dict
```

```python
# D1/D2: data vs penalty gradient on each w_head, pre-clip norm, tanh saturation, readout inputs
for b in batches:                                         # 50 x 256 training jets
    model.zero_grad(); ce = F.cross_entropy(model(x[b]), y[b]); ce.backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters()}
    pen = {h: 2e-3 * params[h] for h in heads}            # gradient of 1e-3 * ||w||^2
    total_norm = sqrt(sum(||g[n] + pen.get(n, 0)||^2))    # what clip_grad_norm_ sees
# per round: ||pen||, ||g|| per batch, ||mean_b g||, its sampling noise sqrt(sum var / NB), cos(mean g, pen)
# Adam: ck['optimizer']['state'][i]['exp_avg_sq'] for w_head -> median |pen| / sqrt(v) per head row
# saturation: monkeypatch SO3CMessageSetClassifier._edge_pre to record its output; |x|>2 over real pairs
# readout: model.mlp.register_forward_pre_hook -> (B, 201) inputs; per-feature mean and std
```

```python
# D1b: curvature of the mean CE along each head's own direction (exact Hessian-vector product)
g = torch.autograd.grad(ce, head_params, create_graph=True)
u = cat(W, b) / ||cat(W, b)||;  gu = <g, u>
hv = torch.autograd.grad(gu, [W, b]);  h = <hv, u>          # averaged over 20 batches; shift ~ 2*lam/h
```

```python
# D3: paired fine-tune from headline60 with the stored Adam state, same 300 batches per arm
opt = torch.optim.Adam(model.parameters()); opt.load_state_dict(ck['optimizer']); set lr
loss = CE + lam * sum(||W_r||^2 + ||b_r||^2); loss.backward()
clip_grad_norm_(model.parameters(), 1.0); opt.step()        # arms: (1e-3, 3e-4), (0, 3e-4), (0, 3e-3)
# then: head RMS per round, val CE on 5,000 validation jets, round-0 |dz|/|z| p90 per channel
```

```python
# D5: TTA views (beams are the model buffer beam_p4; a slot swap is beam_p4[[1, 0]])
sx: px -> -px;  sz: pz -> -pz;  sxz: both;  Rz(pi): px, py -> -px, -py;  P: all three
checks (max |dlogit| = 0.0): Rz(pi) == id;  P == sz;  swap == sxz;  sz + swap == sx
TTA = mean over views of softmax probabilities (ensemble convention); paired bootstrap over jets
```

```python
# D6: fp16 edge Linear, autocast semantics emulated exactly
h16 = lambda t: t.half().float()   # fp16 storage, fp32 accumulate
pre = F.linear(h16(pair), h16(W)[:, :o], h16(b)) + node shares computed from h16 operands; return h16(pre)
# D4: fresh init (torch.manual_seed(0)), Adam 3e-3, clip 1.0, batch 256, 150 steps; log pre-clip norms
```

## Appendix B. Snippets for section 8

```python
# D7 (idea A): jet-frame Hermitian norm and its checks
def jet_norm(z, P):                        # z (..., 3) complex, P (..., 4) real timelike
    u = P / sqrt(P0^2 - |P|^2);  g, b = u[..., 0], u[..., 1:]
    return (1 + 2|b|^2) |z|^2 - 2 |b.z|^2 - Re(2i g b.(z x conj z))       # sign fixed by test (i)
# (i)  random_lorentz_pair(rot 1.0, boost 0.5 | 2.0): compare jet_norm(z @ Q.T, P @ Lam.T) with jet_norm(z, P),
#      for generic and null z (x + iy, x orthogonal to y, |x| = |y|); raw lift bivector_lift(p, P): jet_norm == |z.z| == <p,P>^2
# (ii) beta with lorentz_matrix(0, beta) @ P = (M, 0, 0, 0); |expm_so3c(1j * beta) @ z|^2 == jet_norm(z, P)
# (iii) float32 lab-frame closed form vs float64, and float32 after boosting the inputs to the rest frame in float64:
#      E* = g(E - b.p),  p* = p + ((g - 1)(b.p)/b^2 - g E) b,  with b = P_vec / P0
# (iv) monkeypatch expm_so3c_apply to record z entering each round; cast with .to(torch.complex128)
#      (NOT .double(), which silently drops the imaginary part); Spearman of the pair's mean |pre| vs sum_c |z*_a||z*_b|
```

```python
# D8 (idea B): rest-frame shape invariants of the raw lift, stacked as in diag_multiplicity_stacking.py
pv = rest-frame 3-momenta of the leading 64 (float64);  S = sum_a pv_a pv_a^T
nu = eigvalsh(S)[::-1] / tr S;  c3 = sum nu^3;  d = prod nu
t_a = pv_a^T S^2 pv_a / (|pv_a|^2 (tr S)^2) -> mean, max;  Theta = sum_a pv_a pv_a^T / |pv_a| -> normalised eigenvalues
check: tr(sum_a z_a z_a^T)^k == M^(2k) tr S^k on the lab-frame bivector_lift (k = 1, 2, 3)
half = default_rng(12345).permutation(n) < n // 2;  HistGradientBoostingClassifier(200, 0.05, 15 leaves)
control: [logit, N(0, 1) noise] -> dAUC -0.00003, "d rej" +125 (the binning artifact)
```
