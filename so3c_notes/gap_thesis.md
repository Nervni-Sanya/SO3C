# Gap thesis: what the literature offers against the 0.0030 AUC gap to PELICAN

Written 2026-10-04 in a single chat, with no agent fan-out. The target is PELICAN at 0.9870. Our best single model scores 0.98363 and our best ensemble 0.98396.

The verification rules for this note:

- Every literature number below was read from the full text of the paper, never from an abstract. The text was extracted from the arXiv PDF with pypdf. Each number carries the table, section or page it came from.
- Every number about our own model is marked **measured**. Those were taken read-only on 2026-10-04 from local data and stored checkpoints, with no code changed. The snippets are in the appendix.
- Nothing here was run on Kaggle.

| paper | arXiv version read | code read |
|---|---|---|
| PELICAN, Bogatskiy et al. | 2307.16506v4 | github.com/abogatskiy/PELICAN, master |
| LorentzNet, Gong et al. | 2201.08187v6 | github.com/sdogsq/LorentzNet-release, master |
| ParT, Qu, Li, Qian | 2202.03772v3 | — |
| L-GATr for all of the LHC, Brehmer et al. | 2411.00446v3 | — |
| Economical jet taggers (L-GATr-slim), Petitjean et al. | 2512.17011v2 | — |
| LLoCa, Spinner et al. | 2505.20280v2 | — |
| What do Lorentz-equivariant jet taggers learn?, Agarwal, Khare, Kumar | 2606.21790v1 | — |

PDF page numbers below are the physical pages of the arXiv PDF. The printed page number is one less for PELICAN.

---

## 1. Bottom line

- **Nothing in this literature shows +0.003 AUC for an architecture like ours.** Every equivariant tagger trained from scratch on this dataset, from 48k to 2.0M parameters, lands at 0.9866–0.9870 (§6). The only results above 0.9870 come from JetClass pretraining, which is a different comparison class.
- **What PELICAN's 1k model has that ours lacks** is not the lift and not the parameter count. There are three differences:
  1. All of PELICAN's parameters sit in the equivariant core. About 75% of ours sit in the readout MLP.
  2. PELICAN carries a rank-2 pair state through every block. We collapse pairs to nodes every round, and our covariant flow is almost the identity in round 0.
  3. PELICAN feeds log-compressed inputs in GeV and normalises every block. Our inputs sit in the linear regime of asinh, and the network has no normalisation layer anywhere.
- **The bivector hypothesis does not hold up** (§3). arXiv:2606.21790 shows that the bivector channels inside an L-GATr whose inputs are vectors are redundant. It says nothing about a bivector input lift. For massless constituents, our lift loses no information.
- **Ranked ideas** (§7):
  1. Input conditioning and normalisation: new, costs no runtime, untested.
  2. The pair latent at the long schedule: already implemented, never run there.
  3. PELICAN's N^α aggregation: free.
  4. Depth, once idea 1 is in.
  5. Lion, as a check on whether training is the limit.
- **Realistic expectation:** +0.0005 to +0.0015 AUC if ideas 1–3 work and stack. Closing the full 0.0030 is not expected from this literature on this protocol.

## 2. Measured on our model

All of this was measured read-only on 2026-10-04. The snippets are in appendix B.

### 2.1 Where the inputs sit relative to the compression knee

- **Global input scale: 38.0 GeV.** This is the RMS over all four components of the real constituents among the leading 64, from 30k training jets. Over all 1,211,000 training jets it is 37.90 GeV ([`gap_generic_ml.md`](gap_generic_ml.md)). None of the conclusions here change. The loader divides every momentum by it (`benchmarks/datasets.py:538`).
- **Minkowski pair products ⟨p_a,p_b⟩ in model units**, from 2000 test jets with all pairs:

  | quantile | 1% | 10% | 25% | 50% | 75% | 90% | 99% |
  |---|---|---|---|---|---|---|---|
  | ⟨p_a,p_b⟩ | 3.4e-6 | 4.6e-5 | 1.6e-4 | 6.6e-4 | 2.8e-3 | 1.1e-2 | 9.9e-2 |

  99.0% of pairs are below 0.1. Below 0.1, asinh(x) agrees with x to 0.2%.
- **The edge invariant the model actually reads**, Re s_ab = Re z_a·z_b after the invariant soft bound (1500 test jets): the 1–99% range is −0.17 to +0.17, and **94.9% of pairs have |s_ab| < 0.1**.
- **The same products in GeV²:** the median is 1.0 GeV², the 1–99% range is 5.1e-3 to 143 GeV², and **50.7% of pairs are below 1 GeV²**. PELICAN and LorentzNet feed raw GeV into log(1+x) (§4.1, §5). They therefore log-compress about half of all pairs. **We log-compress almost none.**
- **Angles in the jet rest frame:** 1−cosθ*_ab has median 1.00. Only 2.3% of pairs are below 0.01. Our edge invariant is ∝ E*_a E*_b cosθ*_ab (§3.3), so collinear pairs are not squeezed against a ceiling. The scale problem is in the energies, not in the angles.

### 2.2 What training did to the first edge layer

The edge layer is `nn.Linear(64, 16)`. Its default initialisation is U(±1/8), which has RMS 0.072.

| checkpoint | epochs | pair-invariant columns, RMS (rounds 0/1/2) | node-invariant columns | h columns |
|---|---|---|---|---|
| beams_k64 seed 0 | 30 | 0.70 / 0.88 / 0.95 | 0.44 / 0.42 / 0.57 | 0.42 / 0.86 / 0.67 |
| beams_k64 seed 1 | 30 | 1.56 / 0.58 / 1.07 | 0.61 / 0.64 / 0.53 | 0.43 / 0.89 / 0.69 |
| headline seed 2 | 30 | 1.08 / 0.89 / 0.92 | 0.68 / 0.55 / 0.56 | 0.58 / 0.78 / 0.63 |
| headline60 seed 0 | 60 | 1.63 / 1.26 / 1.42 | 0.95 / 1.05 / 0.80 | 1.03 / 0.91 / 0.86 |

Every block grows 6–23× over initialisation, and is larger again at 60 epochs. The pair block is usually the largest. This is **consistent with** Adam's fixed step size slowly scaling up weights on small inputs. That could be part of why the model kept improving from 30 to 60 epochs. **It is not proof:** the growth is general, not specific to the pair block. (Audit, plan_2026-10-10 §1 #3: the 30 → 60 gain itself, +0.00037, is 2.2 standard errors, t on 2 df, p ≈ 0.15 — suggestive, not established.)

### 2.3 The covariant flow is used sparingly

The size of the flow step was measured as |Δz|/|z| per channel and round, on 600 real test jets. The loaded 60-epoch model scores AUC 0.9869 on these jets, a sanity check.

| | round 0 | round 1 | round 2 |
|---|---|---|---|
| headline60 (seed 0, 60 ep) | p90 ≤ 0.003 in **every** channel | 2 of 8 channels move: medians 6.7%, 3.3% | 4 of 8 move: medians 4.5%–17% |
| headline (seed 2, 30 ep) | p90 ≤ 0.002 in every channel | 2 of 8 move: 6.9%, 1.3% | 3 of 8 move: 5.8%–13% |

**Round 0 is the identity to within 0.3%.** After round 0 every channel is still a complex multiple w_c·z_a of the same lift. The round-0 channel mixing keeps it that way. So the round-1 edge features are complex rescalings of the round-0 invariants, and new pair information enters only through the few channels that rotate in rounds 1–2.

The weight heads `w_head` that drive the flow have RMS 0.0014, 0.0034 and 0.0054 in the three 30-epoch checkpoints and 0.0068 in the 60-epoch one. They are zero-initialised and penalised at 1e-3·‖w‖². [`gap_generic_ml.md`](gap_generic_ml.md) §2 shows the penalty is not what keeps them small in live channels. Round 0 is quiet because the rotation generator is quadratic in the state and round-0 states are small.

In the 60-epoch run, **3 of the 8 channel weights collapsed**: |w_c| = 0.0005, 0.0006 and 0.0011, with a fourth at 0.073. In the 30-epoch checkpoints the smallest |w_c| is 0.33–0.51. Those channels can be repopulated only through mixing.

### 2.4 Where our 22,834 parameters are

| part | parameters | share |
|---|---|---|
| readout MLP 201 → 64 → 64 → 2 | 17,218 | 75% |
| edge layers (3 × 64 → 16) | 3,120 | 14% |
| node MLPs, message heads, weight heads, h_init, channel mixing and weights | 2,496 | 11% |
| **equivariant core** | **5,616** | **25%** |

PELICAN's output MLP "is just one layer that mixes 132 channels down to 2 classification weights" (2307.16506 §4.2, PDF p.12). The usual "1k PELICAN matches our 22.8k" comparison is really **1k against our 5.6k core**, a 5× gap rather than 23×. That still says the core is inefficient.

---

## 3. Q2: arXiv:2606.21790, bivectors, and our lift

### 3.1 What the paper is and what it measured

It is a workshop paper (AI4Physics@ICML 2026; BITS Pilani). It reproduces L-GATr with 12 blocks, C_mv = 16 and C_s = 32, over 3 seeds (§3.2, PDF p.3). The ablations zero grade groups of the **hidden** multivectors. The input embedding is left alone (§3.3).

| quantity | value | source |
|---|---|---|
| L-GATr AUC (their reproduction) | 0.9869 ± 0.0001 | Table 1, p.3 |
| zero bivector (G2) at every layer | ΔAUC 0.001 (+0.004 / −0.001) | Table 2, p.6 |
| zero vector-like (G1+G3) | ΔAUC 0.239 (+0.577 / −0.027); seeds 0.809, 0.239, 0.219 | Table 2, §6.1 |
| zero scalar-like (G0+G4) | ΔAUC 0.118 (+0.056 / −0.082) | Table 2 |
| keep only G2 | ΔAUC 0.337 | Table 2 |
| vector-like ablation by layer | largest at the input stage (mean 0.422), decaying with depth; G2 < 0.002 at every depth | §6.2, p.6 |
| **retrained with G2 disabled** | **0.9865 ± 0.0002** vs 0.9867 ± 0.0003 | §6.3, p.6; App. E.1 |
| linear probe on G2 invariants alone, top/QCD | **AUC 0.966** [0.961, 0.971] (vector-like 0.982, scalar-like 0.986) | Table 4, p.15 |
| five independent grades: G2 / G1 / G0 | 0.003 [0.001, 0.005] / 0.120 / 0.039 | Table 7, p.19 |
| L-GATr-slim, zero all vectors | ΔAUC 0.585 | Table 6, p.19 |

The paper's own L-GATr baseline differs between places: 0.9869 in Table 1, 0.9867 in §6.3 and 0.986 in Table 7.

### 3.2 Why it does not explain our gap

1. **It tests something different.** In L-GATr the four-momenta enter as grade-1 vectors (§2.2, p.2). Bivectors arise only inside the network. "G2 can be dropped" therefore means "an optional internal grade is redundant when the inputs are vectors". The paper never trains a model whose input lift is a bivector, and it does not claim bivectors are less informative. Its G2 invariants alone still probe to AUC 0.966.
2. **L-GATr's own best configuration encodes the beam as a bivector.** The default reference is the x–y plane bivector x^B_12 = 1 plus the time vector: 0.9870 ± 0.0001 / 2240 ± 70. Vector beams x^V_3 = ±1 plus time give 0.9869 ± 0.0001 / 2293 ± 141 (2411.00446 Table 3, PDF p.13). Both representations work.
3. **Our lift loses nothing for massless constituents.** With z_a = bivec(p_a, P), M² = ⟨P,P⟩ and ⟨p_a,p_a⟩ = 0:
   - z_a·z_a = ⟨p_a,P⟩².
   - Re z_a·z_b = ⟨p_a,P⟩⟨p_b,P⟩ − ⟨p_a,p_b⟩M².
   - Im z_a·z_b = 0 (theory.md §9).
   - M² = Σ_a ⟨p_a,P⟩.

   So ⟨p_a,P⟩ = +√(z_a·z_a), M² follows, and then every ⟨p_a,p_b⟩ follows: **PELICAN's entire input Gram matrix is recoverable from our invariants.** In the jet rest frame, z_a = M·p⃗*_a, the constituent's 3-momentum there. The "component along P" that the lift drops is the rest-frame energy E*_a, and masslessness fixes it as |p⃗*_a|. The beams are massless too.
4. **We measured it.** The vector channel, which feeds ⟨p_a,p_b⟩ and LorentzNet's coordinate update directly, gave a paired Δ of −0.00002 over 3 seeds, bounded to |Δ| < 0.00015 (SO3C_STATUS.md, three arms re-measured). The caveat in §7.1 applies: that test fed ⟨p_a,p_b⟩ through asinh in its linear regime.

**Verdict:** not the deep cause. **Correction to carry into SO3C_STATUS.md:** the sentence "the lift discards each constituent's component along the jet axis" is formally right for a general vector, but for this dataset's massless constituents no information is lost. What remains true is that the lift is a **re-encoding**. The pair masses that carry top-tagging information sit in a subtractive term next to ⟨p_a,P⟩⟨p_b,P⟩. That is a conditioning question (§7.1), not an information question.

---

## 4. Q1: what makes PELICAN's 1k model match our 22.8k

### 4.1 PELICAN as published and as coded

Sources are 2307.16506v4 §3.1–3.3 (PDF pp.9–11) and §4.2 (p.12), checked against the code.

- **Input.** All pairwise Minkowski products d_ij of the constituents, plus two beams (1,0,0,±1). The units are **GeV**: `--scale` defaults to 1.0 (`src/trainer/args.py`), `config/toptag.yaml` does not override it, and the collate function multiplies Pmu by `scale`.
- **Embedding.** f_α(x) = ((1+x)^{α²} − 1)/α² with several trainable α, "initialized to span the interval [0.05, 0.5]". The code uses `linspace(0.05, 0.5)` and the sign-symmetric `slog` mode. The paper's only evidence for it is "providing a slight boost to performance" (§3.1). No number is given.
- **Pair-state initialisation.** [f_α(d_ij) channels] ⊕ Eq1→2(binary beam/constituent label), giving a [B, N, N, C⁰] tensor (§3.1).
- **Each Eq2→2 block** (§3.2):
  - A message MLP on the channels: dense, LeakyReLU, **masked BatchNorm2D** over the real (B, N, N) entries per channel.
  - Dropout.
  - **15 linear aggregators** (Fig. 1): 5 of order zero (identity, transpose and 3 diagonal embeddings), 8 of order one (row, column and diagonal sums, re-embedded) and 2 of order two.
  - Each aggregator is a **mean multiplied by N^α/N̄^α**, α trainable per channel and aggregator, "initialized as a random float in [0, 1]". N̄ = `nobj_avg` = 49 in `config/toptag.yaml`.
  - A dense layer mixes the C × 15 outputs down. It is factorised (Eq. 3.1), and "at very low network widths ... the unfactorized network performs better and may even have fewer parameters".
- **Output.** Eq2→0 with 2 aggregators, the trace and the total sum, then a one-layer output MLP (§3.3, §4.2).
- **Training** (§4.2):
  - Data: constituents capped at 80 ("no noticeable performance gain was seen beyond that number"); batch 100, each minibatch balanced between signal and background. The repo config says batch 64.
  - Regularisation: dropout 0.025; AdamW with weight decay 0.005.
  - Schedule: 35 epochs. Warm-up for 4 epochs to 1e-3, then CosineAnnealingLR with T₀ = 4 and T_mult = 2 for 28 epochs, then decay by γ = 0.5 for 3 epochs.
- **Repo default** (not the paper's setting): `method: spurions`, `stabilizer: so2`. That adds (1,0,0,0) and (0,0,0,1), which span the same plane as the paper's beams.

### 4.2 PELICAN's evidence: the size sweep, nothing else

PELICAN publishes **no ablation of f_α, N^α, the masked BatchNorm or the aggregator subsets.** What it does publish:

**Table 2** (PDF p.13), depth L and width (messaging in / aggregation in):

| L | width | params | accuracy | AUC | 1/ε_B @ 0.3 |
|---|---|---|---|---|---|
| 5 | 132/78 | 208k | 0.9425(1) | 0.9870(1) | 2250 ± 75 |
| 5 | 60/35 | 48k | 0.9424(1) | 0.9868(1) | 2148 ± 125 |
| 5 | 25/15 | 11k | 0.9410(3) | 0.9858(4) | 1879 ± 103 |
| 5 | 10/6 | 3k | 0.9386(2) | 0.9850(1) | 1494 ± 43 |
| 3 | 6/4 | 1k | 0.9358(7) | 0.9835(2) | 1145 ± 74 |
| 2 | 6/3 | 605 | 0.9336(5) | 0.9823(3) | 901 ± 59 |
| 1 | 6/3 | 326 | 0.9291(5) | 0.9801(4) | 669 ± 41 |
| 1 | –/6 | 248 | 0.9258(8) | 0.9780(6) | 516 ± 52 |

The last row has "the fully connected messaging layers disabled altogether". With 248 parameters it reaches 0.9780 / 516. That is the input pipeline plus aggregation alone, and it beats our 9k `so3c_covariant_set` at K=32 (0.9746 / 312).

At the tiny widths, each added block pays:

| step | AUC gain | rejection |
|---|---|---|
| L = 1 → 2 | +0.0022 | 669 → 901 |
| L = 2 → 3 (width 6/3 → 6/4) | +0.0012 | 901 → 1145 |

Width at L = 5 pays less as it grows:

| step | AUC gain |
|---|---|
| 3k → 11k | +0.0008 |
| 11k → 48k | +0.0010 |
| 48k → 208k | +0.0002 |

Our measured width slope is about +0.0002 per doubling (SO3C_STATUS.md, long-schedule window), close to PELICAN's 48k → 208k end, not its 1k → 48k end.

**Table 3** (PDF p.15) gives the data fractions, at 70 epochs for the fractional runs. At 5% of the data the widths still rank: 1270 ± 65, 1148 ± 49 and 1111 ± 108. At 1% they do not separate: 789, 799 and 798. The 100% rows of Table 3 do not match Table 2 for two widths: 60/35 is 2133 ± 148 against 2148 ± 125, and 25/15 is 0.9863 / 1885 ± 109 against 0.9858 / 1879 ± 103. When quoting PELICAN, use Table 2.

**Table 1** (PDF p.13):

| model | accuracy | AUC | 1/ε_B @ 0.3 | params |
|---|---|---|---|---|
| PELICAN | 0.9426(2) | 0.9870(1) | 2250 ± 75 | 208k |
| PELICAN-IRC | 0.9406(2) | 0.9844(11) | 1711 ± 208 | 208k |

PELICAN-IRC uses energy-weighted aggregators and inputs of 1 − cosθ.

### 4.3 The answer

PELICAN's 1k model matches our 22.8k model for three reasons, ordered by how directly the evidence supports each:

1. **Budget.** PELICAN spends its whole budget on equivariant pair processing that shares weights across all N² pairs, through parameter-free aggregators. We spend 25% there (§2.4).
2. **Rank.** PELICAN's pair state is *transformed* pairwise in every block and mixed through 15 aggregation patterns. Each block measurably pays at tiny width (Table 2). Our model reduces pairs to nodes in every round. Its distinctive covariant flow is the identity in round 0 and active in only 2–4 channels afterwards (§2.3), so the later rounds mostly re-read rescaled copies of the same pair invariants. The pair-latent switch is the direct transplant of this difference (§7.2).
3. **Conditioning.** Every PELICAN block normalises (masked BatchNorm), and its inputs are log-compressed in GeV. Our inputs sit in asinh's linear regime (§2.1), and the network has no normalisation layer at all. Its layers are Linear+Tanh or Linear+ReLU only (`benchmarks/so3c_models.py`, `SO3CMessageSetClassifier.__init__`). This one is untested (§7.1).

---

## 5. Q3: LorentzNet (2201.08187v6)

### 5.1 Mechanism

From §3.1, Eq. 3.2–3.4 (PDF pp.7–8), checked against the release code:

- **Edge message:** m_ij = φ_e(h_i, h_j, ψ(‖x_i − x_j‖²), ψ(⟨x_i, x_j⟩)), with ψ(·) = sgn(·)·log(|·| + 1). The inputs are in **GeV**: `top/dataset.py` calls `collate_fn(..., scale=1, add_beams=True, beam_mass=1)`.
- **"Minkowski dot-product attention"**, the coordinate update: x_i ← x_i + c·Σ_j φ_x(m_ij)·x_j, with **c = 5e-3** for top tagging (§3.3). The code averages over neighbours (`unsorted_segment_mean`) and then multiplies by c.
- **Node update:** h_i ← h_i + φ_h(h_i, Σ_j w_ij m_ij), with the edge significance w_ij = sigmoid(φ_m(m_ij)) ∈ [0, 1]. This is not a softmax.
- **Network** (§3.3, p.9): L = 6 blocks of width 72. φ_x, φ_e and φ_h are each Linear → ReLU → **BatchNorm1d(72)** → Linear. The output is average pooling of h, then dropout 0.2, then 2 linear layers.
- **Training:** AdamW with weight decay 0.01, 35 epochs on the same schedule as PELICAN, batch 32 per GPU on 4 GPUs. The final model is the checkpoint with the highest validation accuracy.
- **Graph:** the code removes the self-edge (`~torch.eye`).

### 5.2 Numbers

| | accuracy | AUC | 1/ε_B @ 0.5 | 1/ε_B @ 0.3 | source |
|---|---|---|---|---|---|
| LorentzNet (6 runs) | 0.942 | 0.9868 | 498 ± 18 | 2195 ± 173 | Table 1, p.11 |
| LorentzNet without equivariance | 0.934 | 0.9832 | 290 ± 30 | 1105 ± 59 | **Table 4, p.14** (the only ablation) |
| EGNN, Euclidean, 222k params | 0.922 | 0.9760 | 148 ± 8 | 540 ± 49 | Table 1 |

LorentzNet has 224k parameters (Table 5, p.14). PELICAN's Table 1 lists it at 220k.

### 5.3 Fit to us

- The coordinate update is what our `--vector-channel` implements. It measured null (§3.2).
- The ‖x_i − x_j‖² edge is `--relnorm-edge`. It was screened on one seed at −0.00044, inside the ±0.00024 single-run spread (STATUS gap screen), so it is unresolved, not closed.
- The parts never tried are **ψ applied in GeV** and **BatchNorm inside every φ**, both of which go into idea 1. Our dropout 0.2 + weight decay 0.01 bundle (STATUS, closed) tested the training recipe, not the architecture.

---

## 6. Q4 and Q5: ParT, L-GATr, L-GATr-slim and LLoCa

### 6.1 ParT (2202.03772v3)

- **Mechanism** (§4, Eq. 3–4, PDF pp.5–6):
  - Four pair features per pair: ln Δ, ln k_T, ln z, ln m². Only m² is Lorentz-invariant.
  - They are embedded by a pointwise convolution with channels (64, 64, 64, 8), with BN in between, into an interaction matrix U with one channel per head.
  - U is added to the pre-softmax attention logits: softmax(QKᵀ/√d_k + U)V. The same U is used in all 8 particle-attention blocks.
- **Ablation** (Table 1, JetClass, p.7): ParT 0.861 / 0.9877 against ParT (plain), with no pair input, 0.849 / 0.9859. For t → bqq′, Rej50% is **32787 against 17699**. The pair bias costs 30% more FLOPs (Table 4).
- **Top tagging** (Table 5, p.9):

  | model | accuracy | AUC | 1/ε_B @ 0.3 |
  |---|---|---|---|
  | ParT from scratch (20 epochs) | 0.940 | 0.9858 | 1602 ± 81 |
  | ParticleNet | 0.940 | 0.9858 | 1615 ± 93 |
  | ParT-f.t. (JetClass-pretrained) | 0.944 | 0.9877 | 2766 ± 130 |

  From scratch, ParT only matches ParticleNet.
- **Fit to us:** we already have pair features on every edge; our model is all pairs. The P-MHA ablation removes pair information altogether, so it says nothing about our design. What carries over is the **logarithm on the pair features**, which goes into idea 1. Softmax attention is not needed: PELICAN and LorentzNet reach 0.9868–0.9870 with mean and sigmoid aggregation.

### 6.2 What does not scale: parameters, from scratch

| model | params | AUC | 1/ε_B @ 0.3 | source |
|---|---|---|---|---|
| PELICAN 60/35 | 48k | 0.9868(1) | 2148 ± 125 | 2307.16506 Table 2 |
| PELICAN 132/78 | 208k | 0.9870(1) | 2250 ± 75 | 2307.16506 Table 2 |
| LorentzNet | 224k | 0.9868 | 2195 ± 173 | 2201.08187 Tables 1, 5 |
| CGENN | 0.3M | 0.9869 | 2172 | 2411.00446 Table 2; 2512.17011 Table 1 |
| L-GATr | 1.1M | 0.9870 ± 0.0001 | 2240 ± 70 | 2411.00446 Table 2, p.12 |
| L-GATr-slim | 2.0M | 0.9869 ± 0.0001 | 2264 ± 93 | 2512.17011 Table 1, p.5 |
| LLoCa-Transformer | 2.0M | 0.9866 ± 0.0001 | 2150 ± 130 | 2512.17011 Table 1, which attributes it to the LLoCa paper; the LLoCa paper itself (2505.20280v2) reports JetClass only |
| PET v2-s | 3.0M | 0.987 | 2167 ± 153 | 2512.17011 Table 1 |
| IAFormer | 0.2M | 0.987 | 2012 ± 30 | 2512.17011 Table 1 |

Above about 48k parameters, nothing trained from scratch beats 0.9870.

### 6.3 What does scale

**Symmetry breaking.** L-GATr Table 3 (2411.00446, PDF p.13, 3 seeds):

| reference inputs | AUC | 1/ε_B @ 0.3 |
|---|---|---|
| none | 0.9846 ± 0.0002 | 1494 ± 62 |
| beam as spatial vectors x^V_3 = ±1 only | 0.9854 ± 0.0005 | 1684 ± 221 |
| time x^V_0 = 1 only | 0.9865 ± 0.0001 | 2179 ± 123 |
| beam bivector x^B_12 + time, as channels | 0.9866 ± 0.0001 | 2150 ± 82 |
| beam (1,0,0,±1) vectors + time | 0.9869 ± 0.0001 | 2282 ± 174 |
| none, plus extra kinematic scalars (pT, E, ΔR) | 0.9869 ± 0.0001 | 2179 ± 156 |
| **beam bivector x^B_12 + time (default)** | **0.9870 ± 0.0001** | **2240 ± 70** |

LLoCa Table 6 (2505.20280, p.23, JetClass) gives the same ordering: no symmetry breaking 0.9870; reference vectors 0.9877; non-invariant scalars 0.9881; both 0.9882. **We already have this lever:** our beams b± = E_b(1, 0, 0, ±1) span the time axis, since b₊ + b₋ = 2E_b·t. The time-axis reference is closed in STATUS for this reason, and L-GATr's (1,0,0,±1) + time row agrees with that. The "extra kinematic scalars" row matches the references without adding any (0.9869 / 2179). Giving the network log-scaled lab kinematics is therefore equivalent to giving it references, not additive (§7.1c).

**The training recipe.** "ParT (longer)", trained with the slim L-GATr recipe: **0.9865 / 1808 ± 33**, against ParT's original **0.9858 / 1602 ± 81** (2512.17011 Table 1). That is +0.0007 AUC and +13% rejection from training alone.

| | optimiser | lr | weight decay | other |
|---|---|---|---|---|
| L-GATr (2411.00446 Table 9, p.28) | Lion | 3e-4 | 0.2 | batch 128, cosine, 2e5 iterations ≈ 21 epochs; inputs scaled by 20 GeV (App. A) |
| L-GATr-slim (2512.17011 App. A, p.17) | Lion | **3e-5** | **2** | the product lr·wd is the same as L-GATr's |
| slim mini-taggers (2512.17011 Table 8) | Adam | 3e-3 at 20k params | 0 | batch 512, 2e5 iterations ≈ 85 epochs of 1.2M jets |

**Depth at small size.** 2512.17011 §2.5 and Fig. 3, p.9:

- If blocks and width are cut together, LLoCa-Transformer leads at small sizes.
- If the network is held at 10 blocks and only width is cut, L-GATr-slim keeps "a background rejection rate above 1000 using only two vector and four scalar channels". That is the "2k (deep)" configuration of Table 7.

This is a text claim backed by a figure. **No table gives the rejection values of Fig. 3.**

**Pretraining**, a different comparison class:

| model | AUC | 1/ε_B @ 0.3 | source |
|---|---|---|---|
| L-GATr-f.t. | 0.98793 ± 0.00001 | 2894 ± 84 | 2411.00446 Table 2 |
| L-GATr-slim-f.t. | 0.9879 | 2927 ± 70 | 2512.17011 Table 1 |
| OmniLearned-L (423M params, 1058M training jets) | 0.9880 | 3486 ± 157 | 2512.17011 Table 1 |

### 6.4 LLoCa specifics, and what they do not show

- **Table 2** (2505.20280, p.9) is **amplitude regression (Z+4g), not tagging**. MSE ×1e-5 by message representation:

  | messages | MSE ×1e-5 |
  |---|---|
  | 16 scalars | 40 ± 4 |
  | a single rank-2 tensor | 2.0 ± 0.4 |
  | 4 vectors | 1.4 ± 0.2 |
  | 8 scalars + 2 vectors | 1.0 ± 0.1 |

  The "2-tensor" is a general 16-component tensor, not a bivector. The lesson is specific to canonicalisation: purely scalar messages between local frames are poor. It does not bear on a bivector lift.
- **Table 4:** Minkowski attention 1.0 ± 0.1 against Euclidean 2.5 ± 0.4, on the same regression.
- **Cost** (§5.3): FLOPs +10–50% and training time +60–100% over the non-equivariant backbone.

LLoCa is a way to make a non-equivariant backbone equivariant. We already are.

---

## 7. Ideas, ranked

**Status after the 2026-10-05 screen (audit, plan_2026-10-10 §1 #8):** 7.1(a) and 7.1(b) are **measured null on AUC** (`gap_generic_ml.md` §9); 7.2, the pair latent, is unresolved at +0.00012 (t = 1.42); 7.3 N^α, 7.4 depth and 7.5 Lion are untested.

**Protocol for every cost below.** The K=32 long-schedule screen: all 1.211M training jets, 60 epochs, compiled, on a T4. A base seed costs about 3.4–3.5 h (STATUS final table). ~~The arms are paired by seed against the existing `results_long_seeds/base` seeds, so no new control is needed.~~ **Corrected (audit, plan_2026-10-10 §1 #1):** those bases ran on torch 2.10 and the card now has 2.11, and pairing across that change is not established. Every arm needs a **same-stack control**, which raises every cost below by the control's cost (K=32 long: about +3.5 h per seed). Within one stack, same seed means the same batch order, and the paired differences there agreed to ±0.00001 over two pairs — fragile, but far tighter than the ±0.00020 unpaired spread.

**Budget.** The current window has used about 13.9 h, plus the two K=64 width kernels pushed at commit 82235c4 (about 7.2 + 4.3 h). That leaves roughly 4–5 h until the reset on Saturday 2026-10-10. This is an estimate; test by pushing.

### 7.1 Input conditioning and normalisation (new; untested; exactly covariant)

**Mechanism.** Every reference tagger conditions its invariant inputs and normalises inside the network:

| tagger | input compression | normalisation |
|---|---|---|
| PELICAN | f_α on GeV | masked BatchNorm2D every block |
| LorentzNet | ψ = sgn·log(1+\|x\|) on GeV | BatchNorm1d in every φ |
| ParT | logarithms of all pair features | BN in the pair embedding; LayerNorm in every block |
| L-GATr | inputs in 20 GeV units | equivariant LayerNorm in every block |
| ours | asinh in 38 GeV units: the linear regime for ~95% of edge invariants and ~99% of pair masses | none |

**Why it fits.** The change touches only invariant features and the hidden activations of the edge MLP, so exact Lorentz covariance and permutation equivariance are untouched. It also offers a mechanism for two things already in STATUS:

- the model's continued improvement from 30 to 60 epochs, with weights growing across the board (§2.2) — **suggestive only** (audit #3): +0.00037 is 2.2 standard errors (t on 2 df, p ≈ 0.15), a falling validation loss is weak evidence because cosine flattens at the end, and the K=32 60-epoch base has its lowest validation loss at epochs 56–60 and does not overfit;
- lr 3e-3 beating 1e-3 — **a single-seed comparison** (audit #4), like the 4 → 8 channel claim; not resolved unless far beyond the ±0.00024 single-run spread.

Neither is proof.

**Forms**, cheapest first:

- **(a) A log knee where the data are.** asinh(e^κ·x) per invariant family (pair s_ab, node s_aa, the h_init inputs), with a learnable κ initialised so the knee sits near the median |x|. For s_ab that means e^κ ≈ 30–100; for a Minkowski pair product, about 1/6.6e-4 ≈ 1500. This replaces the `--falpha 3` screen, which put f_α's knee at x = 1. In our units that is the linear regime, so **that screen did not test PELICAN's embedding as PELICAN uses it**, and its bank cost 3.54× on CPU.
- **(b) Normalisation of the edge pre-activation.** Either a per-pair LayerNorm over the 16 hidden units (no batch dependence) or PELICAN's masked BatchNorm over real pairs. A fixed standardisation of the edge inputs, computed once from training statistics, is the zero-risk minimum. **Since measured:** in rounds 1–2, 61–84% of edge pre-activations have |x| > 2, where tanh′ ≤ 0.07; in round 0, 14–40% ([`gap_generic_ml.md`](gap_generic_ml.md) §3.2). Inputs too small in round 0 and saturated in rounds 1–2 make one problem, and a per-pair LayerNorm addresses both.
- **(c) Log lab kinematics in `h_init`:** log E, log pT, log E/E_J, log pT/pT,J, and ΔR to the jet axis. Each is a function of the invariants ⟨p,b±⟩, ⟨P,b±⟩, ⟨p,P⟩ and ⟨p,p⟩ of the set including the beams, so the model stays exactly covariant with the beams moved along. L-GATr Table 3 says this only matches references we already have (§6.3), so (c) is a conditioning change, not new information.

**Counter-evidence, stated plainly.** The 8 complex channel weights already act as learnable input scales, since s_ab scales as w_c². Training never pushed |w_c|² above about 2.3, and it collapsed three channels instead (§2.3). Either the network does not want larger scales, or the route through w_c is poor, because w_c also sets the flow strength. Only the run decides.

**Expected gain:** 0 to +0.001 AUC. No paper isolates this: PELICAN says only "slight boost" for f_α, and nobody ablates BatchNorm.
**Cost:** (a) and (c) about 1.00×; (b) estimated 1.05–1.10× (unmeasured). Two paired seeds take about **7.5 GPU-h**. Before any GPU run, check on CPU: exact covariance with excited heads in float64, and the step-time ratio.

### 7.2 Pair latent at the long schedule (`--pair-latent 8`, already implemented)

**Mechanism.** This is PELICAN's central difference (§4.3): a rank-2 state E_ab carried through the rounds and updated by permutation-equivariant maps. Ours uses 7 of PELICAN's 15. It is the clean way to give later rounds pair information that is not a rescaling of round 0, which §2.3 shows is currently missing.

**Evidence.**
- PELICAN Table 2 gains +0.0012 to +0.0022 per block at tiny width.
- Our probe protocol gave a paired **+0.00012 (t = 1.42)**, with a spread four times smaller than the base (STATUS). That was at 20 epochs on 400k jets, a regime that cannot show a capacity gain. It has never been run at the long schedule.

**Expected gain:** +0.0002 to +0.0008. A cheap add-on is PELICAN's (N/49)^α on its aggregators (§7.3).
**Cost:** at K=64 it measured 918 s per epoch against 589 for the headline (1.56×; STATUS speed-up table). K=32 compiled is unmeasured, because the profile rows for pair8 ran eager. The estimate is about 5.3–5.5 h per seed, so two paired seeds take about **11 GPU-h**.

### 7.3 N^α aggregation scaling (PELICAN §3.2; N̄ = 49)

**Mechanism.** Multiply each mean aggregation by (N/N̄)^α, with α learned per channel. α = 0 is a mean and α = 1 is a sum (PELICAN §3.2: "α = 1 returns the sum aggregation function"). For us it applies to:
- the scalar-message mean,
- the flow reference ref_a = (1/N)Σ_b w_ab z_b,
- the pair-latent aggregators,
- the readout pools.

Today we assert mean semantics everywhere by hand.

**Fit.** N is a count of real constituents, which is an invariant, so covariance is untouched.

**Expected gain:** ≤ +0.0003. Multiplicity already carries little: stacking it on our scores gave +0.00002 (STATUS Closed). Any gain would come from sum-versus-mean semantics, not from new information.
**Cost:** about zero. Bundle it into 7.1, and ablate only if the bundle wins.

### 7.4 Depth (4–5 rounds), only after 7.1

**Evidence.**
- PELICAN Table 2: L = 1 → 2 → 3 gives 0.9801 → 0.9823 → 0.9835.
- 2512.17011 §2.5, text and figure only: depth held at 10 keeps rejection above 1000 at about 1k parameters.
- Our `rounds 6` probe (−0.00036, one seed, 1.5σ) is inside the noise, and it ran in the probe regime.

**Fit.** The flow is the identity at initialisation (zero `w_head`) and h is residual, so depth is cheap to add structurally. Without normalisation, deeper invariant stacks are exactly where the conditioning problem compounds. Hence the ordering after 7.1.

**Expected gain:** 0 to +0.001.
**Cost:** the edge work grows linearly with the number of rounds. Five rounds is estimated at about 1.5–1.7× (unmeasured), about 5.5 h per seed.

### 7.5 Lion, as a check on whether training is the limit

**Evidence.** The slim recipe moved ParT from 0.9858 / 1602 to 0.9865 / 1808 (§6.3). Every L-GATr-family tagger here uses Lion. Our decoupled weight decay at 0.002 is **not shown to help** (audit #2): it measured −0.00011, but the decay runs were on torch 2.11 and their controls on 2.10, so the pairing is unconfirmed, and unpaired −0.00011 is inside the ±0.00027 spread. Lion's sign update is a different optimiser, not that knob.

**Expected gain:** 0 to +0.0005. Its main value is diagnostic: it separates a training limit from an architecture limit.
**Cost:** 1.0× per run, but the Lion learning rate needs at least two values (Adam's 3e-3 divided by 3–10), so about 7 GPU-h for a fair pair.

### 7.6 Not recommended

| idea | why not |
|---|---|
| Rebuild on vectors; drop the bivector lift | §3: the lift is information-complete for massless inputs, the vector channel measured null, and the 2606.21790 result does not apply to input lifts |
| Softmax attention over w_b | PELICAN and LorentzNet reach 0.9870 with mean and sigmoid aggregation; ParT's ablation is about having pair features, which we have |
| LLoCa | It makes non-equivariant backbones equivariant, at +60–100% training time; it tops out at 0.9866 / 2150 on top tagging |
| K > 64 | PELICAN capped at 80 with "no noticeable performance gain"; our energy measurement already closed it (STATUS) |
| The `--falpha` bank as implemented | Knee in the linear regime (§7.1a); 3.54× CPU |
| JetClass pretraining | The only route above 0.9870 (L-GATr: +0.0009 AUC, +654 rejection), but a different comparison class, at ≥ 25 GPU-h |
| Weight decay, dropout | Measured negative (STATUS) |
| PELICAN's trace/total readout stream | Already present: our pooled invariants keep the diagonal (s_aa) and off-diagonal (s_ab) means apart (theory.md §13) |

### 7.7 Expectation and a proposal for the window opening 2026-10-10 (30 GPU-h)

| arm | seeds | GPU-h |
|---|---|---|
| 7.1 bundle: knee (a) + edge LayerNorm (b) + N^α (7.3) | 0, 1 | ~7.5 |
| 7.2 pair latent 8 | 0, 1 | ~11 |
| 7.5 Lion at one learning rate | 0 | ~3.5 |
| reserve: a second Lion lr, or 7.1 + 7.2 stacked, on seed 0 | | ~6–8 |

All arms run on the K=32 long schedule, paired against the existing base seeds. The K=64 width runs now on Kaggle decide whether winners move to K=64, where each run costs 2–3× as much.

**Stated plainly.** If 7.1–7.3 all land near the middle of their ranges and stack, the gain is +0.0005 to +0.0015 AUC, taking the single model to about 0.9841–0.9851. That is not 0.9870. Closing the full 0.0030 would need every idea at the top of its range plus width transferring to K=64, and nothing in the literature predicts that for an architecture like ours. If 7.1 and 7.2 both come back null at the long schedule, the honest conclusion is that this construction saturates near PELICAN's 1–3k operating point (0.9835–0.9850), and the paper should be framed around that.

---

## 8. Corrections to earlier notes

| where | claim | correction |
|---|---|---|
| `gap_to_sota_2026-09-13.md` #19 | slim L-GATr "20k ≈ 1000–1100, 200k ≈ 1400–1500 rejection" | Read off 2512.17011 Fig. 3. No table gives these values; the only verifiable statement is the §2.5 text (> 1000 at ~1k parameters, 10 blocks) |
| same, #21 | LLoCa Table 2 shows "vectors beat rank-2" for our purpose | That table is amplitude regression, and its "single 2-tensor" is a general rank-2 tensor, not a bivector |
| same, #21 and STATUS | 2606.21790 implies bivector lifts are weak | It zeroes internal G2 channels of an L-GATr with vector inputs; G2 invariants alone still probe to AUC 0.966 (§3) |
| same, #24 | "2512.17011 confirms the same [Lion 3e-4, wd 0.2]" | Slim uses Lion lr 3e-5, wd 2 (App. A); 3e-4 / 0.2 is L-GATr's (2411.00446 Table 9) |
| STATUS, Open: vector channel | "The bivector lift … discards each constituent's component along the jet axis" | True for a general vector; for massless constituents the dropped component is fixed by p² = 0, so nothing is lost (§3.2) |
| STATUS gap screen, f_α and vector rows | read as tests of PELICAN's and LorentzNet's input handling | Both fed tiny invariants into compressors in their linear regime (§2.1), so neither tested the feature as the reference model uses it |
| PELICAN itself | Table 2 vs Table 3, 100% rows | 60/35: 2148 ± 125 vs 2133 ± 148; 25/15: 0.9858 / 1879 ± 103 vs 0.9863 / 1885 ± 109. Quote Table 2 |
| PELICAN repo vs paper | batch, symmetry breaking | Repo `toptag.yaml` uses batch 64 and `spurions so2`; the paper uses batch 100 and beams (1,0,0,±1) |
| `gap_to_sota` #26 | cites arXiv:2608.02735 | Not verified in this pass |

STATUS itself was not edited; these are recommendations.

---

## 9. Generic ML and optimisation levers (follow-up, 2026-10-05)

The details are in [`gap_generic_ml.md`](gap_generic_ml.md). Every diagnostic there ran read-only on CPU against the two headline checkpoints. In brief:

- **The flow-head L2 penalty does not bind.**
  - In live channels the data gradient is 47–405× the penalty gradient, and the penalty is ≤ 2.5% of Adam's normaliser.
  - The Hessian curvature implies it shrinks the heads by ≲ 9%.
  - A paired fine-tune without it changes nothing, and nothing breaks.
  - No coefficient-0 arm is proposed.
- **Tanh saturation in the edge layer does bind:** 61–84% in rounds 1–2. It strengthens idea 7.1(b).
- **Clipping, TTA and fp16 are not levers.**
  - Clipping binds on 29% of early steps and ≤ 12% later.
  - TTA over the detector reflections gives +0.00002 ± 0.00002, against +0.00034 for the 4-seed ensemble.
  - fp16 on the edge Linear: forward ΔAUC −3e-8, with ≤ 10% speed (prior).
- **Ranked generic candidates:**
  1. Edge pre-activation normalisation plus readout standardisation (about 7.5 GPU-h).
  2. A longer or warmup–stable–decay schedule.
  3. Distilling the ensemble.

  EMA and dropping the penalty ride along as passengers.
- **Measured on GPU, 2026-10-05** (`gap_generic_ml.md` §9): the probe protocol, three seeds paired with a fresh torch-2.11 control.
  - **7.1(b), the edge LayerNorm:** ΔAUC −0.00004 ± 0.00010. It halves the saturation it targets, and tail rejection drops about 5%.
  - **7.1(a), the log-knee:** ΔAUC +0.00000 ± 0.00008. Rejection rises +39 ± 30, suggestive only.
  - **Conditioning, the top-ranked cheap lead, does not move AUC.** The remaining candidates are structural: the pair latent at the long schedule, depth, a longer schedule, distillation.

---

## Appendix A. Number-to-source log

Page numbers are physical PDF pages.

| number(s) | paper | location |
|---|---|---|
| PELICAN 0.9870(1) / 2250 ± 75 / 208k; PELICAN-IRC 0.9844(11) / 1711 ± 208; LorentzNet 220k | 2307.16506v4 | Table 1, p.13 |
| PELICAN size sweep (8 rows) | 2307.16506v4 | Table 2, p.13 |
| PELICAN data fractions; inconsistencies | 2307.16506v4 | Table 3, p.15 |
| f_α formula, [0.05, 0.5], "slight boost"; beams (1,0,0,±1); Eq1→2 label | 2307.16506v4 | §3.1, p.9 |
| masked BatchNorm2D; 15 aggregators; N^α/N̄^α, α ∈ [0,1]; factorisation and its low-width caveat | 2307.16506v4 | §3.2, pp.10–11; §2.4, p.8 |
| Eq2→0 trace + total sum | 2307.16506v4 | §3.3, p.11 |
| 5 blocks; cap at 80; dropout 0.025; AdamW 0.005; 35 epochs; batch 100; one-layer output MLP | 2307.16506v4 | §4.2, p.12 |
| N̄ = 49, scale 1.0, batch 64, spurions so2, α linspace | PELICAN repo | `config/toptag.yaml`, `src/trainer/args.py`, `src/layers/generic_layers.py` |
| LorentzNet 0.942 / 0.9868 / 498 ± 18 / 2195 ± 173; EGNN 540 ± 49 | 2201.08187v6 | Table 1, p.11 |
| LorentzNet without equivariance 0.9832 / 1105 ± 59 | 2201.08187v6 | Table 4, p.14 |
| LorentzNet 224k | 2201.08187v6 | Table 5, p.14 |
| L = 6, width 72, BatchNorm1d, c = 5e-3, AdamW 0.01, best-val checkpoint | 2201.08187v6 | §3.3, p.9 |
| ψ = sgn·log(1+\|x\|), scale 1, mean coordinate update, self-edge masked | LorentzNet-release | `models.py`, `top/collate.py`, `top/dataset.py` |
| ParT pair features, U as attention bias, (64, 64, 64, 8) embedding | 2202.03772v3 | §4, Eq. 3–4, pp.5–6 |
| ParT vs ParT (plain), JetClass | 2202.03772v3 | Table 1, p.7 |
| ParT top tagging 1602 ± 81; ParT-f.t. 2766 ± 130 | 2202.03772v3 | Table 5, p.9 |
| L-GATr 0.9870 ± 0.0001 / 2240 ± 70; L-GATr-f.t. 0.98793 / 2894 ± 84; CGENN 0.9869 / 2172 | 2411.00446v3 | Table 2, p.12 |
| symmetry-breaking rows | 2411.00446v3 | Table 3, p.13 |
| Lion 3e-4, wd 0.2, batch 128, 2e5 iterations; 20 GeV inputs | 2411.00446v3 | Table 9, p.28; App. A, p.27 |
| L-GATr-slim 0.9869 ± 0.0001 / 2264 ± 93; ParT (longer) 0.9865 / 1808 ± 33; LLoCa-T 0.9866 / 2150 ± 130; slim-f.t. 2927 ± 70; OmniLearned-L 3486 ± 157 | 2512.17011v2 | Table 1, p.5 |
| ultra-mini taggers, text claim > 1000 at 10 blocks | 2512.17011v2 | §2.5, pp.8–9; Fig. 3 |
| configurations; training hyperparameters of the mini-taggers | 2512.17011v2 | Tables 7–8, p.18 |
| slim Lion 3e-5, wd 2 | 2512.17011v2 | App. A, p.17 |
| LLoCa message representations (regression) | 2505.20280v2 | Table 2, p.9 |
| Minkowski vs Euclidean attention | 2505.20280v2 | Table 4, p.22 |
| LLoCa symmetry breaking (JetClass) | 2505.20280v2 | Table 6, p.23 |
| +10–50% FLOPs, +60–100% training time | 2505.20280v2 | §5.3, p.10 |
| grade-ablation numbers | 2606.21790v1 | Tables 1, 2, 4, 6, 7; §6.2–6.3; pp.3–19 |
| vector channel −0.00002, pair latent +0.00012 (t = 1.42), width +0.00048, weight decay −0.00011 (not shown to help: pairing across torch 2.10 → 2.11 unconfirmed) | ours | SO3C_STATUS.md |
| §2 numbers | ours | measured 2026-10-04, appendix B |

## Appendix B. Read-only diagnostic snippets

The snippets were run from the worktree root with the main checkout's `data/toptagging/*.parquet`. They write nothing.

```python
# B1. Input scale and pair-product ranges (train parquet for the scale, test for pairs)
import numpy as np, pyarrow.parquet as pq
K = 64
def load(split, n):
    t = pq.ParquetFile(f'../../../data/toptagging/{split}.parquet').read_row_group(0).to_pandas().iloc[:n]
    c = lambda s: [f'{s}_{i}' for i in range(K)]
    p = np.stack([t[c('E')].values, t[c('PX')].values, t[c('PY')].values, t[c('PZ')].values], -1).astype(np.float64)
    return p, t['is_signal_new'].values
ptr, _ = load('train', 30000); m = ptr[..., 0] > 0
scale = np.sqrt((ptr[m] ** 2).mean())           # 38.0 GeV
# Pairs: G_ab = E_a E_b - p_a . p_b over real constituents; raw lift Re z_a.z_b = <a,P><b,P> - G_ab M^2,
# soft bound 1 + |z.z|^(1/2) = 1 + <a,P>; 1 - cos(theta*) = G_ab M^2 / (<a,P><b,P>).
```

```python
# B2. Flow step sizes on real jets with the trained headline model
import sys, torch; sys.path.insert(0, '.')
import benchmarks.so3c_models as M
model = M.SO3CMessageSetClassifier(channels=8, beams=True)
model.load_state_dict(torch.load('results_k64/headline60/checkpoints/top_tagging_canonical__so3c_message_set__seed0.pt',
                                 map_location='cpu', weights_only=False)['model']); model.eval()
orig, rec = M.expm_so3c_apply, []
def wrapped(a, z, t):
    out = orig(a, z, t)
    rec.append(((out - z).abs().pow(2).sum(-1).sqrt() / z.abs().pow(2).sum(-1).sqrt().clamp_min(1e-30)))
    return out
M.expm_so3c_apply = wrapped
# x: (B, 64, 5) test jets divided by the scale, mask in column 4; then: with torch.no_grad(): model(x)
# rec[r][:, c, :64] over real constituents gives the per-round, per-channel |dz|/|z|.
```

```python
# B3. Weight statistics: edge.{r}.0.weight columns [0:16] pair (Re s, Im s), [16:48] node (s_aa, s_bb),
# [48:64] h_a, h_b; channel_weights norm per channel; w_head.{r}.weight RMS.
```
