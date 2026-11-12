# Related-work notes for the SO3C preprint

Rewritten 2026-11-12 to match the preprint (`SO3C_preprint/so3c_preprint.tex`, 12 pages).
The earlier version described an SO(3,3) design with approximate equivariance and an
adaptive-solver flow. Neither describes SO3C: the construction is exactly covariant, the
flow is a closed-form complex Rodrigues step, and no ODE solver is used. SO(3,3) is a
separate paper (see `REMOVED.md`), so it is not positioned against here.

Numbers below are the preprint's Table 3 (canonical split, test set, 1/ε_B at ε_S = 0.3).
Published values are as reported in the cited tables; SO3C values are our runs.

## Our construction, in one paragraph (for the positioning)

Each constituent is lifted to a complex 3-vector through the bivector it forms with the jet
momentum, z_a = bivec(p_a, P). A message-passing round builds pair invariants z_a·z_b, a
covariant connection a_a = z_a × r_a from invariant-weighted neighbour states, and moves each
state by the closed-form exponential exp(−T[a]×). The output is covariant with the jet and
the beams transformed together, and invariant under rotations about the beam axis with the
beams fixed. The equivariance is exact in exact arithmetic: it holds to 1e-14 relative in
float64 (preprint §2.1, §4.4). In float32 it does not hold on boosted jets (AUC drops 0.0232
against 0.0001 in float64), so evaluation on boosted jets needs float64.

## Lorentz-equivariant taggers (the direct competitors)

### LGN, Bogatskiy et al. 2020 (ICML 2020, arXiv:2006.04780)
- First Lorentz-group-equivariant network for particle physics, built on the group's
  finite-dimensional tensor representations.
- The group action is fixed analytically; there is no learned connection.
- **Positioning:** a predecessor on the same symmetry. Not in the comparison table; cited
  in the introduction.

### LorentzNet, Gong et al. 2022 (JHEP 07 (2022) 030, arXiv:2201.08187)
- Message passing on Minkowski four-momenta with invariant edge features.
- Table 3 value: 0.9868, 224k parameters, 1/ε_B = 2195 ± 173.
- **Positioning:** a reference point we do not beat. SO3C's final model (0.98504 / 1286) is
  below it in rejection and in AUC. Its published numbers come from Table 1 of the paper.

### PELICAN, Bogatskiy et al. 2024 (JHEP 03 (2024) 113, arXiv:2307.16506)
- Permutation-equivariant aggregator with a rank-2 pair state carried through every block.
- Table 3 values: 0.9870 / 2250 ± 75 at 208k; 0.9858 / 1879 ± 103 at 11k; 0.9850 / 1494 ± 43
  at 3k; 0.9835 / 1145 ± 74 at 1k.
- **Positioning:** the strongest reference for the parameter comparison. Our final
  configuration is level with PELICAN's 3k model in AUC and below it in rejection. The gap
  to PELICAN is argued to be architectural, not a matter of size. The pair latent in SO3C
  takes 7 of PELICAN's 15 rank-2 maps.
- **Note:** an earlier version of these notes cited PELICAN as 2022. The JHEP version is 2024.

### L-GATr, Brehmer et al. 2025 (SciPost Phys. 19 (2025) 108, arXiv:2411.00446) and Spinner et al. 2024 (NeurIPS 2024, arXiv:2405.14806)
- Geometric-algebra transformer on multivectors; the beam enters as a bivector or as vectors.
- Table 2 value (Brehmer): 0.9870 / 2240 ± 70 at 1.1M.
- **Positioning:** the bivector-with-beam default is the closest analogue to our beams.
  Their reference-vector ablation (Table 3 of the paper, 3 seeds, read in
  `so3c_notes/gap_thesis.md` §6.3) shows that a time-direction reference alone gives most of
  the gain, 0.9846 → 0.9865; the full default reaches 0.9870.

### L-GATr-slim and ParT (longer), Petitjean et al. 2025 (arXiv:2512.17011)
- L-GATr-slim: 0.9869 ± 0.0001 / 2264 ± 93 at 2.0M, trained with Lion (lr 3·10⁻⁵, weight
  decay 2, 200k iterations, cosine schedule).
- ParT (longer): the original ParT with the slim recipe, 0.9865 / 1808 ± 33 at 2.1M, against
  0.9858 / 1602 ± 81 with the original recipe. Training choices are a comparable lever to
  architecture in this table.
- **Positioning:** the source of the two newest equivariant rows. We did not test Lion on
  SO3C; listed as an open item in the preprint's discussion.

### LLoCa, Spinner et al. 2025 (arXiv:2505.20280)
- Local canonicalisation: each four-vector gets a local reference frame, and latent features
  are invariant in those frames. Any network can be made Lorentz-equivariant this way.
- The LLoCa paper reports JetClass only. Its top-tagging numbers come from Petitjean et al.
  Table 1: LLoCa-Transformer, 0.9866 ± 0.0001 / 2150 ± 130 at 2.0M, trained there with a
  reduced Frames-Net (32 hidden channels).
- **Positioning:** the other 2.0M-parameter equivariant competitor. Do not cite the LLoCa
  paper for the 0.9866 number.

### Agarwal, Khare and Kumar 2026 (arXiv:2606.21790)
- Probes and grade ablations on L-GATr, L-GATr-slim and LLoCa-T. The bivector channels inside
  L-GATr, with vector inputs, are negligible for top tagging; vector-like channels dominate.
- **Positioning:** a result about hidden bivector channels fed by vector inputs. It does
  **not** test an input bivector lift, so it does not support or undermine our lift. The
  preprint says exactly this in §2.2. Do not describe it as evidence for the lift.

### ParticleNet, Qu and Gouskos 2020 (PRD 101, 056019, arXiv:1902.08570); Particle Transformer, Qu, Li and Qian 2022 (PMLR 162, arXiv:2202.03772)
- Non-equivariant taggers. ParT's ln Δ, ln k_T, ln z and ln m² pair features are the source of
  our pair features.
- **Positioning:** a one-sentence mention in the comparison is enough, except for the pair
  features, which are credited to ParT in the preprint's §2.5.

## Broader equivariance background

### Cohen & Welling 2016 (ICML 2016, arXiv:1602.07576)
- Group-equivariant convolutional networks: building the symmetry into the architecture
  rather than learning it from augmentation.
- **Positioning:** one sentence in the introduction, framing equivariance against invariance
  and augmentation.

### EGNN, Satorras, Hoogeboom and Welling 2021 (ICML 2021, arXiv:2102.09844)
- E(n)-equivariant graph networks: invariant distance features, equivariant coordinate
  updates.
- **Positioning:** our pair invariants z_a·z_b follow the same invariant-feature pattern. The
  update is different: a closed-form complex rotation rather than an MLP on invariants.

### EMLP, Finzi, Welling and Wilson 2021 (ICML 2021, arXiv:2104.09459)
- A practical method for equivariant MLPs for arbitrary matrix groups, from the group's
  Lie-algebra generators.
- **Positioning:** the closest precedent for building the symmetry in from the algebra. It
  builds equivariant linear maps; the SO3C activation is non-linear and built from the same
  algebra (so(3,C) ≅ so(3,1)). Cite it honestly.
- **Note:** an earlier version of these notes listed the authors as Finzi, Welling, Wilson,
  which is correct; this version keeps them. The earlier claim that EMLP covers O(p,q) was
  not checked against the paper and has been removed.

## ODE methods

### Neural ODE, Chen, Rubanova, Bettencourt and Duvenaud 2018 (NeurIPS 2018, arXiv:1806.07366)
- A network layer as a continuous flow integrated by a numerical solver.
- **Positioning:** we use the flow picture but integrate in closed form. The connection is
  frozen within each round, so each state moves by one exact group element and no solver is
  needed. A variant that uses an adaptive solver (torchdiffeq, dopri5) reached a lower AUC at
  about nine times the training time (preprint Table 1 and §2.3).

## Higher body order

### MACE, Batatia et al. 2022 (arXiv:2206.07697)
- Higher-body-order equivariant messages for molecular force fields.
- **Positioning:** a natural extension of our two-body messages. Cited in the preprint's
  Discussion as untested. A jet-level test of 3- and 4-body shape invariants of the lift found
  no AUC gain (internal, `so3c_notes/gap_generic_ml.md` §8.4); learned higher-order tensors
  inside the network were not tested.

## Things we do not need to cite extensively

- Other jet taggers (IAFormer, PET, OmniLearned, EFN and the rest): a one-sentence mention in
  the comparison is enough; they are not equivariant.
- Generic group-theory references: keep to one canonical textbook.

## Positioning sentence for the preprint (abstract or conclusion)

> "Prior Lorentz-equivariant taggers build the symmetry into linear layers or message passing
> on four-vectors or fixed tensor representations. SO3C instead lifts each constituent to a
> complex 3-vector through its bivector with the jet momentum and applies a closed-form complex
> rotation whose connection is built from invariants, so the update is exactly covariant with
> no ODE solver. On the canonical top-tagging benchmark it reaches AUC 0.98504 with 67,874
> parameters, below PELICAN, L-GATr-slim and LLoCa-Transformer."

The abstract currently says only that SO3C is below PELICAN, which is still true. Add the
other two comparisons only if the abstract is reopened together with `CITATION.cff`.
