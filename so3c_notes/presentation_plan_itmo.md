# Talk plan: V International Conference «Young Professionals» (ITMO)

Section: AI in applied data. Level: international. Slot: 5 minutes talk + 2
minutes questions. Language: English.

Numbers below come from [`presentation_brief.md`](presentation_brief.md),
[`SO3C_STATUS.md`](../SO3C_STATUS.md), `docs/so3c/experiments.md` and
`figures/*.csv`, all checked against the result files; the equations come
from `docs/so3c/theory.md` (§1, §3, §4, §9, §10). If anything disagrees,
`SO3C_STATUS.md` and `theory.md` win.

## The one message

A neural network built directly from the mathematics of special relativity:
particles become complex 3-vectors, and an attention-like update moves each
one by an exact Lorentz rotation instead of adding to it. Small enough to
train on free GPUs, it took its largest step since the beams by adding
physics knowledge in a symmetry-preserving form: 192 parameters gave as much
as a network with five times the parameters.

Audience: applied machine-learning researchers, mostly not particle
physicists. Lead with the data problem; show the mechanism with a few
equations, each read out in one plain sentence; then the results.

## Slide plan (7 slides, 5:00, plus one backup slide)

The "Say:" text of slides 1–7 is 633 words, 4.5 minutes at 140 words a
minute (4.9 at 130); slide 4 has the least slack. Visible slide text stays short; detail goes into the speaker
notes.

How to show the math: at most four short equation lines per slide, typeset
properly and large; each line gets a 3–6 word caption; colour the complex
covariant vectors (z, r, a) in the accent colour, the momenta and energies
(p, P, E) in normal text colour, and the invariant scalars (s, w, h) in dark
grey. Never derive on a slide; say what each line means.

### 1. Title (0:00–0:10)

- Headline: **SO3C: a Lorentz-covariant neural network for particle jets, built on complexified rotations**
- Aleksandr Panchenko, [affiliation]; conference name and date; QR code to the DOI.

Say: "Good afternoon. I will show a small neural network whose structure
follows special relativity, and the one step that made it much better."

### 2. Jet tagging is set classification on four-vectors (0:10–0:45)

On the slide: a top jet (three sub-jets) beside a QCD jet (one); "a jet = a
set of up to 200 particles, each (E, p)"; "1.2M training jets, public
benchmark; AUC and rejection at 30% signal efficiency".

Say: "At the Large Hadron Collider, a fast top quark decaying into quarks
leaves one jet with three sub-jets; ordinary quark and gluon jets usually
have one. Each jet is a set of up to two hundred particles, each an
energy-momentum four-vector. On the public benchmark with 1.2 million
training jets we report the AUC and the background rejection. The physics is
the same for every observer, so the best taggers build Lorentz symmetry in."

### 3. Lorentz transformations are complex rotations (0:45–1:30)

On the slide, three equation lines with captions:

| equation | caption |
|---|---|
| SO⁺(1,3) ≅ SO(3,ℂ),  like F = E + iB | Lorentz group = complex rotations |
| z_a = (E_a **P** − E_P **p**_a) + i (**p**_a × **P**) | each particle paired with the jet |
| s_ab = z_a · z_b  (complex, no conjugate) | invariant under every proper Lorentz transformation |

Small picture: particles → complex 3-vectors z_a.

Say: "Our building block is a fact from relativity: the Lorentz group is the
group of rotations of complex three-vectors. The electromagnetic field
already works this way: E plus i B turns under rotations, and boosts mix its
real and imaginary parts. So we turn each particle into such a vector: the
'electric' and 'magnetic' parts of the particle paired with the whole jet,
the second line. A Lorentz transformation of the jet then simply rotates all
these vectors, and their complex dot products s_ab do not change. Everything
the network decides is built from invariants like these."

### 4. An attention-like update that is a Lorentz rotation (1:30–2:35)

On the slide, four equation lines with captions, and a small diagram: the
state z_a, the consensus r_a, the axis a_a = z_a × r_a, and z_a turning about
that axis away from r_a (schematic):

| equation | caption |
|---|---|
| w_ab = MLP(s_ab, s_aa, s_bb, h_a, h_b) | pair weight from invariants only (+ pair features, slide 5) |
| r_a = (1/n) Σ_b w_ab z_b | weighted consensus of the others |
| a_a = z_a × r_a | rotation axis (soft-bounded) |
| z_a ← exp(−[a_a]_×) z_a | an exact Lorentz rotation, closed form |

Under the equations: "z_a · z_a conserved by each rotation · covariant by
construction · complex Rodrigues formula, no ODE solver".

Say: "This is the core of SO3C, and it works like attention. For every pair
of particles, a small network turns their invariants into a weight, like an
attention score. The weighted average of the other particles' vectors gives
a consensus r_a. But instead of adding it to the particle's state, we take
its cross product with the state as a rotation axis and rotate the state by
the complex rotation it generates. That rotation is a Lorentz
transformation, computed in closed form by a complex Rodrigues formula, with
no differential-equation solver. It keeps each state's invariant length z
dot z exactly. Two beam vectors mark the collision axis, as in LorentzNet
and PELICAN; transform the jet and the beams together and the output does
not change, which a unit test confirms to better than one part in a
billion."

### 5. Physics pair features: a gain for 192 parameters (2:35–3:35)

On the slide: left, the four pair features (ln Δ, ln k_T, ln z, ln m²) "as
Lorentz invariants of the momenta and the beams"; right, the bar chart of the
gain on three protocols (chart data below); under it, "+192 parameters
(+0.8%), about the same training time".

Say: "The step that paid most was not more capacity. The Particle
Transformer feeds every pair of particles four numbers: their distance in
rapidity and azimuth, relative transverse momentum, momentum sharing and pair
mass. Three of them depend on the frame, so we rewrote them as invariants
built with the two beam vectors; the pair mass already is one. On three
training protocols, each against its own control runs, the gain was about
plus 0.0005, 0.0006 and 0.0010 in AUC, and every run with the features beat
every control run. It costs 192 parameters and about the same training time;
on the long protocol, five times the parameters gained no more. A short
ablation says the distance, transverse-momentum and momentum-sharing
features carry it."

### 6. 23k parameters, AUC 0.9846: 0.0024 below the best, not state of the art (3:35–4:30)

On the slide: the scatter of parameters (log scale) against AUC (chart data
below); big numbers "AUC 0.9846 ± 0.0003 (sd over two seeds)" and "ensemble
of two: 0.9852"; "gap to PELICAN: 0.0063 → 0.0024 (single) / 0.0018
(ensemble)"; "rejection 1302 ± 89 (ensemble 1422) against PELICAN's 2250".

Say: "Here is where that leaves us: parameters on a log scale against AUC.
With 23 thousand parameters our model reaches 0.9846, the mean of two seeds,
and an ensemble of the two reaches 0.9852. That is above ResNeXt with a
million and a half parameters, but below ParticleNet and the Particle
Transformer at 0.9858, and below the best Lorentz-symmetric models. The AUC
gap to PELICAN is now 0.0024 for a single model and 0.0018 for the ensemble,
down from 0.0063. In background rejection the gap is larger: 1302 against
2250. PELICAN's own small versions are strong too, so this is not a
state-of-the-art or a size claim: the claim is the construction, and the
steps that closed the gap."

### 7. Summary (4:30–5:00)

On the slide: Lorentz group = complex rotations; attention-like updates that
are exact Lorentz rotations; physics pair features in invariant form, the
largest step since the beams; AUC 0.9846 (ensemble 0.9852) with 23k
parameters; open code, results and checkpoints: DOI + QR.

Say: "To summarise: Lorentz transformations as complex rotations, an
attention-like update that moves each particle's state by an exact Lorentz
rotation, a cheap physics-informed step, and an honest benchmark. The code,
the results behind every number and the checkpoints are open under this DOI.
Next we add a pair state between particles, as in PELICAN, and more compute.
Thank you."

### Backup B1 (questions only). Same seed, same code, different run

On the slide: two validation-loss curves of one seed from two separate
Kaggle jobs, apart from epoch 1 (drawn schematically and labelled so, or the
numbers: epoch-1 loss 4.5e-3 apart at an identical batch order; final AUC up
to 0.00018 apart, as far as different seeds); "→ compare against pooled
controls".

Say, if asked about statistics or reproducibility: "On Kaggle GPUs with
torch dot compile, the same seed and the same code in two separate jobs gave
different training runs: with a verified identical batch order the losses
differed from the first epoch, and two full runs of one seed ended as far
apart as two different seeds. Most likely each job compiles its own GPU
kernels. So we compare every model against pooled control runs."

## Chart data (use these numbers; do not read them from `figures/`)

The committed figures in `figures/` are stale (drawn 2026-09-11, before the
beams model); draw new charts from the data below.

Gain in AUC from the pair features, mean ± standard error, each against its
own control runs (every pair-feature run beat every control run on all three):

| protocol | gain | ± | runs (arm vs control) |
|---|---|---|---|
| short probe (K=32, 400k jets, 20 epochs) | +0.00051 | 0.00009 | 3 vs 6 |
| long (K=32, 1.2M jets, 60 epochs) | +0.00064 | 0.00009 | 2 vs 5 |
| headline (K=64, 1.2M jets, 60 epochs) | +0.00100 | 0.00029 | 2 vs 2 |

For comparison, a muted bar beside the long-protocol bar only: 5.3×
parameters (wider edge network, 120,202 parameters), K=32 long, torch 2.10:
+0.00048 ± 0.00022 (2 runs vs 3).

Parameters against AUC (scatter, log x):

| model | params | AUC |
|---|---:|---|
| SO3C, pair features, 120 epochs (mean of 2) | 23,026 | 0.98460 |
| SO3C, pair features, ensemble of 2 | 46,052 | 0.98515 |
| SO3C, beams only, 120 epochs | 22,834 | 0.98392 |
| ResNeXt | 1,460,000 | 0.9837 |
| ParticleNet | 366,000 | 0.9858 |
| ParT | 2,100,000 | 0.9858 |
| LorentzNet | 224,000 | 0.9868 |
| PELICAN | 208,000 | 0.9870 |
| L-GATr | 1,100,000 | 0.9870 |
| PELICAN 11k | 11,000 | 0.9858 |
| PELICAN 3k | 3,000 | 0.9850 |
| PELICAN 1k | 1,000 | 0.9835 |
| PFN | 82,000 | 0.9819 |
| P-CNN | 348,000 | 0.9803 |

Gap to PELICAN since the first message-passing model, for a progress line if
wanted: 0.0063 (message passing, 30 epochs, 2 seeds) → 0.0037 (+ beams,
8 channels, 3 seeds) → 0.0031 (120 epochs, one run) → 0.0024 (+ pair
features, mean of 2) → 0.0018 (ensemble of 2).

## The equations, exactly (for checking; `docs/so3c/theory.md`)

- §1: SO(3,ℂ) = {Q ∈ ℂ³ˣ³ : QᵀQ = I, det Q = 1} is isomorphic to the proper
  orthochronous Lorentz group; the physical model is the Riemann–Silberstein
  vector F = E + iB.
- §9: z = bivec(p, q) = (E_p **q** − E_q **p**) + i (**p** × **q**); the jet
  lift is z_a = bivec(p_a, P) with P = Σ_a p_a. Invariant under the proper
  orthochronous group; under parity s_ab → conj(s_ab).
- §10: the lift is first bounded, z_a ← z_a / (1 + |z_a · z_a|^{1/2}). One
  round r (per channel c; channels then mixed by a complex matrix; T = 1 in
  every trained model): e_ab = tanh(W_r[asinh(Re, Im) of s^(c)_ab for all c;
  with --pair-lab the four pair features ln Δ, ln k_T, ln z, ln m²;
  asinh(Re, Im) of s^(c)_aa, s^(c)_bb; h_a; h_b]); the weight w_r^(c)(e_ab)
  is real, one per channel, unbounded and can be negative;
  r_a = (1/n) Σ_b w_r(e_ab) z_b, with b and n running over all nodes (the
  constituents, the two beams and a itself, whose term drops out of
  z_a × r_a); a_a = (z_a × r_a) / (1 + |(z_a × r_a)·(z_a × r_a)|^{1/2});
  z_a ← exp(−T[a_a]_×) z_a. The slide's "w_ab = MLP(...)" stands for
  w_r(e_ab).
- §3: exp(tA) z = z + (sin θ/θ)(b × z) + ((1 − cos θ)/θ²) b × (b × z), b = t a,
  θ² = b · b ∈ ℂ: the complex Rodrigues formula.
- §4: the flow conserves z · z exactly; the channel mixing after each round
  does not (§10: the invariant state size grows 4–8× over the rounds).
- The weighted sum is divided by the node count n, not softmax-normalised,
  and the weights can be negative: "attention-like", not attention in the
  transformer sense. For real vectors the update is dz/dt = (z·r) z − (z·z) r,
  so a state always turns away from its own r_a: with positive weights away
  from the other particles, with negative weights towards them. For complex
  states this is only a picture, because the same element also boosts.

## Questions to expect (2 minutes)

- **Is it really attention?** Attention-like: pair scores computed from
  invariants weight the other particles' vectors. Two differences: the
  weighted sum is divided by the number of particles, not
  softmax-normalised, and the weights can be negative; and its cross product
  with the state is used as a rotation axis, so each state moves by a group
  element that keeps z · z fixed (a learned channel mixing follows each
  round).
- **How is it different from LorentzNet or L-GATr?** As far as we know, both
  update their vector-valued features additively, through residual updates:
  LorentzNet adds an invariant-weighted sum of the other four-vectors (its
  paper calls this Minkowski dot-product attention), and L-GATr adds the
  outputs of equivariant attention and of other equivariant layers on
  multivectors. SO3C instead moves each complex state by an exponential, a
  Lorentz rotation, in closed form. Whether that structure is the better
  inductive bias is the research question. (Check the L-GATr paper,
  arXiv:2411.00446, before the talk: the repository notes describe only its
  attention block.)
- **Why not just use PELICAN or LorentzNet?** It is now 0.0024 AUC from
  PELICAN with about a ninth of its 208k parameters, and 0.0018 for an
  ensemble of two (about two ninths). PELICAN's own small models are strong
  too (its 3k model reaches 0.9850), so the argument is not parameter count.
- **Do lab-frame features not break Lorentz symmetry?** They are written as
  Minkowski products with the two beam vectors, so the network stays exactly
  covariant when the beams are transformed with the jet; with the beams fixed,
  only rotations about the beam axis remain, the same residual symmetry
  LorentzNet and PELICAN have with their beam inputs.
- **Two seeds is not much.** The gain was measured three times, on three
  protocols, each against its own controls (t = 5.9, 6.8 and 3.5; the K=64 one,
  two runs each, is p ≈ 0.1 on its own and gets its weight from the other
  two). Every pair-feature run beat every control run on all three protocols.
  Backup slide B1 explains why runs are compared against pooled controls.
- **Compute?** Free Kaggle T4 GPUs with a 30-hour weekly quota. One 120-epoch
  model takes about 17 hours on one T4. Two train side by side on the two GPUs
  of one session, over two sessions of about 9 hours each, so the pair cost
  about 18 of the 30 weekly hours.
- **What next?** A pair state carried between particles (as in PELICAN)
  together with the pair features, wider edge networks, and more compute.

## Prompt for Claude Design

Paste this, with the repository attached:

```
Make a 16:9 presentation in English: 7 timed slides plus 1 backup slide, for
a 5-minute talk (+2 minutes questions) at the V International Conference
"Young Professionals" (ITMO), section "AI in applied data". The audience is
applied machine-learning researchers, mostly not particle physicists.

Use ONLY these two files from the repository as content:
- so3c_notes/presentation_plan_itmo.md: the slide plan, the headlines, the
  on-slide content and equations, the speaker notes ("Say:") and the chart
  data;
- so3c_notes/presentation_brief.md: background, if a detail is needed.
Do not take numbers or equations from anywhere else, and do not reuse the
images in figures/: they are outdated. Draw new charts from the "Chart data"
tables.

Follow the slide plan's order, headlines, equations, charts and numbers.
Each headline states the slide's takeaway; keep the plan's headlines.
Condense each slide's "On the slide" text to at most ~25 words (equations,
their captions and chart labels excluded), keep every number, and put the
"Say:" text, with the slide's time window, into the speaker notes.

Equations (slides 3 and 4): typeset them properly, large and centred, at
most four lines per slide, each with its short caption from the plan.
Colour the complex covariant vectors (z, r, a) in the accent colour, the
momenta and energies (p, P, E) in normal text colour, and the invariant
scalars (s, w, h) in dark grey. Copy them exactly as in the plan.

Visuals:
1. Slide 3: particles -> complex 3-vectors z_a (small sketch beside the
   equations).
2. Slide 4: a geometric sketch: state z_a, consensus r_a, rotation axis
   a_a = z_a x r_a, and z_a turning about that axis AWAY from r_a
   (schematic, labelled as such).
3. Slide 5: a bar chart of the AUC gain from the pair features on three
   protocols, with error bars; beside the long-protocol bar only, a muted
   reference bar "5.3x parameters (K=32 long only)", +0.00048 +- 0.00022.
4. Slide 6: a scatter of parameters (log x) against AUC; published taggers
   in grey with labels; the SO3C points highlighted in the accent colour.
5. Backup B1: two diverging loss curves, drawn schematically and labelled as
   schematic, or the two numbers given in the plan.

Style: clean, light background, one accent colour for SO3C, sans-serif,
large numbers for the headline results, no clip art. Keep it honest: do not
call the model state of the art, novel or parameter-efficient, and call the
update "attention-like", not attention; the plan's wording is deliberate.
Leave placeholders for the affiliation and for the DOI / QR code.
```
