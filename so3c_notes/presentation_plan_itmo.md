# Talk plan: V International Conference «Young Professionals» (ITMO)

Section: AI in applied data. Level: international. Slot: 5 minutes talk + 2
minutes questions. Language: English.

Numbers below come from [`presentation_brief.md`](presentation_brief.md),
[`SO3C_STATUS.md`](../SO3C_STATUS.md), `docs/so3c/experiments.md` and
`figures/*.csv`, all checked against the result files; if they ever disagree,
`SO3C_STATUS.md` wins.

## The one message

A neural network built directly from the mathematics of special relativity,
small enough to train on free GPUs, took its largest step since the beams by
adding physics knowledge in a symmetry-preserving form: 192 parameters gave as
much as a network with five times the parameters.

Audience: applied machine-learning researchers, mostly not particle
physicists. Lead with the data problem and the inductive-bias idea; keep the
group theory to one picture.

## Slide plan (7 slides, 5:00)

The "Say:" text is 656 words, 4.7 minutes at 140 words a minute and 5.0 at
130; it leaves little room to slow down. Visible slide text stays short; detail goes into the
speaker notes.

### 1. Title (0:00–0:10)

- Headline: **SO3C: a Lorentz-covariant neural network for particle jets, built on complexified rotations**
- Aleksandr Panchenko, [affiliation]; conference name and date; QR code to the DOI.

Say: "Good afternoon. I will show a small neural network whose structure
follows special relativity, and the one step that made it much better."

### 2. Jet tagging is set classification on four-vectors (0:10–0:50)

On the slide: a top jet (three sub-jets) beside a QCD jet (one); "a jet = a
set of up to 200 particles, each (E, p)"; "1.2M training jets, public
benchmark; AUC and rejection at 30% signal efficiency".

Say: "At the Large Hadron Collider, a fast top quark decaying into quarks
leaves one jet with three sub-jets; ordinary quark and gluon jets usually
have one. Each jet is an unordered set
of up to two hundred particles, each an energy-momentum four-vector, so this
is set classification. We use the public benchmark with 1.2 million training
jets and report the AUC and the background rejection at thirty percent signal
efficiency. The physics is the same for every observer, which is Lorentz
symmetry, and the strongest taggers build it into the network."

### 3. Lorentz transformations are complex rotations (0:50–1:50)

On the slide: one left-to-right schematic: four-momenta → complex 3-vectors
relative to the jet → message passing by complex rotations (closed form) →
invariant read-out → top / QCD, with two beam arrows along the collision axis.
One line: SO⁺(1,3) ≅ SO(3,ℂ), "exact by construction".

Say: "Our starting point is a fact from relativity: Lorentz transformations
act exactly like rotations of complex three-vectors, the way electric and
magnetic fields combine into E plus i B. So each particle becomes a complex
three-vector measured relative to the whole jet. The network then passes
messages between particles: in each round it rotates every vector by a
learned complex rotation, computed in closed form with a complex Rodrigues
formula, so no differential-equation solver is needed. Because the vectors only ever move by such rotations
and everything else sees only invariants, the symmetry is exact by
construction: boost the jet and the two beam vectors
together and the output does not change; a unit test confirms it to better
than one part in a billion. The beams mark the collision axis, as they do in
LorentzNet and PELICAN."

### 4. Physics pair features: a gain for 192 parameters (1:50–3:05)

On the slide: left, the four pair features (ln Δ, ln k_T, ln z, ln m²) "as
Lorentz invariants with the beams"; right, the bar chart of the gain on three
protocols (chart data below); under it, "+192 parameters (+0.8%), about the
same training time".

Say: "The step that paid was not more capacity. The Particle Transformer
feeds every pair of particles four numbers: their distance in rapidity and
azimuth, their relative transverse momentum, how they share momentum, and
their pair mass. Three of them depend on the frame, so we rewrote them as
invariants built from the momenta and the two beam vectors; the pair mass
already is one. So the network stays exactly symmetric. We tested it on three training protocols of
increasing size, each against its own control runs without the features. The
gain was about plus 0.0005, plus 0.0006 and plus 0.0010 in AUC, and every run
with the features beat every control run. It costs 192 parameters, under one
percent, and about the same training time. On the same long protocol, a network
with five times the parameters gained no more. A short ablation says the
distance, transverse-momentum and momentum-sharing features are enough to
carry the gain."

### 5. 23k parameters, AUC 0.9846: 0.0024 below the best, not state of the art (3:05–4:00)

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

### 6. Same seed, same code, different run (4:00–4:35)

On the slide: two validation-loss curves of one seed from two separate Kaggle
jobs, apart from epoch 1 (drawn schematically and labelled so, or the
numbers: epoch-1 loss 4.5e-3 apart at an identical batch order; final AUC up
to 0.00018 apart, as far as different seeds); "→ compare against pooled
controls".

Say: "One practical lesson for applied machine learning. On Kaggle GPUs with
torch dot compile, the same seed and the same code in two separate jobs gave
different training runs: with a verified identical batch order the losses
differed from the first epoch, and two full runs of one seed ended as far
apart as two different seeds. Most likely each job compiles its own GPU
kernels. So we compare every model against pooled control runs."

### 7. Summary (4:35–5:00)

On the slide: exact Lorentz covariance from complex rotations, 23k
parameters; physics pair features in invariant form, the largest step since
the beams; AUC 0.9846 (ensemble 0.9852), 0.0024 (ensemble 0.0018) below
PELICAN and L-GATr trained on this dataset; open code, results and
checkpoints: DOI + QR.

Say: "To summarise: an exactly symmetric network built from complex
rotations, a cheap physics-informed step, the largest since the beams, and an
honest benchmark. The code, the results behind every number and the
checkpoints are open under this DOI. Next we add a pair state between
particles, as in PELICAN, and more compute. Thank you."

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

## Questions to expect (2 minutes)

- **Why not just use PELICAN or LorentzNet?** A different construction of the
  same symmetry: complex rotations in closed form, very small. Whether that
  structure is a good inductive bias is the research question; it is now
  0.0024 AUC from PELICAN with about a ninth of its 208k parameters, and
  0.0018 for an ensemble of two (about two ninths). PELICAN's own small models
  are strong too (its 3k model reaches 0.9850), so the argument is not
  parameter count.
- **Do lab-frame features not break Lorentz symmetry?** They are written as
  Minkowski products with the two beam vectors, so the network stays exactly
  covariant when the beams are transformed with the jet; with the beams fixed,
  only rotations about the beam axis remain, the same residual symmetry
  LorentzNet and PELICAN have with their beam inputs.
- **Two seeds is not much.** The gain was measured three times, on three
  protocols, each against its own controls (t = 5.9, 6.8 and 3.5; the K=64 one,
  two runs each, is p ≈ 0.1 on its own and gets its weight from the other
  two). Every pair-feature run beat every control run on all three protocols.
- **Why complexified rotations?** SO(3,ℂ) is isomorphic to the proper
  orthochronous Lorentz group SO⁺(1,3), so complex 3-vectors carry exact
  Lorentz transformations, and the exponential has a closed form: no ODE
  solver.
- **Compute?** Free Kaggle T4 GPUs with a 30-hour weekly quota. One 120-epoch
  model takes about 17 hours on one T4. Two train side by side on the two GPUs
  of one session, over two sessions of about 9 hours each, so the pair cost
  about 18 of the 30 weekly hours.
- **What next?** A pair state carried between particles (as in PELICAN)
  together with the pair features, wider edge networks, and more compute.

## Prompt for Claude Design

Paste this, with the repository attached:

```
Make a 7-slide, 16:9 presentation in English for a 5-minute talk (+2 minutes
questions) at the V International Conference "Young Professionals" (ITMO),
section "AI in applied data". The audience is applied machine-learning
researchers, mostly not particle physicists.

Use ONLY these two files from the repository as content:
- so3c_notes/presentation_plan_itmo.md: the slide plan, the headlines, the
  on-slide content, the speaker notes ("Say:") and the chart data;
- so3c_notes/presentation_brief.md: background, if a detail is needed.
Do not take numbers from anywhere else, and do not reuse the images in
figures/: they are outdated. Draw new charts from the "Chart data" tables.

Follow the slide plan's order, headlines, charts and numbers. Each headline
states the slide's takeaway; keep the plan's headlines. Condense each slide's
"On the slide" content to at most ~25 words of visible text (chart labels
excluded), keep every number it contains, and put the "Say:" text, with the
slide's time window, into the speaker notes.

Charts to draw:
1. Slide 4: a bar chart of the AUC gain from the pair features on three
   protocols, with error bars; beside the long-protocol bar only, a muted
   reference bar "5.3x parameters (K=32 long only)", +0.00048 +- 0.00022.
2. Slide 5: a scatter of parameters (log x) against AUC; published taggers
   in grey with labels; the SO3C points highlighted in one accent colour.
3. Slide 3: a simple left-to-right schematic: 4-momenta -> complex
   3-vectors -> message passing with complex rotations -> invariant read-out
   -> top / QCD, with two beam arrows along the collision axis.
4. Slide 6: two diverging loss curves, drawn schematically and labelled as
   schematic, or the two numbers given in the plan.

Style: clean, light background, one accent colour for SO3C, sans-serif,
large numbers for the headline results, no clip art. Keep it honest: do not
call the model state of the art, novel or parameter-efficient; the plan's
wording is deliberate. Leave placeholders for the affiliation and for the
DOI / QR code.
```
