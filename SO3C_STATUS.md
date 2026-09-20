# SO3C status

Last updated 2026-09-20. This is the single place to start from. Commit hashes point at the evidence for every number; unless marked **(SO3C)** they are commits in the **so33** repository this one was split from (see [Provenance](#provenance)).

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

  Above that band sits only L-GATr **fine-tuned after JetClass pretraining** (0.98793 / 2894 ± 84, same table): a different comparison class, not the target here.

## Current best

`so3c_message_set` with **beams + channels 8**, K=64, canonical protocol, 30 epochs, Adam 3e-3 cosine, P100 float32, 22834 parameters (`b72437f`):

| | AUC | 1/ε_B@0.3 | 1/ε_B@0.5 |
|---|---|---|---|
| seed 0 (P100) | 0.98326 | 1147.2 | 287.6 |
| seed 1 (P100) | 0.98340 | 1115.5 | 297.4 |
| seed 2 (T4, rewritten forward pass) | 0.98312 | 1004.5 | 287.2 |
| **mean** | **0.98326 ± 0.00014** | **1089 ± 75** | **291 ± 5** |
| **three-seed ensemble** | **0.98378** | **1160** | **316** |

**The rejection is much noisier than two seeds suggested.** Seeds 0 and 1 alone gave 1131 ± 22; the third seed landed at 1004.5 and the three-seed standard deviation is 75, that is 7%. The AUC is stable to 0.00014. Rejection at 30% signal efficiency is a tail statistic — a few hundred background jets decide it — so **a rejection gain under ~150 on a single seed is not evidence**, while +0.0004 AUC remains about three standard deviations. Seed 2 is not the rewrite's doing: its training curve matches seed 0's (final validation loss 0.16455 against 0.16449) and the two implementations agree to 1e-14 in float64.

- **Gap to PELICAN:** 0.0037 AUC and a factor 2.1 in rejection (from 0.0063 and 2.6 before beams).
- **The sobering comparison (verified):** PELICAN's own 1k-parameter model (0.9835 / 1145) matches this 22.8k-parameter model. The gap is architectural, not a matter of size.
- **Exact equivariance, measured** (float64, `87206e7`): with the beams transformed along with the jet, AUC stays 0.9684 at every boost scale (drop −0.0001); with the beams pinned to the lab it falls to 0.9332. That is exact covariance, the correct physics, and the same property PELICAN has.
- **Reproduce:** `notebooks/kaggle_k64_headline_seed2.ipynb` (one session, ~5.2 h since the speedup) for a new seed; seeds 0 and 1 came from `notebooks/kaggle_beams_k64_seed{0,1}.ipynb` plus `kaggle_beams_k64_finish.ipynb` (they stopped at epoch 24 on the session cap and resumed). Seed 2 with its checkpoint and log is in `results_k64/headline/`; results, the epoch-30 checkpoints of seeds 0 and 1 and the full kernel logs are in `results_message/beams_k64/` (weights archived from Kaggle in `a76d837`; every other SO3C run's checkpoints sit beside its results the same way). Ensemble: `python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64`.

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
| kNN(16) graph at K=32 | −0.0011 AUC for 2% wall clock saved. Note 16 neighbours cover half the jet at K=32 and an eighth at K=128, so the penalty should grow, not shrink, with K | `1f01ddd` |
| Capacity **without** beams (K=32 probe) | flat: scalar_dim 24, hidden 256, channels 8/16, rounds 6 all within 0.0006 AUC of the anchor | `abd5660` |
| Scalar channel | load-bearing: −0.0025 AUC and −40% rejection without it | `abd5660` |
| Published LorentzNet/PELICAN recipe (AdamW wd 0.01, dropout 0.2, lr 1e-3, 35-epoch warm restarts) | **negative**: 0.97769 alone (below anchor 0.9786), and beams + recipe 0.98065 < beams alone 0.98088 | `837c567` |
| Learning rate | 3e-3 beats both 6e-3 (−0.0003) and 1e-3 | `c325950` |
| ~~Longer training~~ **REOPENED 2026-09-20** | The verdict rested on one probe-protocol row (35 vs 20 epochs on 400k jets, K=32: +0.00015 AUC, −14 rejection). Against it: validation loss is lowest at the **final** epoch in all four K=64 runs, and on the older 9k `so3c_equivariant_set` at K=32 on the full training set, 60 epochs beat 30 by +0.00024 AUC (6.4 σ) and +30 rejection (9.5%, 5.2 σ) — `results_60ep` against its three 30-epoch seeds, same batch, same learning rate, only the epoch count differing. Being tested at K=64 | `c325950`, `results_60ep` |
| Regularisation | dropout 0.05 + wd 1e-4: −0.0011 AUC, −133 rejection. The model **underfits** | `c325950` |
| Capacity **with** beams | channels 8: +0.0004 AUC, +67 rejection; **channels 16: worse than 8** (0.98111 / 811) at 1.5× cost | `c325950`, `90f2ead` |
| K=128 wall clock | dense 602 s/epoch on 100k jets (60.8 h full protocol); kNN(16) 191 s (19.3 h), 3.15× faster | `90f2ead` |
| Time-axis reference particle (1,0,0,0), gap-analysis finding 23 | redundant with beams: b+ + b- = 2 (1,0,0,0), so <p, t> = (<p,b+> + <p,b->)/2 is already available from inputs the model has. Closed by argument, not run | 2026-09-13 |
| Two-seed ensemble | +0.0004 AUC; member scores correlate at 0.994 | `49e0c61`, `b72437f` |
| Epochs 25–30 vs 24 (session cap) | +0.00022 AUC | `b72437f` |
| **Jet multiplicity as an input** (true count, count clipped at 64, energy fraction in the leading 64) | Stacked on the two-seed ensemble's test scores, fit on one half of the test set and scored on the other: **+0.00002 AUC**; rejection +65, inside the ±60 spread between the two halves. The model already carries it. Diagnostic only (fits on test jets) | (SO3C) `benchmarks/diag_multiplicity_stacking.py`, `results_message/beams_k64/multiplicity_stacking.json` |
| Best-validation checkpoint | Validation loss is lowest at the **final** epoch in both headline seeds and both beamless K=64 seeds (cosine schedule), so the best-validation checkpoint is the last one | (SO3C) 2026-09-19, from the stored histories |

**Lesson worth keeping:** a closed question can re-open when the inputs change. Capacity was flat without beams and paid once beams supplied the information. kNN was useless at K=32 and 3.15× faster at K=128.

## Open — candidates

### Screened on 2026-09-20, none better than the base

The two defect fixes, the vector channel, the relative-norm edge, the pair latent, their combination, depth 6 and f_α were all measured on the probe protocol (see the screen table below). None improved on the base; the interpretation waits on the seed spread being measured now.

### Verified defects in the current model (free to fix)

Both are now switchable and default to the old behaviour: `--no-mass-input --no-self-edges` (22634 params on beams + channels 8, CPU cost 1.00x the headline, `6771158`). Not yet run on GPU.

- **m² fed as noise.** `h_init` takes `asinh(m²)` per node, and three of the seven `_minkowski_stats` readout features are per-particle m² moments. All of it is rounding noise (see Closed).
- **Self-edges in the dense graph.** The dense path includes `a = b` in the scalar message sum and counts N, not N−1, in the denominator. The flow itself is unaffected, since `z_a × w_aa z_a = 0`. LorentzNet masks the self-edge; our sparse path already excludes it.

### Best-motivated architectural candidates

- **Rank-2 pair latent** with a reduced Eq2→2 aggregator basis (7 of PELICAN's 15). PELICAN carries a [B,N,N,C] state through its blocks; we collapse pairs to nodes every round. The verified table-2 comparison — 1k PELICAN parameters match our 22.8k — is consistent with this being the dominant carrier. Outcome likely bimodal: a real step toward 1400–1800 rejection, or under +0.0005 if our flow already supplies it. **Implemented** as `--pair-latent N`: 3680 parameters at C_p = 8 on beams + channels 8, CPU cost 1.56x the headline (~16.5 h per 30-epoch seed at K=64), 1.64x combined with the vector channel and both fixes. Exactly Lorentz covariant (1.7e-16) and permutation invariant (0.0), with the 7 aggregator maps checked for permutation equivariance on their own. Not yet run on GPU. Agent estimate: screen at 20% data × 3 seeds ≈ 11 GPU-h.
- **A vector channel alongside the bivector.** Two reasons, one measured elsewhere and one structural:
  - arXiv:2606.21790 (abstract, verified) finds that in L-GATr "bivector channels are negligible for top-quark tagging while vector-like channels are dominant". The LLoCa (2505.20280) and slim L-GATr (2512.17011) abstracts do not address grades; their bodies are unchecked.
  - Our lift `z_a = bivec(p_a, P)` is **unchanged under p_a → p_a + λP**, so it discards each constituent's component along the jet axis. The scalar channel recovers ⟨p_a, P⟩ only as an invariant. A covariant vector v_a = p_a keeps it — the same kind of missing-information problem that beams fixed. **Implemented** as `--vector-channel`: 531 parameters on beams + channels 8, CPU cost 1.10x the headline, exactly covariant to 7e-15 with and without beams, and the vector update moves the logits by 7e-2 to 9e-2. Not yet run on GPU.

### Cheap candidates (unverified gain)

| candidate | mechanism | note |
|---|---|---|
| N^α / N̄^α aggregation rescaling | sum-vs-mean semantics; our flow angle scales with Σ_b w_b z_b | ~0 runtime |
| Time-axis reference particle (1,0,0,0) | a third symmetry-breaking reference, as in L-GATr -- **redundant with beams** (b+ + b- = 2t; see Closed) | 1.03× cost |
| Softmax attention over w_b | normalised aggregation; needed for depth | ~1.05× |
| Lion optimiser, lr 3e-4, wd 0.2 | a different optimiser, not the ruled-out AdamW bundle | a training-limit control |
| JetClass pretraining | the only published result above PELICAN is **L-GATr fine-tuned after JetClass pretraining**: AUC 0.98793 ± 0.00001, 1/ε_B 2894 ± 84 (arXiv:2411.00446 table 2, verified). That is a **different comparison class** and not our target — the same L-GATr *trained from scratch* on this dataset scores 0.9870 ± 0.0001 / 2240 ± 70 at 1.1M parameters, matching PELICAN rather than beating it | ≥ 25 GPU-h truncated; changes the comparison class |

Full agent text: `so3c_notes/gap_to_sota_2026-09-13.md`. Three of the workflow's six agents failed on the usage limit, so the findings there are unranked.

## GPU window 2026-09-19 → 2026-09-26 (in progress)

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

**Before calling these closed, the protocol's own seed spread is being measured** — it never has been. The ±0.00014 AUC figure comes from K=64 runs on the full training set, and the K=64 *rejection* spread turned out to be three times what two seeds suggested. `nsanya/so3c-probe-seeds-{base,pair8}` (`4769835`) repeat the base and the closest candidate over three seeds each on exactly this protocol, with the rewritten forward pass and `--compile` so the comparison is paired. ~3.2 GPU-h.

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

### 60 epochs at K=64 (`nsanya/so3c-k64-headline60-seed0`, `fcc137a`)

The headline configuration, seed 0, 60 epochs on the full training set, one session. Cost from measured numbers: seed 2 ran at 621 s per epoch eager on a T4, and the profile's compiled/eager ratios (0.82 per training step, 0.61 per eval chunk) give ~495 s per epoch, so ~8.3 h; on a P100 compile is skipped and it is ~10.4 h, still inside the 12 h session cap, with `--max-seconds 39000` so either way it checkpoints cleanly. The summary prints the epoch-by-epoch validation curve, which is the point: whether the curve is still falling at 60.

Why this and not more probes: every K=64 run so far has its lowest validation loss at the final epoch, and the one doubling ever measured on the full training set (`results_60ep`, the older 9k model) paid 6 σ in AUC and 5 σ in rejection. If the model is simply undertrained at 30 epochs, then every capacity candidate has been screened in a regime where it could not pay, and the screen has to be redone at the longer schedule.

### Sparse graph now carries beams (`829c4c0`)

The dense-only restriction on beams was conservative, not structural; it is lifted, with the two beam nodes forced into every node's neighbourhood (a fixed, frame-independent index set) so a ranked list cannot discard the only lab-frame information the model has. Cost: k + 2 edges per node. This unlocks K=128 with kNN and beams at roughly 7.7 h per 30-epoch seed, but that experiment is **not** scheduled: its case rests on +0.0009 AUC with rejection flat on the 9k covariant model, and the kNN penalty should grow with K.

### Budget

GPU used this window: ~9.2 h (screen) + 5.2 h (headline seed 2) + ~0.7 h (profiles, diagnostics, two guard failures) + ~3.1 h (seed replicates) ≈ 18.3 h of 30, leaving ~11.7 h. The 60-epoch run takes ~8.3 h of that; the width probes (~4.8 h) do not fit alongside it and are held for the next window. A 30-epoch K=64 seed now costs ~3.9 h with `--compile`, a 60-epoch one ~8.3 h, and a probe row 0.4–0.9 h.

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
| Kaggle may hand out a **Tesla T4** (Python 3.12, torch 2.10 preinstalled) instead of the P100 | all three profile kernels on 2026-09-19 got a T4; timings do not transfer between cards blindly | record the device with every timing (the nvidia-smi cell, `summary.json`) |
| Kaggle's API has **no cancel** (the status enum has CANCEL_REQUESTED, set only from the web UI) | the gap screen was already running on the old, 2× slower forward pass when the speedup landed | push expensive kernels only once the code they depend on is final; a cancel needs the user in the web UI |
| Bash collapses backslashes inside notebook-builder cells too | an escaped newline in a generated `print` became a real line break, and the profile kernel's last cell died with a SyntaxError (its data was already saved) | write builder fragments with the Write tool or use `chr(10)`; `ast.parse` every generated code cell before pushing |
| A root-level ignore written as `data/` also matches `tests/data/` | the golden fixture could not be added | anchor root-only ignores: `/data/` (SO3C `2a3b344`) |
| **Exactly two** Kaggle GPU sessions at once | a third push returns HTTP 200 with "Maximum batch GPU session count of 2 reached." and is refused, not queued; the client printed it as "pushed" | use both slots, push the next kernel when one frees; `kaggle_client push` exits 2 on a refusal (SO3C `8d7dff7`) |
| A numerical fixture written on one platform, checked on another | the on-card guard failed at 4e-8 on every case; a diagnostic kernel showed even the **old** code on Kaggle's CPU was 4e-8 off, while new vs old on the T4 agreed to 3e-15 (`results_profile/t4_cuda_diag_3a3b10b.txt`). `nn.Linear` initialises in float32 before `.to(float64)`, and float32 init differs at the ulp level between torch versions | draw fixture parameters and inputs from numpy's legacy RandomState straight into float64, and compare against the shipped reference implementation on the same device (SO3C `89d67f3`) |
| Kaggle's image has a regular top-level `tests` package in site-packages | while our `tests/` was a namespace package, `tests.message_golden` failed to import on the card and the golden guard stopped headline seed 2 after ~5 min (log in `results_k64/headline/`) | `tests/__init__.py` (SO3C `4f397f4`); replay bundles with a fake shadowing `tests` package on PYTHONPATH |

## Compute and quota facts

| configuration | seconds per epoch, full train set, P100 | 30 epochs |
|---|---|---|
| `so3c_covariant_set`, K=64 | ~199 | 1.66 h |
| `so3c_message_set`, K=64 | ~930 | 7.75 h |
| + beams + channels 8, K=64 | 1272 | 10.6 h |
| + beams + channels 8, K=64, **rewritten forward pass, T4** (measured, seed 2) | 621 | 5.2 h |
| K=128 dense (from 100k jets) | ~7290 | 60.8 h |
| K=128 kNN(16) (from 100k jets) | ~2313 | 19.3 h |

- The K=32 probe protocol (400k train jets, 20 epochs) costs 0.6–1.2 h per row.
- Quota is 30 GPU-h per week, resetting on Saturday. The window that opened 2026-09-19 is in use; see [GPU window](#gpu-window-2026-09-19--2026-09-26-in-progress).
- The table above is P100 wall clock on the old forward pass. Since `47ec4ad` the per-epoch cost is about half; the T4 estimates are in the GPU-window section, and headline seed 2 will give the measured figure.

## Provenance

- This repository is a **snapshot** of the so33 repository's branch `feature/so3c-complexification`, taken at its tip `d360acc` (2026-09-13) and pushed as a single root commit, `6f79e70`, tagged `pre-so33-removal`. It shares no history with so33.
- **Every commit hash in this document refers to the so33 repository**, not to a commit reachable from here. The same holds for the hashes in the Closed table and in the commit messages inherited from before the snapshot.
- Before the bench is deleted from so33, **tag `d360acc` there** (for example `so3c-snapshot-2026-09-13`), so those hashes stay reachable after that repository is pruned.
- SO(3,3)-only content was removed here on 2026-09-18 in four commits: `e322d9e`, `f3f1568`, `09aa178` and the packaging/documentation commit. [`REMOVED.md`](REMOVED.md) lists what went, what was kept on purpose and what changed as a result.
- so33 keeps its copy of this bench for now; it remains the home of the SO(3,3) paper, its Zenodo DOI and its `CITATION.cff`.
