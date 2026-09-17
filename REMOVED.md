# What was removed with the SO(3,3) split

This repository started as a snapshot of the `so33` repository, which carried
two projects at once: the SO(3,3) activation and SO3C. On 2026-09-18 everything
belonging only to SO(3,3) was removed here, so that this tree is the SO3C bench
and nothing else.

**Nothing was lost.** The SO(3,3) project keeps its own repository,
[Nervni-Sanya/so33](https://github.com/Nervni-Sanya/so33), which holds all of
it together with its preprint, its Zenodo DOI and its `CITATION.cff`. Inside
this repository the snapshot root — tagged `pre-so33-removal` — is the last
commit that still contains every file listed below.

## What went

| Removed | What it was | Where it is now |
|---|---|---|
| `so33/` | The SO(3,3) activation package: `SO33Activation`, `SO33Network`, `BottleneckClassifier`, the 15-generator basis, the ODE function | so33 repository |
| `benchmarks/models.py`: `MultiChannelSO33`, `EquivariantSO33Classifier`, the eight `so33*` registry names, and the parameters only they used (`T`, `adjoint`, `so33_method`, `so33_step_size`, `bound_input`, `max_input_norm`) | The SO(3,3) model families and their solver plumbing | so33 repository |
| `benchmarks/synthetic.py`, `run_boost_ood.py`, `run_synthetic_ood.py`, `run_synthetic_dataeff.py`, `run_synthetic_equivariance.py`, `diagnose_equivariant.py`, `figure_equivariance.py` | The SO(3,3) synthetic battery, its boost-OOD runner, the equivariance diagnosis and its figure | so33 repository |
| `tests/test_ablations_and_dtype.py`, `test_adjoint.py`, `test_basis.py`, `test_causal_classification.py`, `test_forward.py`, `test_regularization.py`, `test_training.py` | The 14 tests of the SO(3,3) package | so33 repository |
| `paper/main.tex`, `paper/refs.bib`, `paper/README.md`, `paper/runs/week1.sh`, `paper/notes/{diagnosis_findings,equivariant_diagnosis,results_week1,top_tagging_finding}.md` | The SO(3,3) preprint workspace and the week-1/week-2 notes diagnosing the `so33_equivariant` OOD failure | so33 repository |
| `paper/figures/equivariance_data.csv`, `paper/figures/equivariance_vs_norm.pdf` | Output of the deleted `figure_equivariance.py` | so33 repository |
| `REPORT.md` | The SO(3,3) benchmarking report (in Russian) | so33 repository |
| `examples/demo_minimal.py` | End-to-end demo of `SO33Activation` and `SO33Network` | so33 repository |
| `results_higgs_ablation/higgs*__so33__*` (18 files) | Per-seed JSON and per-jet scores for the `so33` rows of the HIGGS ablation | so33 repository |

## What was deliberately kept

- **`eta_invariants`** stays, reimplemented so it is self-contained. It is the
  SO(3,3)-invariant readout and the anchor row of every canonical top-tagging
  table; it needs only the metric η and the deterministic 4→6 lift, not the
  SO(3,3) activation. `DIM = 6` is defined in `benchmarks/models.py` and `ETA`
  comes from `so3c.algebra`, where it always was the same tensor.
- **`BottleneckClassifier`** was reimplemented in `benchmarks/models.py`
  (it used to be imported from `so33`), so the matched-bottleneck pointwise
  baselines — `relu_bottleneck`, `tanh_bottleneck`, `gelu_bottleneck` — still
  build and still match their recorded parameter counts.
- **`so3c_notes/related_work.md`** (was `paper/notes/related_work.md`): the
  Lorentz-equivariant literature survey. It mentions SO(3,3) nowhere and is
  what the SO3C write-up needs, so it moved instead of being deleted.
- **`paper_tables.md`** keeps its SO(3,3) rows: it is an aggregated record of
  measurements that were made, not code, and rewriting it would falsify it.
- **The `so33` rows in `docs/so3c/experiments.md`** (tables F, H and I) are
  kept as measured baselines, each marked as no longer reproducible here.
- **`checkpoints/dataprep_out/kernel.log`** keeps its `so33` paths: it is a
  log of what a Kaggle kernel actually printed.

## What changed as a result

- `build_model(name, in_features, out_features, *, natural_hidden=256,
  dtype=float64, representation="flat", pool="mean", so3c_kwargs=None)` —
  the SO(3,3) solver and input-bounding arguments are gone. SO3C's flow is a
  closed-form group element, so it never needed them.
  `run_tabular_experiment` lost its `T` passthrough for the same reason.
- `benchmarks.build_notebooks.CODE_PACKAGES` is now
  `("so3c", "benchmarks", "tests")`. Notebooks already committed were not
  regenerated; they still carry a `so33/` directory in their embedded bundle,
  which nothing imports.
- The figures moved from `paper/figures/` to [`figures/`](figures/) when the
  paper workspace was deleted. `plotting.DEFAULT_OUT_DIR` follows.
- The distribution is `so3c` (was `so33-activation`), with no version or DOI:
  SO3C has no release yet. Both repositories ship a top-level `benchmarks`
  package, so never `pip install -e` both into the same environment.
- The test suite went from 78 tests (76 passing, 2 CUDA skips) to 64 (62 and
  the same 2 skips). No SO3C or harness test was changed except
  `tests/test_so3c_algebra.py`, which now pins `so3c.algebra.ETA` against the
  written-out so(3,3) metric instead of importing `so33.basis.ETA`.

## The removal commits

`e322d9e` (model families and synthetic scripts) → `f3f1568` (the package, its
tests and the paper sources) → `09aa178` (figures out of `paper/`) → the
packaging, documentation and provenance commit that added this file.
