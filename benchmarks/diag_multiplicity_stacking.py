"""
benchmarks.diag_multiplicity_stacking
-------------------------------------
Does jet multiplicity carry signal the headline model does not use?

Post-hoc stacking on the saved per-jet test scores of the current best model
(beams + channels 8, K=64, seeds 0 and 1, ensemble-averaged). The canonical
test set is split in two fixed halves; a small gradient-boosted model is fit
on half A with [score] or [score, extra jet features] and evaluated on half
B, then the halves swap. A gain over the score-only fit means the feature
holds information the tagger is not using.

Extra features, from the raw 200-slot test parquet: the true multiplicity,
the multiplicity the model sees (clipped at K=64), and the fraction of jet
energy in the 64 leading-pT constituents. The score files are aligned with
the parquet by checking that the 404k label sequences are identical.

Diagnostic only: it fits on test jets, so no number from it may be reported
as a tagger result.

Run:
    python -m benchmarks.diag_multiplicity_stacking results_message/beams_k64/multiplicity_stacking.json
"""
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results_message" / "beams_k64"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None


def rej_at(y, s, eff):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(tpr, eff)
    return 1.0 / max(fpr[i], 1e-12)


# --- raw test jets: multiplicity and energy beyond the kept 64 -------------
pf = pq.ParquetFile(REPO / "data" / "toptagging" / "test.parquet")
cols = ["E_%d" % i for i in range(200)] + ["PX_%d" % i for i in range(200)] + \
       ["PY_%d" % i for i in range(200)] + ["is_signal_new"]
t = pf.read(columns=cols)
E = np.stack([t.column("E_%d" % i).to_numpy() for i in range(200)], axis=1)
PX = np.stack([t.column("PX_%d" % i).to_numpy() for i in range(200)], axis=1)
PY = np.stack([t.column("PY_%d" % i).to_numpy() for i in range(200)], axis=1)
lab = t.column("is_signal_new").to_numpy().astype(int)
pt = np.hypot(PX, PY)
real = pt > 0
n_true = real.sum(1)
# leading-64 by pT, as select_leading_constituents does
order = np.argsort(-np.where(real, pt, -1.0), axis=1)
Es = np.take_along_axis(np.where(real, E, 0.0), order, axis=1)
frac64 = Es[:, :64].sum(1) / Es.sum(1)
n_obs64 = np.minimum(n_true, 64)
n_obs32 = np.minimum(n_true, 32)

# --- scores, aligned by the label sequence --------------------------------
S = []
for seed in (0, 1):
    z = np.load(RES / ("top_tagging_canonical__so3c_message_set__seed%d__scores.npz" % seed))
    assert np.array_equal(z["labels"].astype(int), lab), "label order differs from the parquet"
    S.append(z["scores"].astype(np.float64))
score = (S[0] + S[1]) / 2.0
print("labels aligned: 404000 jets, signal fraction %.4f" % lab.mean())
for nm, s in (("seed0", S[0]), ("seed1", S[1]), ("ensemble", score)):
    print("  %-8s AUC %.5f  rej@0.3 %.1f" % (nm, roc_auc_score(lab, s), rej_at(lab, s, 0.3)))
print("n_true: signal mean %.1f, background mean %.1f, clipped at 64: %.1f%%"
      % (n_true[lab == 1].mean(), n_true[lab == 0].mean(), 100 * (n_true > 64).mean()))
print("n_true alone AUC %.4f" % roc_auc_score(lab, n_true))

logit = np.log(np.clip(score, 1e-7, 1 - 1e-7) / np.clip(1 - score, 1e-7, 1))
rng = np.random.default_rng(12345)
half = rng.permutation(len(lab)) < len(lab) // 2

feature_sets = {
    "score only": [logit],
    "+ n_obs(64)": [logit, n_obs64],
    "+ n_true": [logit, n_true],
    "+ n_true, frac64": [logit, n_true, frac64],
    "+ n_obs(64), frac64": [logit, n_obs64, frac64],
}
rows = {}
for name, feats in feature_sets.items():
    X = np.stack(feats, axis=1)
    pred = np.zeros(len(lab))
    for a in (half, ~half):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                             max_leaf_nodes=15, random_state=0)
        clf.fit(X[a], lab[a])
        pred[~a] = clf.predict_proba(X[~a])[:, 1]
    # AUC per held-out half, then averaged, so the two fits never share a ROC
    aucs = [roc_auc_score(lab[~a], pred[~a]) for a in (half, ~half)]
    rejs = [rej_at(lab[~a], pred[~a], 0.3) for a in (half, ~half)]
    rows[name] = {"auc": float(np.mean(aucs)), "auc_halves": aucs,
                  "rej30": float(np.mean(rejs)), "rej30_halves": rejs}
    print("%-22s AUC %.5f (halves %.5f %.5f)  rej@0.3 %7.1f (halves %.1f %.1f)"
          % (name, rows[name]["auc"], *aucs, rows[name]["rej30"], *rejs))

base = rows["score only"]
for name, r in rows.items():
    print("  %-22s dAUC %+.5f  d rej %+7.1f" % (name, r["auc"] - base["auc"], r["rej30"] - base["rej30"]))

if OUT:
    OUT.write_text(json.dumps({"rows": rows,
                               "n_true_mean_sig": float(n_true[lab == 1].mean()),
                               "n_true_mean_bkg": float(n_true[lab == 0].mean()),
                               "clipped64_frac": float((n_true > 64).mean())}, indent=1))
