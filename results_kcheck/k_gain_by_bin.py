"""Where does K=32 -> K=64 help? Per-bin AUC gain of the K=64 pair-feature
runs (results_k64/pairlab60, seeds 0 and 1) over the K=32 ones
(results_long_seeds/pairlab211, seeds 0 and 1): same stack, features and 60
epochs, same 404k test jets. Bins: multiplicity (n <= 32 is the control:
both models see every constituent), prong separation (dR3_min, dR2), and
multiplicity x dR3_min. Per bin: the mean AUC of each K over its two seeds,
the gain, the gain as a fraction of the K=32 model's remaining error
(1 - AUC), a paired bootstrap SE of the gain (resampling jets, all four
models on the same resample), the seed spread, and the pT fraction each
model never sees. Then two crude extrapolations to K=128: the per-bin gain
fitted against the pT share, and against the number, of constituents only
the K=64 model sees, with the slope applied to ranks 65-128.

AUC levels are not comparable across bins (their signal fractions differ
widely); only the K=32 vs K=64 difference within a bin is. CPU, about 15
minutes. Usage, from the repo root, after results_kcheck/jet_geometry.py:

    python results_kcheck/k_gain_by_bin.py [results_kcheck/jet_geometry.npz]

Writes results_kcheck/k_gain_by_bin.json and prints the tables
(results_kcheck/k_gain_by_bin.txt is that output).
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEOM = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_kcheck" / "jet_geometry.npz"
OUT = ROOT / "results_kcheck" / "k_gain_by_bin.json"
g = dict(np.load(GEOM))
y = g["label"].astype(int)
for k in ("dR3_min", "dR2"):
    g[k] = np.where(np.isfinite(g[k]), g[k], np.nan)
hid32 = g["f33_64"] + g["f65_128"] + g["f129"]      # pT the K=32 model never sees
hid64 = g["f65_128"] + g["f129"]                    # pT the K=64 model never sees

runs = {}
for name, d in (("k32", "results_long_seeds/pairlab211"), ("k64", "results_k64/pairlab60")):
    for s in (0, 1):
        z = np.load(ROOT / d / ("seed%d" % s) / ("top_tagging_canonical__so3c_message_set__seed%d__scores.npz" % s))
        assert np.array_equal(z["labels"].astype(int), y)
        runs[(name, s)] = z["scores"].astype(np.float64)
for k, v in runs.items():
    print("overall %s seed %d AUC %.5f" % (k[0], k[1], roc_auc_score(y, v)))


def rej30(lab, sc):
    fpr, tpr, _ = roc_curve(lab, sc)
    e = np.interp(0.3, tpr, fpr)
    return 1 / e if e > 0 else np.inf


rng = np.random.default_rng(0)
NBOOT = 300


def summarise(name, sel):
    idx = np.flatnonzero(sel)
    lab = y[idx]
    if lab.min() == lab.max():
        return None
    auc = {k: roc_auc_score(lab, v[idx]) for k, v in runs.items()}
    a32 = (auc[("k32", 0)] + auc[("k32", 1)]) / 2
    a64 = (auc[("k64", 0)] + auc[("k64", 1)]) / 2
    boots = []
    for _ in range(NBOOT):
        b = idx[rng.integers(0, len(idx), len(idx))]
        lb = y[b]
        if lb.min() == lb.max():
            continue
        r = {k: roc_auc_score(lb, v[b]) for k, v in runs.items()}
        boots.append((r[("k64", 0)] + r[("k64", 1)]) / 2 - (r[("k32", 0)] + r[("k32", 1)]) / 2)
    j32 = np.mean([rej30(lab, runs[("k32", s)][idx]) for s in (0, 1)])
    j64 = np.mean([rej30(lab, runs[("k64", s)][idx]) for s in (0, 1)])
    return {"bin": name, "n": int(len(idx)), "sig_frac": float(lab.mean()),
            "auc32": a32, "auc64": a64, "gain": a64 - a32, "gain_rel": (a64 - a32) / (1 - a32),
            "gain_se": float(np.std(boots)),
            # seed noise of the gain: half the spread of the four seed-by-seed differences
            "seed_gain_spread": float(np.ptp([auc[("k64", a)] - auc[("k32", b)] for a in (0, 1) for b in (0, 1)]) / 2),
            "hid32": float(hid32[idx].mean()), "hid64": float(hid64[idx].mean()),
            "frac_mult_gt64": float((g["mult"][idx] > 64).mean()),
            "rej32": j32, "rej64": j64}


def table(title, bins):
    print("\n" + title)
    print("%-22s %7s %5s %8s %8s %9s %6s %7s %7s %7s %7s %6s %7s %7s"
          % ("bin", "n", "sig", "AUC32", "AUC64", "gain", "rel", "+-boot", "+-seed", "hid32", "hid64", ">64", "rej32", "rej64"))
    rows = []
    for name, sel in bins:
        r = summarise(name, sel)
        if r is None:
            print("%-22s %7d  (one class only)" % (name, sel.sum()))
            continue
        rows.append(r)
        print("%-22s %7d %5.2f %8.5f %8.5f %+9.5f %5.1f%% %7.5f %7.5f %7.4f %7.4f %6.2f %7.0f %7.0f"
              % (r["bin"], r["n"], r["sig_frac"], r["auc32"], r["auc64"], r["gain"], 100 * r["gain_rel"], r["gain_se"],
                 r["seed_gain_spread"], r["hid32"], r["hid64"], r["frac_mult_gt64"], r["rej32"], r["rej64"]))
    return rows


m = g["mult"]
out = {}
out["all"] = table("all jets", [("all", np.ones_like(y, bool))])
out["mult"] = table("by multiplicity (n <= 32: both models see every constituent)",
                    [("n <= 32", m <= 32), ("33 <= n <= 48", (m > 32) & (m <= 48)),
                     ("49 <= n <= 64", (m > 48) & (m <= 64)), ("65 <= n <= 80", (m > 64) & (m <= 80)),
                     ("n > 80", m > 80)])
for var in ("dR3_min", "dR2"):
    v = g[var]
    q = np.nanquantile(v, [0.2, 0.4, 0.6, 0.8])
    edges = [-np.inf] + list(q) + [np.inf]
    out[var] = table("by %s quintile (edges %s)" % (var, ", ".join("%.3f" % e for e in q)),
                     [("%s q%d" % (var, i + 1), (v > edges[i]) & (v <= edges[i + 1])) for i in range(5)])
v = g["dR3_min"]
t = np.nanquantile(v, [1 / 3, 2 / 3])
cells = []
for lo, hi, nm in ((-np.inf, t[0], "close"), (t[0], t[1], "mid"), (t[1], np.inf, "wide")):
    for mlo, mhi, mn in ((0, 64, "n<=64"), (64, 10_000, "n>64")):
        cells.append(("%s, %s" % (nm, mn), (v > lo) & (v <= hi) & (m > mlo) & (m <= mhi)))
out["dR3_x_mult"] = table("dR3_min tercile x multiplicity (edges %.3f, %.3f)" % tuple(t), cells)

# Does the gain follow what the K=32 model never saw? Weighted fits of the
# per-bin gain across the multiplicity and dR3_min x multiplicity bins,
# against the pT share and against the number of constituents only the K=64
# model sees; the same slope applied to ranks 65-128 (what K=128 would add).
m_f = g["mult"].astype(float)
add32_64 = np.minimum(m_f, 64) - np.minimum(m_f, 32)
add64_128 = np.minimum(m_f, 128) - np.minimum(m_f, 64)
sels = {"n <= 32": m <= 32, "33 <= n <= 48": (m > 32) & (m <= 48), "49 <= n <= 64": (m > 48) & (m <= 64),
        "65 <= n <= 80": (m > 64) & (m <= 80), "n > 80": m > 80}
sels.update(dict(cells))
rows = out["mult"] + out["dR3_x_mult"]
yv = np.array([r["gain"] for r in rows])
w = 1 / np.array([r["gain_se"] ** 2 + r["seed_gain_spread"] ** 2 for r in rows])
out["fits"] = {}
for label, xnow, xnext, overall_next in (
        ("pT share", lambda s_: g["f33_64"][s_].mean(), lambda s_: g["f65_128"][s_].mean(), g["f65_128"].mean()),
        ("count", lambda s_: add32_64[s_].mean(), lambda s_: add64_128[s_].mean(), add64_128.mean())):
    x = np.array([xnow(sels[r["bin"]]) for r in rows])
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], yv * np.sqrt(w), rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ A)
    chi2 = float((w * (yv - A @ coef) ** 2).sum())
    print("\nweighted fit, gain = a + b * (%s only K=64 sees): a %+.5f +- %.5f, b %+.6f +- %.6f, chi2 %.1f on %d dof"
          % (label, coef[0], np.sqrt(cov[0, 0]), coef[1], np.sqrt(cov[1, 1]), chi2, len(rows) - 2))
    print("  naive overall 64 -> 128: b x mean over jets of ranks 65-128 (%.4f) = %+.5f" % (overall_next, coef[1] * overall_next))
    for r in rows:
        s_ = sels[r["bin"]]
        print("  %-22s x %8.4f  gain %+.5f  fit %+.5f | ranks 65-128 %8.4f -> naive 64->128 %+.5f"
              % (r["bin"], xnow(s_), r["gain"], coef[0] + coef[1] * xnow(s_), xnext(s_), coef[1] * xnext(s_)))
    out["fits"][label] = {"a": float(coef[0]), "b": float(coef[1]), "chi2": chi2, "dof": len(rows) - 2,
                          "naive_overall_64_128": float(coef[1] * overall_next)}
r_pt = g["f65_128"].mean() / g["f33_64"].mean()
r_n = add64_128.mean() / add32_64.mean()
g_all = out["all"][0]["gain"]
print("\nmean per jet: ranks 33-64 pT share %.4f, %.2f constituents; ranks 65-128 pT share %.4f, %.2f constituents"
      % (g["f33_64"].mean(), add32_64.mean(), g["f65_128"].mean(), add64_128.mean()))
print("overall 32 -> 64 gain %+.5f scaled by the pT-share ratio %.3f: %+.5f; by the count ratio %.3f: %+.5f"
      % (g_all, r_pt, g_all * r_pt, r_n, g_all * r_n))
out["ratio_extrapolation"] = {"pt_ratio": float(r_pt), "count_ratio": float(r_n),
                              "by_pt": float(g_all * r_pt), "by_count": float(g_all * r_n)}
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("\nsaved", OUT.relative_to(ROOT))
