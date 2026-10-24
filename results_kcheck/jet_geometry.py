"""Per-jet geometry of the top-tagging test set, from the raw 200-slot parquet
(2026-10-25; input to results_kcheck/k_gain_by_bin.py).

For every test jet: label, multiplicity, the scalar-pT fraction in pT-rank
bands (33-64, 65-128, 129-200), jet mass and pT, and the prong separation
from exclusive-kT clustering (d_ij = min(pT_i^2, pT_j^2) dR_ij^2, E-scheme,
no beam distance) of the 32 hardest constituents down to 3 subjets (dR3_min:
the smallest pairwise dR among them) and to 2 subjets (dR2). Also checks
that the parquet's jet order is the order of data/top_tagging_test.npz, which
the models were tested on. CPU, about 6 minutes. Usage, from the repo root:

    python results_kcheck/jet_geometry.py [results_kcheck/jet_geometry.npz]

The output (26 MB) is not committed; rerun this to rebuild it.
"""
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_kcheck" / "jet_geometry.npz"
NCLU = 32


def rap_phi(p):
    E, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    y = 0.5 * np.log(np.clip(E + pz, 1e-12, None) / np.clip(E - pz, 1e-12, None))
    return y, np.arctan2(py, px), px * px + py * py


def dR2(y, phi):
    dy = y[:, :, None] - y[:, None, :]
    dphi = np.abs(phi[:, :, None] - phi[:, None, :])
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return dy * dy + dphi * dphi


def cluster(p, active, target):
    """Merge pairs by the kT measure until `target` objects remain per jet."""
    B, n, _ = p.shape
    iu = np.triu(np.ones((n, n), bool), 1)
    rows = np.arange(B)
    while True:
        count = active.sum(1)
        todo = count > target
        if not todo.any():
            return p, active
        y, phi, pt2 = rap_phi(p)
        d = np.minimum(pt2[:, :, None], pt2[:, None, :]) * dR2(y, phi)
        ok = active[:, :, None] & active[:, None, :] & iu[None]
        d = np.where(ok, d, np.inf)
        k = d.reshape(B, -1).argmin(1)
        i, j = k // n, k % n
        sel = rows[todo]
        p[sel, i[todo]] += p[sel, j[todo]]
        p[sel, j[todo]] = 0.0
        active[sel, j[todo]] = False


def subjet_dr(p, active):
    y, phi, pt2 = rap_phi(p)
    d = np.sqrt(dR2(y, phi))
    n = p.shape[1]
    ok = active[:, :, None] & active[:, None, :] & np.triu(np.ones((n, n), bool), 1)[None]
    return np.where(ok, d, np.inf).reshape(len(p), -1).min(1)


f = pq.ParquetFile(ROOT / "data" / "toptagging" / "test.parquet")
nslot = sum(1 for c in f.schema_arrow.names if c.startswith("PX_"))
cols = [f"{c}_{i}" for i in range(nslot) for c in ("E", "PX", "PY", "PZ")] + ["is_signal_new"]
recs = {k: [] for k in ("label", "mult", "f33_64", "f65_128", "f129", "mass", "pt", "dR3_min", "dR2", "top32")}
t0 = time.time()
for batch in f.iter_batches(batch_size=20_000, columns=cols):
    tab = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in batch.schema.names}
    P = np.stack([np.stack([tab[f"{c}_{i}"] for c in ("E", "PX", "PY", "PZ")], -1) for i in range(nslot)], 1).astype(np.float64)
    pt = np.hypot(P[..., 1], P[..., 2])
    real = pt > 0
    order = np.argsort(-np.where(real, pt, -1.0), axis=1)
    P = np.take_along_axis(P, order[..., None], 1)
    pt = np.take_along_axis(np.where(real, pt, 0.0), order, 1)
    real = np.take_along_axis(real, order, 1)
    tot = pt.sum(1)
    J = (P * real[..., None]).sum(1)
    recs["label"].append(tab["is_signal_new"].astype(np.int8))
    recs["mult"].append(real.sum(1).astype(np.int16))
    recs["f33_64"].append(pt[:, 32:64].sum(1) / tot)
    recs["f65_128"].append(pt[:, 64:128].sum(1) / tot)
    recs["f129"].append(pt[:, 128:].sum(1) / tot)
    recs["mass"].append(np.sqrt(np.clip(J[:, 0] ** 2 - (J[:, 1:] ** 2).sum(1), 0, None)))
    recs["pt"].append(np.hypot(J[:, 1], J[:, 2]))
    recs["top32"].append(pt[:, :NCLU].sum(1) / tot)
    q = P[:, :NCLU].copy() * real[:, :NCLU, None]
    act = real[:, :NCLU].copy()
    q, act = cluster(q, act, 3)
    recs["dR3_min"].append(subjet_dr(q, act))
    q, act = cluster(q, act, 2)
    recs["dR2"].append(subjet_dr(q, act))
    if len(recs["label"]) == 1:
        first = P[:1000, :32].copy(), real[:1000, :32].copy()
    print("%d jets, %.0f s" % (sum(len(x) for x in recs["label"]), time.time() - t0), flush=True)
out = {k: np.concatenate(v) for k, v in recs.items()}

# the parquet order is the order of the K=32 npz the models were trained and tested on
z = np.load(ROOT / "data" / "top_tagging_test.npz")
C = z["constituents"][:1000]
Pk, rk = first
match = np.allclose(np.abs(C).sum((1, 2)), np.abs(Pk * rk[..., None]).sum((1, 2)), rtol=1e-4) or \
        np.allclose(np.abs(C).sum((1, 2)) / np.abs(C).sum((1, 2)).mean(),
                    np.abs(Pk * rk[..., None]).sum((1, 2)) / np.abs(Pk * rk[..., None]).sum((1, 2)).mean(), rtol=1e-3)
corr = np.corrcoef(np.abs(C).sum((1, 2)), np.abs(Pk * rk[..., None]).sum((1, 2)))[0, 1]
print("order check against data/top_tagging_test.npz (first 1000 jets): match %s, corr %.6f, labels equal %s"
      % (match, corr, np.array_equal(z["labels"].astype(int), out["label"].astype(int))))
np.savez(OUT, **out)
for k in ("mult", "f33_64", "f65_128", "dR3_min", "dR2", "mass", "top32"):
    v = out[k]
    print("%-8s mean %.4f median %.4f p10 %.4f p90 %.4f | sig %.4f bkg %.4f" % (
        k, v.mean(), np.median(v), np.quantile(v, 0.1), np.quantile(v, 0.9),
        np.median(v[out["label"] == 1]), np.median(v[out["label"] == 0])))
print("done %.0f s" % (time.time() - t0))
