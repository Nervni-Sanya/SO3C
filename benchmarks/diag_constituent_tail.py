"""
benchmarks.diag_constituent_tail
--------------------------------
How much of a top-tagging jet lies beyond the K leading constituents?

The models keep the K hardest constituents by transverse momentum (K=64 for
the headline numbers; datasets.select_leading_constituents). A K=128 run
costs about four times as much per epoch (the dense graph scales with K^2),
so before spending GPU time on it this measures, on the raw 200-slot parquet
files, what K=64 throws away:

  * the multiplicity: how many jets have more than 32, 64 and 128 real
    (pT > 0) constituents, per class;
  * the scalar-pT fraction carried by the constituents ranked 33-64, 65-128
    and 129-200 by pT: the per-jet fraction sum_{rank in band} pT / sum pT,
    summarised by its mean, median, 90th, 99th percentile and maximum, and
    the share of jets where it exceeds 1% and 5%;
  * the same for energy.

CPU only; reads the parquet in batches. Usage:

    python -m benchmarks.diag_constituent_tail --data-dir data/toptagging \
        --out results_kcheck/constituent_tail.json
"""

import argparse
import json
import pathlib

import numpy as np

BANDS = {"33-64": (32, 64), "65-128": (64, 128), "129-200": (128, 200)}
THRESHOLDS = (32, 64, 128)


def _summary(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.90)), "p99": float(np.quantile(x, 0.99)),
            "max": float(x.max()), "frac_above_1pct": float((x > 0.01).mean()),
            "frac_above_5pct": float((x > 0.05).mean())}


def analyse(path: pathlib.Path, batch_rows: int = 50_000) -> dict:
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    names = f.schema_arrow.names
    nslot = sum(1 for n in names if n.startswith("PX_"))
    cols = ([f"E_{i}" for i in range(nslot)] + [f"PX_{i}" for i in range(nslot)]
            + [f"PY_{i}" for i in range(nslot)] + ["is_signal_new"])
    mult, label = [], []
    pt_frac = {b: [] for b in BANDS}
    e_frac = {b: [] for b in BANDS}
    for batch in f.iter_batches(batch_size=batch_rows, columns=cols):
        t = batch.to_pydict()
        E = np.stack([np.asarray(t[f"E_{i}"], dtype=np.float64) for i in range(nslot)], 1)
        px = np.stack([np.asarray(t[f"PX_{i}"], dtype=np.float64) for i in range(nslot)], 1)
        py = np.stack([np.asarray(t[f"PY_{i}"], dtype=np.float64) for i in range(nslot)], 1)
        pt = np.sqrt(px * px + py * py)
        real = pt > 0.0
        # rank by pT, as select_leading_constituents does (padding sorts last)
        order = np.argsort(-np.where(real, pt, -1.0), axis=1)
        pt_s = np.take_along_axis(np.where(real, pt, 0.0), order, 1)
        e_s = np.take_along_axis(np.where(real, E, 0.0), order, 1)
        tot_pt, tot_e = pt_s.sum(1), e_s.sum(1)
        for b, (lo, hi) in BANDS.items():
            pt_frac[b].append(pt_s[:, lo:hi].sum(1) / tot_pt)
            e_frac[b].append(e_s[:, lo:hi].sum(1) / tot_e)
        mult.append(real.sum(1))
        label.append(np.asarray(t["is_signal_new"]))
    mult = np.concatenate(mult)
    label = np.concatenate(label).astype(int)
    pt_frac = {b: np.concatenate(v) for b, v in pt_frac.items()}
    e_frac = {b: np.concatenate(v) for b, v in e_frac.items()}
    out = {"n_jets": int(len(mult)), "slots": nslot}
    for name, sel in (("all", np.ones_like(label, bool)), ("signal", label == 1),
                      ("background", label == 0)):
        m = mult[sel]
        out[name] = {
            "n_jets": int(sel.sum()),
            "multiplicity": {"mean": float(m.mean()), "median": float(np.median(m)),
                             "p99": float(np.quantile(m, 0.99)), "max": int(m.max()),
                             **{"frac_more_than_%d" % k: float((m > k).mean()) for k in THRESHOLDS}},
            "pt_fraction": {b: _summary(v[sel]) for b, v in pt_frac.items()},
            "energy_fraction": {b: _summary(v[sel]) for b, v in e_frac.items()},
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", default="data/toptagging")
    p.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    p.add_argument("--out", default="results_kcheck/constituent_tail.json")
    args = p.parse_args(argv)
    res = {}
    for split in args.splits:
        res[split] = analyse(pathlib.Path(args.data_dir) / f"{split}.parquet")
        a = res[split]["all"]
        print("%-10s %8d jets | >32 %.4f  >64 %.4f  >128 %.5f | pT in 65-128: mean %.5f p99 %.5f max %.4f"
              % (split, res[split]["n_jets"], a["multiplicity"]["frac_more_than_32"],
                 a["multiplicity"]["frac_more_than_64"], a["multiplicity"]["frac_more_than_128"],
                 a["pt_fraction"]["65-128"]["mean"], a["pt_fraction"]["65-128"]["p99"],
                 a["pt_fraction"]["65-128"]["max"]), flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
