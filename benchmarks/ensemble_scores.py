"""
benchmarks.ensemble_scores
--------------------------
Test metrics of a seed ensemble, computed from the per-jet scores each run
already saved -- no model is re-run.

Every run writes ``{experiment}__{model}__seed{s}__scores.npz`` with the
positive-class probability and the label of every canonical test jet, in
the loader's fixed order. Averaging the probabilities over seeds gives the
ensemble's score for each jet.

Before any ensemble number is printed, each member's AUC and background
rejection are recomputed from its own scores and compared with the values
the runner stored. A mismatch means the metric code here disagrees with the
runner's, and the script stops rather than report. Members whose test
labels differ -- a different ordering or a different test set -- stop it
too.

Measured on K=64 message passing (2026-09-12): seeds 0 and 1 score AUC
0.98077 / 0.98068 and 1/eps_B at eps_S = 0.3 of 889.5 / 810.9; their
two-seed ensemble scores 0.98109 and 901.4. Their scores correlate at
0.9953, which is why averaging buys so little.

Seeds may live in several directories (runs from different kernels);
pass them all to --results-dir. Every directory must hold at least one
matching scores file, so a mistyped directory stops the script instead of
quietly shrinking the ensemble. A seed number appearing twice is an error:
two files with one seed are usually one run in two places (a copy, a
worktree, a re-download). Two different runs can share a seed number,
though -- the 30- and the 60-epoch headline runs are both seed 0 -- and
--allow-shared-seed accepts that. The same file twice, or two files with
identical scores, is an error either way.

Run:
    python -m benchmarks.ensemble_scores --results-dir results_message/k64
    python -m benchmarks.ensemble_scores --results-dir results_message/beams_k64 results_k64/headline
    python -m benchmarks.ensemble_scores --allow-shared-seed --results-dir results_message/beams_k64 results_k64/headline results_k64/headline60
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np


def rejection(y: np.ndarray, s: np.ndarray, eff: float) -> float:
    """1 / eps_B at signal efficiency eff, interpolated on the ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return float(1.0 / np.interp(eff, tpr, fpr))


def metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    return {"auc": float(roc_auc_score(y, s)),
            "bg_rej_30": rejection(y, s, 0.3),
            "bg_rej_50": rejection(y, s, 0.5)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, nargs="+", required=True)
    p.add_argument("--model", type=str, default="so3c_message_set")
    p.add_argument("--experiment", type=str, default="top_tagging_canonical")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="Allowed mismatch between a member's recomputed and "
                        "stored metric, relative to max(1, stored value).")
    p.add_argument("--out", type=str, default=None,
                   help="Also write the summary as JSON to this path.")
    p.add_argument("--allow-shared-seed", action="store_true",
                   help="Accept different runs that share a seed number (for "
                        "example a 30- and a 60-epoch run of seed 0, in "
                        "different directories). The same file twice, or two "
                        "files with identical scores, is still an error.")
    args = p.parse_args(argv)

    def seed_of(f: pathlib.Path) -> int:
        return int(re.search(r"seed(\d+)__scores", f.name).group(1))

    pattern = f"{args.experiment}__{args.model}__seed*__scores.npz"
    folders = [pathlib.Path(d) for d in args.results_dir]
    empty = [d.as_posix() for d in folders if not any(d.glob(pattern))]
    if empty:
        raise SystemExit(f"[ensemble] no {pattern} in {empty}")
    files = sorted((f for d in folders for f in d.glob(pattern)),
                   key=lambda f: (seed_of(f), str(f)))
    resolved = [f.resolve() for f in files]
    if len(set(resolved)) != len(resolved):
        raise SystemExit(f"[ensemble] the same scores file appears twice across "
                         f"{args.results_dir}")
    seeds = [seed_of(f) for f in files]
    shared = len(set(seeds)) != len(seeds)
    if shared and not args.allow_shared_seed:
        raise SystemExit(f"[ensemble] a seed appears twice across {args.results_dir}: "
                         f"{seeds} (pass --allow-shared-seed if these are different runs)")
    if len(files) < 2:
        print(f"[ensemble] need at least two scored seeds in {args.results_dir}, "
              f"found {len(files)}", file=sys.stderr)
        return 1

    labels = None
    probs: list[np.ndarray] = []
    members: list[dict] = []
    for f in files:
        seed = seed_of(f)
        z = np.load(f)
        y, s = z["labels"], z["scores"]
        where = f"{f.parent.as_posix()}/{f.name}"
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise SystemExit(f"[ensemble] {where}: test labels differ from the "
                             f"first member, so the jets are not in the same order")
        for g, prev in zip(files, probs):
            if np.array_equal(prev, s):
                raise SystemExit(f"[ensemble] {where} holds the same scores as "
                                 f"{g.parent.as_posix()}/{g.name}: one run in two places")
        stored = json.loads(f.with_name(f.name.replace("__scores.npz", ".json"))
                            .read_text(encoding="utf-8"))["test_metrics"]
        m = metrics(y, s)
        for key_stored, key in (("test_auc", "auc"), ("bg_rej_30", "bg_rej_30"),
                                ("bg_rej_50", "bg_rej_50")):
            if key_stored not in stored:
                continue
            ref = stored[key_stored]
            if abs(m[key] - ref) > args.tolerance * max(1.0, abs(ref)):
                raise SystemExit(f"[ensemble] seed {seed} ({where}): recomputed {key} "
                                 f"{m[key]:.5f} does not match stored {ref:.5f}")
        member = {"seed": seed, **m}
        if shared:
            member["source"] = f.parent.as_posix()
        members.append(member)
        probs.append(s)

    ensemble = metrics(labels, np.mean(probs, axis=0))
    corr = np.corrcoef(np.stack(probs))
    mean_corr = float(corr[np.triu_indices(len(probs), 1)].mean())

    print(f"{'member':<12}{'AUC':>10}{'rej@0.3':>10}{'rej@0.5':>10}")
    for m in members:
        print(f"{'seed ' + str(m['seed']):<12}{m['auc']:>10.5f}"
              f"{m['bg_rej_30']:>10.1f}{m['bg_rej_50']:>10.1f}"
              + (f"  {m['source']}" if shared else ""))
    mean_auc = float(np.mean([m["auc"] for m in members]))
    mean_rej = float(np.mean([m["bg_rej_30"] for m in members]))
    print(f"{'ensemble':<12}{ensemble['auc']:>10.5f}"
          f"{ensemble['bg_rej_30']:>10.1f}{ensemble['bg_rej_50']:>10.1f}")
    print(f"\n{len(members)} {'members' if shared else 'seeds'}; ensemble minus member mean: "
          f"{ensemble['auc'] - mean_auc:+.5f} AUC, "
          f"{ensemble['bg_rej_30'] - mean_rej:+.1f} rejection at 0.3; "
          f"mean score correlation {mean_corr:.4f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "results_dir": [d.as_posix() for d in folders], "model": args.model,
            "experiment": args.experiment, "members": members,
            "ensemble": ensemble, "mean_score_correlation": mean_corr,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
