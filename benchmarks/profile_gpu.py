"""
benchmarks.profile_gpu
----------------------
Where does so3c_message_set spend its GPU time?

Beams + channels 8 at K=64 costs 1272 s per epoch on a P100, about 0.24 s
per training step of 256 jets. The (B, K, K, *) edge tensors the model
builds are a few hundred MB each, so a bandwidth estimate puts a step
several times lower than that. CPU profiles cannot settle it: on the CPU
the complex batched matmuls run as per-matrix loops and dominate, which
says nothing about cuBLAS. This script measures on the card itself.

Synthetic massless jets are enough, because the cost depends only on shapes:
the flow has no data-dependent branches (expm_so3c evaluates both the series
and the closed form everywhere). For each configuration it reports:

  * a training step (forward, backward, Adam) and an eval forward pass,
    timed with CUDA synchronisation after warm-up;
  * peak memory;
  * the CUDA kernels ranked by self time, and the same time grouped by the
    Python source line that launched it, so the cost maps onto model code.

Run:
    python -m benchmarks.profile_gpu --out profile_out \
        --config base --config vector --config pair8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from benchmarks.so3c_models import SO3CMessageSetClassifier

CONFIGS = {
    "base": {},
    "vector": {"vector_channel": True},
    "pair8": {"pair_latent": 8},
    "fixes": {"mass_input": False, "self_edges": False},
    "rounds1": {"rounds": 1},
}


def synthetic_batch(B: int, K: int, device, dtype, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = torch.randint(K // 3, K + 1, (B,), generator=g)
    mask = (torch.arange(K)[None, :] < n[:, None]).to(dtype)
    pv = (torch.randn(B, K, 3, generator=g, dtype=dtype) * 0.3
          + torch.tensor([3.0, 0.0, 0.0], dtype=dtype))
    E = pv.norm(dim=-1, keepdim=True)
    x = torch.cat([torch.cat([E, pv], -1) * mask[..., None], mask[..., None]], -1)
    y = torch.randint(0, 2, (B,), generator=g)
    return x.to(device), y.to(device)


def build(config: str, channels: int, dtype, device):
    kw = dict(channels=channels, rounds=3, beams=True)
    kw.update(CONFIGS[config])
    model = SO3CMessageSetClassifier(dtype=dtype, **kw).to(device)
    # Excite the zero-initialised heads: the identity flow at step 0 is not
    # the regime training runs in, even if the cost barely depends on it.
    with torch.no_grad():
        for h in model.w_head:
            h.weight.normal_(0.0, 0.3)
        for h in model.v_head:
            h.weight.normal_(0.0, 0.3)
    return model, kw


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_steps(model, x, y, steps: int, device):
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    lossf = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()

    for _ in range(3):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    return (time.perf_counter() - t0) / steps, step


def time_eval(model, x, reps: int, device):
    with torch.no_grad():
        model(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        _sync(device)
    return (time.perf_counter() - t0) / reps


def profile_step(step, device, top: int):
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, with_stack=True, record_shapes=True) as prof:
        for _ in range(2):
            step()
        _sync(device)
    key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    by_kernel = prof.key_averages().table(sort_by=key, row_limit=top)
    # Source lines need Python stacks, which some builds do not record; the
    # input shapes always identify the tensor, e.g. (B, K, K, C) edge
    # features against (B*C*K, 3, 3) flow matrices.
    by_line = prof.key_averages(group_by_stack_n=4).table(sort_by=key, row_limit=top)
    by_shape = prof.key_averages(group_by_input_shape=True).table(
        sort_by=key, row_limit=top, max_name_column_width=40,
        max_shapes_column_width=90)
    return by_kernel, by_line, by_shape


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", action="append", choices=sorted(CONFIGS))
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-chunk-size", type=int, default=1024)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--out", default="profile_out")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = args.config or ["base"]
    summary = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "k": args.k, "batch_size": args.batch_size, "dtype": args.dtype,
               "torch": torch.__version__, "rows": []}
    print("device %s | K=%d | batch %d | %s" % (summary["device"], args.k,
                                              args.batch_size, args.dtype), flush=True)

    for cfg in configs:
        model, kw = build(cfg, args.channels, dtype, device)
        n_params = sum(q.numel() for q in model.parameters())
        x, y = synthetic_batch(args.batch_size, args.k, device, dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_step, step = time_steps(model, x, y, args.steps, device)
        peak = (torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else float("nan"))
        xe, _ = synthetic_batch(args.eval_chunk_size, args.k, device, dtype, seed=1)
        t_eval = time_eval(model, xe, max(2, args.steps // 4), device)
        row = {"config": cfg, "kwargs": kw, "n_params": n_params,
               "train_step_s": t_step, "eval_chunk_s": t_eval,
               "peak_train_gib": peak,
               # one epoch of the canonical protocol: 1.211M train jets and
               # the 403k validation jets evaluated once
               "epoch_estimate_s": (1_211_000 / args.batch_size) * t_step
                                   + (403_000 / args.eval_chunk_size) * t_eval}
        summary["rows"].append(row)
        print("%-8s %6d par | step %.4f s | eval chunk %.4f s | peak %.2f GiB | "
              "epoch ~%.0f s" % (cfg, n_params, t_step, t_eval, peak,
                                 row["epoch_estimate_s"]), flush=True)
        if not args.no_profile:
            tables = profile_step(step, device, args.top)
            for name, table in zip(("by_kernel", "by_line", "by_shape"), tables):
                (out / ("%s_%s.txt" % (cfg, name))).write_text(table, encoding="utf-8")
        del model, x, y, xe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
