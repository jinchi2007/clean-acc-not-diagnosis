#!/usr/bin/env python3
"""bench_mps.py - CPU vs Apple-GPU (MPS) benchmark for a 1D-CNN.

Runs the same synthetic bearing-vibration workload (train and inference) on
CPU and MPS, reports per-epoch time, throughput and the speed-up.

    source setup/env.sh
    python setup/bench_mps.py            # 3 epochs per device
    python setup/bench_mps.py --epochs 5 --batch-size 128
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from verify_env import BearingCNN1D, make_bearing_data


@torch.no_grad()
def infer_throughput(model, x, y, device, bs=256):
    model.eval()
    correct = 0
    t0 = time.perf_counter()
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs].to(device)
        yb = y[i:i + bs].to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
    if device.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    return correct / x.shape[0], x.shape[0] / dt


def run(device: torch.device, epochs: int, bs: int, n: int, length: int):
    torch.manual_seed(0)
    x, y = make_bearing_data(n=n, length=length)
    model = BearingCNN1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=bs, shuffle=True, drop_last=True)

    # warm-up so the first timed epoch is not dominated by lazy init / graph capture
    model.train()
    xb, yb = next(iter(dl))
    out = model(xb.to(device))
    loss_fn(out, yb.to(device)).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.synchronize()

    epoch_times = []
    for ep in range(epochs):
        model.train()
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.numel()
            correct += (out.argmax(1) == yb).sum().item()
            tot += yb.numel()
        if device.type == "mps":
            torch.mps.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        print(f"    epoch {ep + 1:>2}: {epoch_times[-1] * 1000:8.1f} ms  "
              f"{tot / epoch_times[-1]:9.0f} samples/s  "
              f"loss={loss_sum / tot:.4f}  acc={correct / tot:.3f}")

    acc, infer_sps = infer_throughput(model, x, y, device)
    return {
        "mean_epoch_s": sum(epoch_times) / len(epoch_times),
        "infer_acc": acc,
        "infer_sps": infer_sps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--length", type=int, default=1024)
    args = ap.parse_args()

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    else:
        print("!! torch.backends.mps.is_available() == False -> only CPU will be benchmarked\n")

    print(f"workload: 1D-CNN, {args.samples} x 1 x {args.length}, "
          f"batch={args.batch_size}, epochs={args.epochs}, threads={torch.get_num_threads()}")
    print(f"torch {torch.__version__}, mps available={torch.backends.mps.is_available()}\n")

    results = {}
    for dev in devices:
        print(f"== {str(dev).upper()} ==")
        torch.set_num_threads(min(8, torch.get_num_threads())) if dev.type == "cpu" else None
        results[dev.type] = run(dev, args.epochs, args.batch_size, args.samples, args.length)

    print("\n" + "=" * 66)
    print(f"{'device':<8}{'mean epoch (ms)':>18}{'train samples/s':>19}{'infer samples/s':>18}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<8}{r['mean_epoch_s'] * 1000:>18.1f}"
              f"{args.samples / r['mean_epoch_s']:>19.0f}{r['infer_sps']:>18.0f}")
    if "mps" in results and "cpu" in results:
        print("-" * 66)
        print(f"speed-up on MPS: "
              f"train {results['cpu']['mean_epoch_s'] / results['mps']['mean_epoch_s']:.2f}x, "
              f"infer {results['mps']['infer_sps'] / results['cpu']['infer_sps']:.2f}x")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
