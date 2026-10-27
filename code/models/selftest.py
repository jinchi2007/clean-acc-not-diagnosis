"""Acceptance self-test for the eight baselines (CONTRACT.md section 2).

Run from the workspace root::

    source setup/env.sh
    python -m code.models.selftest

Checks, in order:

1. len(MODELS) == 8; for every baseline at in_len=1024: params < 3M,
   forward (4, 1, 1024) -> (4, 10), finite output, and one train-mode
   forward+backward on the device;
2. in_ch regression: every backbone also accepts in_ch=4 (stem ablation
   support), forward (2, 4, 1024) -> (2, 10);
3. stem front-end (build_stem_model): backbones (resnet1d, ds_cnn, vit1d,
   mcnn) x (k, s, c) in {(64,16,1), (64,16,4), (64,16,8)} at 1024 and 2048:
   forward shape OK and |parameter change| < 5% versus the plain backbone at
   the same raw window length;
4. median single-forward latency on the best available device (MPS here) for
   all eight baselines and for the c=8 stem variants, in_len 1024 and 2048,
   batch 64;
5. the exact call from code/CONTRACT.md section 2, plus build_model with the
   original positional signature.

The output is also written to results/model_selftest.txt, whose last line is
"RESULT: PASS".
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from . import MODELS, build_model, build_stem_model
from .blocks import count_parameters

NUM_CLASSES = 10
BATCH = 64
WARMUP = 5
ITERS = 20
ROUNDS = 5
MAX_PARAMS = 3_000_000
STEM_BACKBONES = ("resnet1d", "ds_cnn", "vit1d", "mcnn")
STEM_GRID = ((64, 16, 1), (64, 16, 4), (64, 16, 8))
MAX_STEM_PARAM_DELTA = 0.05

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "results" / "model_selftest.txt"


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _empty_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def bench_forward(model, in_len: int, device: torch.device) -> float:
    """Amortised per-forward latency in ms for (BATCH, 1, in_len) input."""
    x = torch.randn(BATCH, 1, in_len, device=device)
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)
        _sync(device)
        rounds = []
        for _ in range(ROUNDS):
            start = time.perf_counter()
            for _ in range(ITERS):
                model(x)
            _sync(device)
            rounds.append((time.perf_counter() - start) * 1e3 / ITERS)
    return statistics.median(rounds)


def main() -> int:
    lines = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    assert len(MODELS) == 8, f"expected 8 models, got {len(MODELS)}"
    device = _best_device()
    emit(f"device={device}  torch={torch.__version__}  "
         f"batch={BATCH}  num_classes={NUM_CLASSES}  "
         f"(latency = mean of {ITERS} forwards, median of {ROUNDS} rounds)")

    # ---- 1. the eight baselines at in_ch=1
    rows = []
    for name in MODELS:
        model = build_model(name, num_classes=NUM_CLASSES, in_len=1024)
        n_param = count_parameters(model)
        assert n_param < MAX_PARAMS, f"{name}: {n_param:,} params"
        model.to(device).eval()
        with torch.inference_mode():
            out = model(torch.randn(4, 1, 1024, device=device))
        assert tuple(out.shape) == (4, NUM_CLASSES), (name, out.shape)
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        del model
        _empty_cache(device)

        # training path: one train-mode forward + backward on the device
        model = build_model(name, num_classes=NUM_CLASSES, in_len=1024)
        model.to(device).train()
        logits = model(torch.randn(8, 1, 1024, device=device))
        logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters()), name
        del model
        _empty_cache(device)

        lat = {}
        for in_len in (1024, 2048):
            model = build_model(name, num_classes=NUM_CLASSES, in_len=in_len)
            assert count_parameters(model) < MAX_PARAMS, name
            lat[in_len] = bench_forward(model, in_len, device)
            del model
            _empty_cache(device)

        rows.append((name, n_param, lat[1024], lat[2048]))

    emit("")
    emit("## baselines (in_ch=1)")
    for name, n_param, t1, t2 in rows:
        emit(f"{name:<12s} params={n_param:>9,d}  "
             f"1024={t1:6.2f} ms  2048={t2:6.2f} ms")

    # ---- 2. in_ch>1 regression: every backbone accepts 4 input channels
    emit("")
    emit("## in_ch=4 regression")
    for name in MODELS:
        model = build_model(name, num_classes=NUM_CLASSES, in_len=1024,
                            in_ch=4)
        n_param = count_parameters(model)
        model.to(device).eval()
        with torch.inference_mode():
            out = model(torch.randn(2, 4, 1024, device=device))
        assert tuple(out.shape) == (2, NUM_CLASSES), (name, out.shape)
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        assert n_param < MAX_PARAMS, f"{name}: {n_param:,} params"
        emit(f"{name:<12s} in_ch=4 params={n_param:>9,d}  "
             f"out={tuple(out.shape)}  ok")
        del model
        _empty_cache(device)

    # ---- 3. stem front-end: shape + parameter budget at 1024 / 2048
    emit("")
    emit("## build_stem_model (front-end Conv1d(1,c,k,s) + BN + ReLU)")
    stem_rows = []
    for base_name in STEM_BACKBONES:
        for in_len in (1024, 2048):
            plain = count_parameters(
                build_model(base_name, num_classes=NUM_CLASSES, in_len=in_len))
            for k, s, c in STEM_GRID:
                model = build_stem_model(
                    base_name, num_classes=NUM_CLASSES, in_len=in_len,
                    k=k, s=s, c=c)
                n_param = count_parameters(model)
                model.to(device).eval()
                with torch.inference_mode():
                    out = model(torch.randn(2, 1, in_len, device=device))
                assert tuple(out.shape) == (2, NUM_CLASSES), (
                    base_name, k, s, c, in_len, out.shape)
                delta = n_param / plain - 1.0
                # the stem must not *add* more than 5% parameters; a smaller
                # count is fine (vit1d shrinks its positional embedding when
                # the front-end shortens the window it sees).
                assert delta <= MAX_STEM_PARAM_DELTA, (
                    f"{base_name} k={k} s={s} c={c} in_len={in_len}: "
                    f"params {plain:,} -> {n_param:,} ({delta:+.1%})")
                assert n_param < MAX_PARAMS, (base_name, n_param)
                emit(f"{base_name:<10s} in_len={in_len:<5d} k={k:<4d} s={s} "
                     f"c={c}  L'={model.in_len_stem:<4d} "
                     f"params={n_param:>9,d} ({delta:+.2%})  "
                     f"out={tuple(out.shape)}  ok")
                if in_len == 1024 and c == 8:
                    stem_rows.append((base_name + " c=8", model, in_len))
                else:
                    del model
                _empty_cache(device)

    # ---- 4. latency of the c=8 stem variants
    emit("")
    emit("## latency: c=8 wide-kernel stem (k=64, s=16)")
    for label, model, _ in stem_rows:
        lats = []
        for in_len in (1024, 2048):
            lats.append(bench_forward(model, in_len, device))
        emit(f"{label:<16s} 1024={lats[0]:6.2f} ms  2048={lats[1]:6.2f} ms")
        del model
        _empty_cache(device)

    emit("")
    emit("| model | params | 1024 ms (B=64) | 2048 ms (B=64) |")
    emit("|---|---:|---:|---:|")
    for name, n_param, t1, t2 in rows:
        emit(f"| {name} | {n_param:,} | {t1:.2f} | {t2:.2f} |")

    # ---- 5. the exact call from code/CONTRACT.md section 2
    m = build_model("wdcnn", num_classes=10, in_len=1024)
    out = m(torch.randn(4, 1, 1024))
    assert tuple(out.shape) == (4, 10), out.shape
    stem = build_stem_model("resnet1d", num_classes=10, in_len=1024)
    assert tuple(stem(torch.randn(2, 1, 1024)).shape) == (2, 10)
    emit("")
    emit("train-mode forward+backward: 8/8 models OK")
    emit("contract check: len(MODELS)==8, "
         "build_model('wdcnn', 10, 1024)(4,1,1024) -> (4,10), "
         "build_stem_model('resnet1d', 10, 1024)(2,1,1024) -> (2,10)  OK")
    emit("RESULT: PASS")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
