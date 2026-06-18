"""GPU timing harness and memory-bandwidth benchmark.

Never use time.time() — GPU work is async and time.time() measures the launch.
Use torch.cuda.Event with record()/synchronize() for wall-clock GPU time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from liteweight.device import RTX_5090, DeviceSpec
from liteweight.perf_model import Workload, predict


def do_bench(fn, warmup: int = 25, rep: int = 100) -> float:
    """Return median kernel latency in milliseconds.

    Performs warmup iterations, then times `rep` iterations with CUDA events.
    """
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]   # median


def bench_kernel(
    fn,
    *args,
    workload: Workload,
    dev: DeviceSpec = RTX_5090,
) -> dict:
    """Time `fn(*args)` and report bandwidth vs prediction.

    Returns dict with keys:
        latency_ms, achieved_bw_gbs, pct_peak_bw, predicted_ms
    """
    latency_ms = do_bench(lambda: fn(*args))
    latency_s  = latency_ms / 1e3
    achieved_bw_gbs = workload.bytes_moved / latency_s / 1e9

    pred = predict(workload, dev)
    predicted_ms = pred["predicted_time"] * 1e3

    bw = dev.measured_mem_bandwidth_gbs or dev.peak_mem_bandwidth_gbs
    pct_peak = achieved_bw_gbs / bw * 100

    return {
        "latency_ms":     latency_ms,
        "achieved_bw_gbs": achieved_bw_gbs,
        "pct_peak_bw":    pct_peak,
        "predicted_ms":   predicted_ms,
        "bound":          pred["bound"],
        "arithmetic_intensity": pred["arithmetic_intensity"],
    }


def bench_copy_bandwidth(size_mb: int = 512, dev: DeviceSpec = RTX_5090) -> float:
    """Measure achievable memory bandwidth with a pure copy kernel.

    Fills DeviceSpec.measured_mem_bandwidth_gbs (updated in the returned dict).
    """
    n = size_mb * 1024 * 1024 // 2   # fp16 elements
    src = torch.randn(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)

    bytes_moved = src.nbytes + dst.nbytes   # read + write
    latency_ms = do_bench(lambda: dst.copy_(src))
    bw_gbs = bytes_moved / (latency_ms / 1e3) / 1e9
    return bw_gbs


def _print_row(label, r):
    print(
        f"  {label:30s}  "
        f"lat={r['latency_ms']:.3f}ms  "
        f"bw={r['achieved_bw_gbs']:.0f}GB/s  "
        f"({r['pct_peak_bw']:.0f}% peak)  "
        f"pred={r['predicted_ms']:.3f}ms  "
        f"[{r['bound']}]  "
        f"AI={r['arithmetic_intensity']:.1f}"
    )


if __name__ == "__main__":
    from kernels.matmul_fp16 import matmul_fp16
    from kernels.matmul_int8 import matmul_int8
    from kernels.matmul_int4 import matmul_int4
    from liteweight.quantize import quantize_int8, quantize_int4
    from liteweight.perf_model import quant_matmul_workload

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    dev = RTX_5090

    # Measure real memory bandwidth
    bw_measured = bench_copy_bandwidth(dev=dev)
    print(f"Copy-kernel bandwidth: {bw_measured:.0f} GB/s "
          f"({bw_measured/dev.peak_mem_bandwidth_gbs*100:.0f}% of {dev.peak_mem_bandwidth_gbs:.0f})")
    print()

    # Target shape: 1B MLP gate/up projection
    K, N = 2048, 8192
    Ms = [1, 4, 16, 64]

    print(f"{'Shape':42s}  latency  bandwidth   peak%  predicted  bound  AI")
    print("-" * 110)

    for M in Ms:
        act  = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W_fp = torch.randn(N, K, dtype=torch.float16, device="cuda")
        q8, s8 = quantize_int8(W_fp)
        q4, s4 = quantize_int4(W_fp)

        label_fp16 = f"fp16  M={M:4d} K={K} N={N}"
        r_fp16 = bench_kernel(
            matmul_fp16, act, W_fp.T,
            workload=quant_matmul_workload(M, K, N, 16),
            dev=dev,
        )
        _print_row(label_fp16, r_fp16)

        r_int8 = bench_kernel(
            matmul_int8, act, q8, s8,
            workload=quant_matmul_workload(M, K, N, 8),
            dev=dev,
        )
        _print_row(f"int8  M={M:4d} K={K} N={N}", r_int8)

        r_int4 = bench_kernel(
            matmul_int4, act, q4, s4,
            workload=quant_matmul_workload(M, K, N, 4),
            dev=dev,
        )
        _print_row(f"int4  M={M:4d} K={K} N={N}", r_int4)

        speedup_8  = r_fp16["latency_ms"] / r_int8["latency_ms"]
        speedup_4  = r_fp16["latency_ms"] / r_int4["latency_ms"]
        print(f"    → speedup: INT8={speedup_8:.2f}x  INT4={speedup_4:.2f}x\n")
