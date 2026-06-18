"""Speedup vs batch plot — the headline analysis.

Sweeps M ∈ {1,2,4,8,16,32,64} for FP16 vs INT8 vs INT4 on the 1B MLP shape,
plots measured speedup and overlays the roofline prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt

from liteweight.device import RTX_5090
from liteweight.perf_model import quant_matmul_workload, predict
from kernels.matmul_fp16 import matmul_fp16
from kernels.matmul_int8 import matmul_int8
from kernels.matmul_int4 import matmul_int4
from liteweight.quantize import quantize_int8, quantize_int4
from bench.benchmark import do_bench


def run_sweep(K: int = 2048, N: int = 8192, Ms=None):
    if Ms is None:
        Ms = [1, 2, 4, 8, 16, 32, 64]

    dev = RTX_5090
    results = []

    for M in Ms:
        act  = torch.randn(M, K, dtype=torch.float16, device="cuda")
        W_fp = torch.randn(N, K, dtype=torch.float16, device="cuda")
        q8, s8 = quantize_int8(W_fp)
        q4, s4 = quantize_int4(W_fp)

        t_fp16 = do_bench(lambda: matmul_fp16(act, W_fp.T))
        t_int8 = do_bench(lambda: matmul_int8(act, q8, s8))
        t_int4 = do_bench(lambda: matmul_int4(act, q4, s4))

        # Roofline predictions
        w16 = quant_matmul_workload(M, K, N, 16)
        w8  = quant_matmul_workload(M, K, N, 8)
        w4  = quant_matmul_workload(M, K, N, 4)
        pred16 = predict(w16, dev)["predicted_time"] * 1e3
        pred8  = predict(w8,  dev)["predicted_time"] * 1e3
        pred4  = predict(w4,  dev)["predicted_time"] * 1e3

        results.append({
            "M": M,
            "t_fp16": t_fp16, "t_int8": t_int8, "t_int4": t_int4,
            "speedup_int8": t_fp16 / t_int8,
            "speedup_int4": t_fp16 / t_int4,
            "pred_speedup_int8": pred16 / pred8,
            "pred_speedup_int4": pred16 / pred4,
        })

    return results


def plot(results, out_path="crossover.png"):
    Ms             = [r["M"]                for r in results]
    sp_int8        = [r["speedup_int8"]     for r in results]
    sp_int4        = [r["speedup_int4"]     for r in results]
    pred_sp_int8   = [r["pred_speedup_int8"] for r in results]
    pred_sp_int4   = [r["pred_speedup_int4"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(Ms, sp_int8, "b-o",  label="INT8 measured",   linewidth=2)
    ax.semilogx(Ms, sp_int4, "r-o",  label="INT4 measured",   linewidth=2)
    ax.semilogx(Ms, pred_sp_int8, "b--", label="INT8 predicted", linewidth=1, alpha=0.7)
    ax.semilogx(Ms, pred_sp_int4, "r--", label="INT4 predicted", linewidth=1, alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(2.0, color="blue", linestyle=":",  linewidth=0.8, alpha=0.4)
    ax.axhline(4.0, color="red",  linestyle=":",  linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Batch size M (log scale)")
    ax.set_ylabel("Speedup over FP16")
    ax.set_title("INT8/INT4 speedup vs batch — Llama-3.2-1B MLP shape (K=2048, N=8192)")
    ax.legend()
    ax.set_xticks(Ms)
    ax.set_xticklabels(Ms)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    return fig


def print_table(results):
    print(f"\n{'M':>6}  {'INT8 speedup':>14}  {'INT4 speedup':>14}  "
          f"{'pred INT8':>11}  {'pred INT4':>11}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['M']:>6}  {r['speedup_int8']:>14.2f}x  {r['speedup_int4']:>14.2f}x  "
            f"{r['pred_speedup_int8']:>11.2f}x  {r['pred_speedup_int4']:>11.2f}x"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    results = run_sweep()
    print_table(results)
    plot(results)
