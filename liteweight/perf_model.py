from dataclasses import dataclass
from liteweight.device import DeviceSpec


@dataclass
class Workload:
    name: str
    flops: int
    bytes_moved: int


def quant_matmul_workload(M: int, K: int, N: int, weight_bits: int) -> Workload:
    flops = 2 * M * K * N
    weight_bytes = K * N * weight_bits // 8
    act_bytes = M * K * 2      # fp16
    out_bytes = M * N * 2      # fp16
    bytes_moved = weight_bytes + act_bytes + out_bytes
    return Workload(
        name=f"matmul_M{M}_K{K}_N{N}_w{weight_bits}",
        flops=flops,
        bytes_moved=bytes_moved,
    )


def predict(w: Workload, dev: DeviceSpec) -> dict:
    ai = w.flops / w.bytes_moved
    t_mem = w.bytes_moved / dev.peak_bw_bytes_s
    t_compute = w.flops / dev.peak_flops
    predicted_time = max(t_mem, t_compute)
    bound = "memory" if t_mem >= t_compute else "compute"
    return {
        "arithmetic_intensity": ai,
        "bound": bound,
        "predicted_time": predicted_time,
        "t_mem": t_mem,
        "t_compute": t_compute,
    }


if __name__ == "__main__":
    from liteweight.device import RTX_5090

    dev = RTX_5090
    # Sanity: M=1 should be memory-bound; sweep until compute-bound
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        w = quant_matmul_workload(M, K=2048, N=8192, weight_bits=16)
        r = predict(w, dev)
        print(f"M={M:4d}  AI={r['arithmetic_intensity']:6.1f}  bound={r['bound']}")
