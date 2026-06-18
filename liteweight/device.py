from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    peak_mem_bandwidth_gbs: float      # theoretical HBM/GDDR7 bandwidth (GB/s)
    peak_bf16_tflops: float            # dense BF16 tensor-core TFLOPs (no sparsity)
    sm_count: int
    shared_mem_per_sm_kb: float
    measured_mem_bandwidth_gbs: float | None = None  # filled from copy-kernel bench

    @property
    def peak_bw_bytes_s(self) -> float:
        return self.peak_mem_bandwidth_gbs * 1e9

    @property
    def peak_flops(self) -> float:
        return self.peak_bf16_tflops * 1e12

    @property
    def ridge_point(self) -> float:
        """FLOPs/byte where memory-bound flips to compute-bound."""
        return self.peak_flops / self.peak_bw_bytes_s

    @property
    def effective_bw_bytes_s(self) -> float:
        bw = self.measured_mem_bandwidth_gbs or self.peak_mem_bandwidth_gbs
        return bw * 1e9


# RTX 5090 (Blackwell sm_120): 1792 GB/s GDDR7, 209.5 TFLOPs BF16 dense
# ridge_point ≈ 117 FLOPs/byte (the spec's 150-600 range assumes different peak_bf16;
# the actual dense BF16 figure is ~209 TFLOPs on this card)
RTX_5090 = DeviceSpec(
    name="RTX 5090 (Blackwell, sm_120)",
    peak_mem_bandwidth_gbs=1792.0,
    peak_bf16_tflops=209.5,
    sm_count=170,
    shared_mem_per_sm_kb=128.0,
)


if __name__ == "__main__":
    dev = RTX_5090
    print(f"Device          : {dev.name}")
    print(f"SM count        : {dev.sm_count}")
    print(f"Shared mem/SM   : {dev.shared_mem_per_sm_kb} KB")
    print(f"BF16 TFLOPs     : {dev.peak_bf16_tflops}")
    print(f"Memory BW       : {dev.peak_mem_bandwidth_gbs} GB/s")
    print(f"Ridge point     : {dev.ridge_point:.1f} FLOPs/byte")
