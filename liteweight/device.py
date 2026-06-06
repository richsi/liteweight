from dataclasses import dataclass

@dataclass(frozen=True)
class DeviceSpec:
    name: str

    # Theoretical peaks 
    mem_bandwidth_gb_s: float      
    bf16_tflops: float             # DENSE tensor-core TFLOPs, not sparse
    sm_count: int
    smem_per_sm_kb: float

    measured_mem_bandwidth_gb_s: float | None = None

    @property
    def peak_bw_bytes_s(self) -> float:
        return self.mem_bandwidth_gb_s * 1e9

    @property
    def peak_flops(self) -> float:
        return self.bf16_tflops * 1e12

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte where memory-bound flips to compute-bound."""
        return self.peak_flops / self.peak_bw_bytes_s

    @property
    def effective_bw_bytes_s(self) -> float:
        """Use measured BW if available, else theoretical."""
        bw = self.measured_mem_bandwidth_gb_s or self.mem_bandwidth_gb_s
        return bw * 1e9



if __name__ == "__main__":

  RTX_5090 = DeviceSpec(
      name="RTX 5090 (Blackwell, sm_120)",
      mem_bandwidth_gb_s=1792,   
      bf16_tflops=209.5,
      sm_count=170,
      smem_per_sm_kb=128
  )

  print(f"""
    ==================================================
    Device: {RTX_5090.name}
    ==================================================
    Hardware:
      SM Count              : {RTX_5090.sm_count}
      Shared Mem per SM     : {RTX_5090.smem_per_sm_kb} KB

    Compute:
      BF16 TFLOPs (Dense)   : {RTX_5090.bf16_tflops:,.1f}
      Peak FLOPs            : {RTX_5090.peak_flops:,.0f}

    Memory:
      Bandwidth (Theoretical): {RTX_5090.mem_bandwidth_gb_s:,.1f} GB/s
      Peak Bandwidth         : {RTX_5090.peak_bw_bytes_s:,.0f} bytes/s
      Effective Bandwidth    : {RTX_5090.effective_bw_bytes_s:,.0f} bytes/s

    Performance:
      Ridge Point            : {RTX_5090.ridge_point:.2f} FLOPs/byte
    ==================================================
    """)