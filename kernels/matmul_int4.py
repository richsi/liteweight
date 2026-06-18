"""INT4 weight-dequant matmul in Triton (Phase 5).

qweight is packed uint8 [N, K//2]: low nibble = element at even K, high nibble = odd K.
Nibbles are 4-bit two's-complement signed integers in [-8, 7].
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K_HALF': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 256, 'BLOCK_K_HALF': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K_HALF': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K_HALF': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 128, 'BLOCK_K_HALF': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  64, 'BLOCK_K_HALF': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 256, 'BLOCK_K_HALF': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_int4_kernel(
    a_ptr, w_ptr, s_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk_half,   # w stride over packed dim (K//2)
    stride_sn,
    stride_cm, stride_cn,
    group_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_HALF: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Tile assignment
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_half = K // 2
    num_groups = tl.cdiv(K, group_size)

    for k0_half in range(0, tl.cdiv(K_half, BLOCK_K_HALF)):
        # Absolute packed-byte indices (0..K//2-1)
        offs_k_half = k0_half * BLOCK_K_HALF + tl.arange(0, BLOCK_K_HALF)  # [BKH]

        # Actual K indices: even positions = 2*j, odd = 2*j+1
        k_even = offs_k_half * 2      # [BKH]
        k_odd  = offs_k_half * 2 + 1  # [BKH]

        # Load packed bytes as [BLOCK_K_HALF, BLOCK_N] to avoid tl.trans
        # qweight[N, K//2] → tile[k_half, n] = qweight[offs_n[n], offs_k_half[k_half]]
        w_ptrs = w_ptr + offs_k_half[:, None] * stride_wk_half + offs_n[None, :] * stride_wn
        packed = tl.load(
            w_ptrs,
            mask=(offs_k_half[:, None] < K_half) & (offs_n[None, :] < N),
            other=0,
        )  # uint8 [BLOCK_K_HALF, BLOCK_N]

        # Unpack nibbles with sign extension
        lo_raw = (packed & 0x0F).to(tl.int8)
        hi_raw = (packed >> 4).to(tl.int8)
        lo = (lo_raw << 4) >> 4   # sign-extend 4-bit low nibble  [BKH, BLOCK_N]
        hi = (hi_raw << 4) >> 4   # sign-extend 4-bit high nibble [BKH, BLOCK_N]

        # Scale indices for even/odd K positions (same group if group_size > 1)
        g_even = k_even // group_size                   # [BKH]
        g_odd  = k_odd  // group_size

        # Scales as [BLOCK_K_HALF, BLOCK_N]
        s_even = tl.load(
            s_ptr + g_even[:, None] * 1 + offs_n[None, :] * stride_sn,
            mask=(g_even[:, None] < num_groups) & (offs_n[None, :] < N),
            other=0.0,
        )  # [BKH, BLOCK_N] fp16
        s_odd = tl.load(
            s_ptr + g_odd[:, None] * 1 + offs_n[None, :] * stride_sn,
            mask=(g_odd[:, None] < num_groups) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Dequant in-register
        w_even = lo.to(tl.float16) * s_even  # [BKH, BLOCK_N]
        w_odd  = hi.to(tl.float16) * s_odd

        # Load activation for even and odd K positions as [BLOCK_M, BLOCK_K_HALF]
        a_even = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_even[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BKH] fp16
        a_odd = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_odd[None, :] < K),
            other=0.0,
        )

        # Two dot products covering even+odd K positions:
        # a_even [BLOCK_M, BKH] @ w_even [BKH, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc = tl.dot(a_even, w_even, acc)
        acc = tl.dot(a_odd,  w_odd,  acc)

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_int4(
    act: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequanting INT4 matmul.

    Args:
        act:      fp16 [M, K]
        qweight:  uint8 [N, K // 2]  (nibble-packed)
        scales:   fp16  [N, K // group_size]

    Returns:
        fp16 [M, N]
    """
    assert act.dtype == torch.float16
    assert qweight.dtype == torch.uint8
    assert scales.dtype == torch.float16

    act = act.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()

    M, K = act.shape
    N, K_half = qweight.shape
    assert K == K_half * 2
    assert scales.shape == (N, K // group_size)

    out = torch.empty((M, N), device=act.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    _matmul_int4_kernel[grid](
        act, qweight, scales, out,
        M, N, K,
        act.stride(0),     act.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),
        out.stride(0),     out.stride(1),
        group_size,
    )
    return out
