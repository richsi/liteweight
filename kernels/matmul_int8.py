"""INT8 weight-dequant matmul in Triton — the core bandwidth-saving kernel.

Computes: out[m, n] = sum_k  act[m, k]  *  (qweight[n, k]  *  scales[n, k // G])
All dequantization happens in-register; quantized bytes never go back to HBM.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_int8_kernel(
    a_ptr, w_ptr, s_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn,             # stride over N dim of scales (= K // group_size)
    stride_cm, stride_cn,
    group_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Tile assignment — same grouped ordering as the fp16 kernel
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]  M-dimension
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]  N-dimension

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        # offs_k are ABSOLUTE K indices (used for scale group indexing)
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load activation tile [BLOCK_M, BLOCK_K] fp16
        # Mask: offs_k < K (absolute bound; equivalent to relative < k_remaining)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )  # fp16 [BLOCK_M, BLOCK_K]

        # Load weight tile as [BLOCK_K, BLOCK_N] (transposed layout avoids tl.trans)
        # qweight[N, K] → tile[k, n] = qweight[offs_n[n], offs_k[k]]
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_tile = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )  # int8 [BLOCK_K, BLOCK_N]

        # Scale indices: one per K element — the #1 bug target
        num_groups = tl.cdiv(K, group_size)
        g_offs = offs_k // group_size                   # [BLOCK_K]

        # Load scales [BLOCK_K, BLOCK_N] fp16
        # scales[N, K//G] → s_tile[k, n] = scales[offs_n[n], g_offs[k]]
        s_ptrs = s_ptr + g_offs[:, None] * 1 + offs_n[None, :] * stride_sn
        s_tile = tl.load(
            s_ptrs,
            mask=(g_offs[:, None] < num_groups) & (offs_n[None, :] < N),
            other=0.0,
        )  # fp16 [BLOCK_K, BLOCK_N]

        # Dequantize in-register: int8 → fp16 then scale
        w_fp = w_tile.to(tl.float16) * s_tile          # [BLOCK_K, BLOCK_N] fp16

        # FP32 accumulation: a_tile [BLOCK_M, BLOCK_K] @ w_fp [BLOCK_K, BLOCK_N]
        acc = tl.dot(a_tile, w_fp, acc)                 # [BLOCK_M, BLOCK_N] fp32

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_int8(
    act: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequanting INT8 matmul.

    Args:
        act:      fp16 [M, K]
        qweight:  int8 [N, K]
        scales:   fp16 [N, K // group_size]
        group_size: must match how qweight was quantized

    Returns:
        fp16 [M, N]   (computes act @ qweight.T with in-register dequant)
    """
    assert act.dtype == torch.float16
    assert qweight.dtype == torch.int8
    assert scales.dtype == torch.float16

    act = act.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()

    M, K = act.shape
    N, Kw = qweight.shape
    assert K == Kw
    assert scales.shape == (N, K // group_size)

    out = torch.empty((M, N), device=act.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _matmul_int8_kernel[grid](
        act, qweight, scales, out,
        M, N, K,
        act.stride(0),    act.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),
        out.stride(0),    out.stride(1),
        group_size,
    )
    return out
