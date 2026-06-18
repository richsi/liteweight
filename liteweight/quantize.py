import torch
from torch import Tensor


# ── INT8 ──────────────────────────────────────────────────────────────────────

def quantize_int8(weight: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor]:
    """RTN symmetric group-wise INT8 quantization along the K (last) axis.

    Args:
        weight: fp16 or fp32 [N, K]
        group_size: number of consecutive K elements per scale group

    Returns:
        qweight: int8 [N, K]
        scales:  fp16 [N, K // group_size]
    """
    assert weight.ndim == 2, "weight must be [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    w = weight.float().reshape(N, K // group_size, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True) / 127.0   # [N, K//G, 1]
    scale = scale.clamp(min=1e-8)                         # guard against all-zero groups
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)

    return q.reshape(N, K), scale.reshape(N, K // group_size).half()


def dequantize_int8(qweight: Tensor, scales: Tensor, group_size: int = 128) -> Tensor:
    """Dequantize INT8 weights back to fp16.

    Args:
        qweight: int8 [N, K]
        scales:  fp16 [N, K // group_size]

    Returns:
        fp16 [N, K]
    """
    N, K = qweight.shape
    w = qweight.float().reshape(N, K // group_size, group_size)
    s = scales.float().reshape(N, K // group_size, 1)
    return (w * s).reshape(N, K).half()


def fakequant_matmul(act: Tensor, weight: Tensor, group_size: int = 128) -> Tensor:
    """Correctness oracle: quantize weight → dequantize → fp16 matmul.

    Defines what the INT8 kernel must match.

    Args:
        act:    fp16 [M, K]
        weight: fp16 [N, K]  (nn.Linear convention)

    Returns:
        fp16 [M, N]
    """
    q, s = quantize_int8(weight, group_size)
    w_dq = dequantize_int8(q, s, group_size)
    return act.float().mm(w_dq.float().T).half()


# ── INT4 ──────────────────────────────────────────────────────────────────────

def quantize_int4(weight: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor]:
    """RTN symmetric group-wise INT4 quantization.

    Packing convention:
        low nibble  = element at even K index (2k)
        high nibble = element at odd  K index (2k+1)
        nibbles stored as two's-complement 4-bit signed integers

    Args:
        weight: fp16 or fp32 [N, K]

    Returns:
        qweight: uint8 [N, K // 2]  (packed)
        scales:  fp16  [N, K // group_size]
    """
    assert weight.ndim == 2, "weight must be [N, K]"
    N, K = weight.shape
    assert K % group_size == 0
    assert K % 2 == 0

    w = weight.float().reshape(N, K // group_size, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True) / 7.0    # INT4 range [-8, 7]
    scale = scale.clamp(min=1e-8)
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8).reshape(N, K)

    # Pack: even → low nibble, odd → high nibble
    even = q[:, 0::2]   # [N, K//2]  values in [-8, 7]
    odd  = q[:, 1::2]   # [N, K//2]

    # Two's-complement: mask to 4 bits, then combine
    lo = even.to(torch.uint8) & 0x0F
    hi = (odd.to(torch.uint8) & 0x0F) << 4
    packed = (lo | hi).to(torch.uint8)

    return packed, scale.reshape(N, K // group_size).half()


def dequantize_int4(qweight: Tensor, scales: Tensor, group_size: int = 128) -> Tensor:
    """Dequantize packed INT4 weights back to fp16.

    Args:
        qweight: uint8 [N, K // 2]
        scales:  fp16  [N, K // group_size]

    Returns:
        fp16 [N, K]
    """
    N, K_half = qweight.shape
    K = K_half * 2

    lo_raw = (qweight & 0x0F).to(torch.int8)      # sign-extend low nibble
    hi_raw = (qweight >> 4).to(torch.int8)         # sign-extend high nibble
    # Four-bit sign extension via int8 arithmetic shifts
    lo = ((lo_raw << 4) >> 4)                       # [N, K//2]
    hi = ((hi_raw << 4) >> 4)                       # [N, K//2]

    # Interleave: even positions from lo, odd positions from hi → [N, K]
    unpacked = torch.empty(N, K, dtype=torch.int8, device=qweight.device)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi

    w = unpacked.float().reshape(N, K // group_size, group_size)
    s = scales.float().reshape(N, K // group_size, 1)
    return (w * s).reshape(N, K).half()


def fakequant_matmul_int4(act: Tensor, weight: Tensor, group_size: int = 128) -> Tensor:
    """INT4 correctness oracle."""
    q, s = quantize_int4(weight, group_size)
    w_dq = dequantize_int4(q, s, group_size)
    return act.float().mm(w_dq.float().T).half()
