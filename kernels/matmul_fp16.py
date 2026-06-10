import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_fp16_kernel(
  a_ptr, b_ptr, c_ptr,
  M, N, K,
  stride_am, stride_ak,
  stride_bk, stride_bn,
  stride_cm, stride_cn,
  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
  GROUP_M: tl.constexpr
):
  pass
  # # map 1D program id to (pid_m, pid_n) tile
  # pid = tl.program_id(0)
  # num_pid_m = tl.cdiv(M, BLOCK_M)
  # num_pid_n = tl.cdiv(N, BLOCK_N)
  # num_pid_in_group = GROUP_M * num_pid_n


def matmul_fp16(a: torch.Tensor, b: torch.Tensor):
  """a [M, K] fp16 @ b [K, N] fp16 -> c [M, N] fp16, fp32 accumulate"""

  assert a.dtype == torch.float16 and b.dtype == torch.float16

  M, K = a.shape
  _, N = b.shape

  c = torch.empty((M, N), device=a.device, dtype=torch.float16)

  grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

  _matmul_fp16_kernel[grid](
    a, b, c,
    M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
  )

  return c