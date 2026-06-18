"""Kernel correctness tests.

Gates:
  matmul_fp16  ≈  torch.matmul
  matmul_int8  ≈  fakequant_matmul (the oracle)
  matmul_int4  ≈  fakequant_matmul_int4
"""

import pytest
import torch
from kernels.matmul_fp16 import matmul_fp16
from kernels.matmul_int8 import matmul_int8
from kernels.matmul_int4 import matmul_int4
from liteweight.quantize import (
    quantize_int8, fakequant_matmul,
    quantize_int4, fakequant_matmul_int4,
)

DEVICE = "cuda"


def skip_no_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── FP16 kernel ───────────────────────────────────────────────────────────────

@skip_no_cuda()
class TestMatmulFP16:
    @pytest.mark.parametrize("M,K,N", [
        (1, 2048, 8192),
        (4, 2048, 8192),
        (16, 2048, 2048),
        (128, 512, 512),
    ])
    def test_close_to_torch(self, M, K, N):
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
        b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)

        ref = torch.matmul(a, b)
        got = matmul_fp16(a, b)

        assert got.shape == (M, N)
        assert got.dtype == torch.float16
        assert torch.allclose(got, ref, atol=1e-2, rtol=1e-3), (
            f"max diff = {(got - ref).abs().max().item():.4f}"
        )

    def test_non_tile_multiple_shape(self):
        a = torch.randn(3, 70, dtype=torch.float16, device=DEVICE)
        b = torch.randn(70, 130, dtype=torch.float16, device=DEVICE)
        ref = torch.matmul(a, b)
        got = matmul_fp16(a, b)
        assert torch.allclose(got, ref, atol=1e-2)


# ── INT8 kernel ───────────────────────────────────────────────────────────────

@skip_no_cuda()
class TestMatmulInt8:
    @pytest.mark.parametrize("M,K,N,G", [
        (1, 2048, 8192, 128),
        (4, 2048, 8192, 128),
        (16, 2048, 2048, 128),
        (1, 2048, 8192, 2048),   # group_size = K (one scale per row)
    ])
    def test_matches_oracle(self, M, K, N, G):
        torch.manual_seed(1)
        act = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
        W   = torch.randn(N, K, dtype=torch.float16, device=DEVICE)

        q, s = quantize_int8(W, group_size=G)
        q, s = q.to(DEVICE), s.to(DEVICE)

        oracle = fakequant_matmul(act, W, group_size=G)
        got    = matmul_int8(act, q, s, group_size=G)

        assert got.shape == (M, N)
        assert got.dtype == torch.float16
        assert torch.allclose(got, oracle, atol=1e-2, rtol=1e-3), (
            f"max diff = {(got - oracle).abs().max().item():.4f}  "
            f"M={M} K={K} N={N} G={G}"
        )

    def test_no_nan(self):
        act = torch.randn(1, 256, dtype=torch.float16, device=DEVICE)
        W   = torch.randn(64, 256, dtype=torch.float16, device=DEVICE)
        q, s = quantize_int8(W)
        out = matmul_int8(act, q.to(DEVICE), s.to(DEVICE))
        assert not out.isnan().any()
        assert not out.isinf().any()


# ── INT4 kernel ───────────────────────────────────────────────────────────────

@skip_no_cuda()
class TestMatmulInt4:
    @pytest.mark.parametrize("M,K,N", [
        (1, 2048, 8192),
        (4, 2048, 8192),
        (16, 2048, 2048),
    ])
    def test_matches_oracle(self, M, K, N):
        torch.manual_seed(2)
        act = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
        W   = torch.randn(N, K, dtype=torch.float16, device=DEVICE)

        q, s = quantize_int4(W)
        q, s = q.to(DEVICE), s.to(DEVICE)

        oracle = fakequant_matmul_int4(act, W)
        got    = matmul_int4(act, q, s)

        assert got.shape == (M, N)
        assert got.dtype == torch.float16
        # INT4 has larger representational error; use looser atol
        assert torch.allclose(got, oracle, atol=0.05, rtol=1e-2), (
            f"max diff = {(got - oracle).abs().max().item():.4f}  M={M} K={K} N={N}"
        )
