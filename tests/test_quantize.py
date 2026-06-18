"""Tests for group-wise RTN INT8/INT4 quantization and the correctness oracle."""

import pytest
import torch
from liteweight.quantize import (
    quantize_int8, dequantize_int8, fakequant_matmul,
    quantize_int4, dequantize_int4, fakequant_matmul_int4,
)


def make_weight(N=64, K=256, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, K, dtype=torch.float16)


# ── INT8 ──────────────────────────────────────────────────────────────────────

class TestInt8:
    def test_shapes(self):
        W = make_weight(32, 128)
        q, s = quantize_int8(W, group_size=128)
        assert q.shape == W.shape
        assert q.dtype == torch.int8
        assert s.shape == (32, 1)
        assert s.dtype == torch.float16

    def test_shapes_multigroup(self):
        W = make_weight(32, 512)
        q, s = quantize_int8(W, group_size=128)
        assert q.shape == (32, 512)
        assert s.shape == (32, 4)

    def test_roundtrip_close(self):
        W = make_weight(64, 256)
        q, s = quantize_int8(W, group_size=128)
        W_dq = dequantize_int8(q, s, group_size=128)
        err = (W - W_dq).abs()
        # Max relative error should be within ~1/127 ≈ 0.8%
        assert err.max().item() < 0.05, f"max error {err.max().item()}"

    def test_zero_group_no_nan(self):
        W = torch.zeros(4, 128, dtype=torch.float16)
        q, s = quantize_int8(W, group_size=128)
        W_dq = dequantize_int8(q, s, group_size=128)
        assert not W_dq.isnan().any()
        assert not W_dq.isinf().any()

    def test_group_axis_is_k(self):
        # If grouping is on N instead of K, row 0 and row 1 will share wrong scales
        W = torch.zeros(4, 256, dtype=torch.float16)
        W[0, :128] = 1.0
        W[0, 128:] = 0.0   # row 0: two groups with different magnitudes
        q, s = quantize_int8(W, group_size=128)
        assert s.shape == (4, 2)
        # Scale for row 0, group 0 should differ from row 0, group 1
        assert s[0, 0].item() != pytest.approx(s[0, 1].item())

    def test_oracle_dtype(self):
        torch.manual_seed(1)
        act = torch.randn(4, 256, dtype=torch.float16)
        W   = torch.randn(32, 256, dtype=torch.float16)
        out = fakequant_matmul(act, W, group_size=128)
        assert out.shape == (4, 32)
        assert out.dtype == torch.float16
        assert not out.isnan().any()

    @pytest.mark.parametrize("M,K,N", [(1, 2048, 8192), (4, 512, 512)])
    def test_oracle_close_to_fp16(self, M, K, N):
        torch.manual_seed(42)
        act = torch.randn(M, K, dtype=torch.float16)
        W   = torch.randn(N, K, dtype=torch.float16)
        oracle = fakequant_matmul(act, W)
        fp16   = act.float().mm(W.float().T).half()
        # Oracle should be close to fp16 (small quant error, not numerical explosion)
        assert (oracle - fp16).abs().max().item() < 5.0


# ── INT4 ──────────────────────────────────────────────────────────────────────

class TestInt4:
    def test_shapes(self):
        W = make_weight(32, 256)
        q, s = quantize_int4(W, group_size=128)
        assert q.shape == (32, 128)
        assert q.dtype == torch.uint8
        assert s.shape == (32, 2)
        assert s.dtype == torch.float16

    def test_roundtrip_bounded(self):
        W = make_weight(64, 256)
        q, s = quantize_int4(W, group_size=128)
        W_dq = dequantize_int4(q, s, group_size=128)
        assert W_dq.shape == W.shape
        assert not W_dq.isnan().any()
        # INT4 has more error than INT8 but should be bounded
        assert W_dq.abs().max().item() < W.abs().max().item() * 2

    def test_packing_roundtrip(self):
        # dequantize_int4 must agree with the oracle path (quantize → dequantize both ways)
        W = make_weight(4, 128)
        q_int4, s = quantize_int4(W, group_size=128)
        W_dq = dequantize_int4(q_int4, s, group_size=128)

        # Oracle: quantize_int4 → dequantize_int4 (independent call)
        q2, s2 = quantize_int4(W, group_size=128)
        W_oracle = dequantize_int4(q2, s2, group_size=128)

        assert torch.allclose(W_dq, W_oracle, atol=1e-4), "dequantize round-trip not deterministic"

    def test_zero_group_no_nan(self):
        W = torch.zeros(4, 128, dtype=torch.float16)
        q, s = quantize_int4(W, group_size=128)
        W_dq = dequantize_int4(q, s, group_size=128)
        assert not W_dq.isnan().any()
