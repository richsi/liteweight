"""QuantLinear: nn.Linear drop-in whose forward dispatches to the quant kernel.

The encapsulation boundary: callers see fp16 in / fp16 out.
Quantized format (int8 or packed int4) never leaks above this module.
"""

import torch
import torch.nn as nn
from torch import Tensor

from liteweight.quantize import quantize_int8, quantize_int4
from kernels.matmul_int8 import matmul_int8
from kernels.matmul_int4 import matmul_int4


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
        bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits

        assert bits in (4, 8), f"bits must be 4 or 8, got {bits}"
        assert in_features % group_size == 0

        if bits == 8:
            self.register_buffer(
                "qweight", torch.zeros(out_features, in_features, dtype=torch.int8)
            )
        else:  # int4
            self.register_buffer(
                "qweight",
                torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            )

        self.register_buffer(
            "scales",
            torch.zeros(out_features, in_features // group_size, dtype=torch.float16),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features).to(torch.float16)  # [M, K]

        if self.bits == 8:
            out = matmul_int8(x, self.qweight, self.scales, self.group_size)
        else:
            out = matmul_int4(x, self.qweight, self.scales, self.group_size)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 128, bits: int = 8
    ) -> "QuantLinear":
        ql = cls(
            linear.in_features,
            linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
            bits=bits,
        )
        w = linear.weight.data.float().half()  # ensure fp16
        if bits == 8:
            q, s = quantize_int8(w, group_size)
        else:
            q, s = quantize_int4(w, group_size)
        ql.qweight.copy_(q)
        ql.scales.copy_(s)
        if linear.bias is not None:
            ql.bias.copy_(linear.bias.data.half())
        return ql

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )
