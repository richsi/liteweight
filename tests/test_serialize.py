"""Serialize round-trip test: save → load in a fresh state → verify weights match."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from liteweight.quantize import quantize_int8, dequantize_int8
from liteweight.quantlinear import QuantLinear
from liteweight.serialize import save_quantized, load_quantized, swap_linears


class TinyModel(nn.Module):
    """Minimal two-layer model for serialize testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.lm_head = nn.Linear(256, 128, bias=False)  # should be skipped

    def forward(self, x):
        return self.lm_head(self.fc2(torch.relu(self.fc1(x))))


def make_model():
    torch.manual_seed(42)
    m = TinyModel()
    m.eval()
    return m


class TestRoundTrip:
    def test_basic_roundtrip(self):
        model = make_model()
        # Capture weight before save_quantized swaps nn.Linear → QuantLinear in-place
        W_orig = model.fc1.weight.data.half().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_quantized(model, path, group_size=128, bits=8)

            # Load into a fresh model instance (simulates a new process)
            fresh = make_model()
            fresh = load_quantized(fresh, path)

            # fc1 and fc2 should be QuantLinear; lm_head should remain nn.Linear
            assert isinstance(fresh.fc1, QuantLinear)
            assert isinstance(fresh.fc2, QuantLinear)
            assert isinstance(fresh.lm_head, nn.Linear)

            # Dequantize the loaded layer and compare to a fresh quantization of the original weight
            q_ref, s_ref = quantize_int8(W_orig, group_size=128)
            W_ref = dequantize_int8(q_ref, s_ref)

            W_loaded = dequantize_int8(fresh.fc1.qweight, fresh.fc1.scales)
            assert torch.allclose(W_loaded, W_ref, atol=1e-4), (
                f"Dequantized weights differ after round-trip: "
                f"max diff = {(W_loaded - W_ref).abs().max().item()}"
            )

    def test_meta_json_content(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_quantized(model, path, group_size=128, bits=8)

            meta = json.loads(Path(str(path) + ".meta.json").read_text())
            assert meta["format_version"] == 1
            assert meta["bits"] == 8
            assert meta["group_size"] == 128
            assert "lm_head" in meta["skip"]
            assert "fc1" in meta["layers"]
            assert "fc2" in meta["layers"]
            assert "lm_head" not in meta["layers"]

    def test_bias_preserved(self):
        model = make_model()
        bias_orig = model.fc2.bias.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_quantized(model, path, group_size=128, bits=8)

            fresh = make_model()
            fresh = load_quantized(fresh, path)

            assert torch.allclose(fresh.fc2.bias, bias_orig.half(), atol=1e-4)

    def test_unknown_format_version_raises(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_quantized(model, path)

            meta_path = Path(str(path) + ".meta.json")
            meta = json.loads(meta_path.read_text())
            meta["format_version"] = 99
            meta_path.write_text(json.dumps(meta))

            with pytest.raises(ValueError, match="Unknown format version"):
                load_quantized(make_model(), path)

    @pytest.mark.parametrize("bits", [8, 4])
    def test_int4_roundtrip(self, bits):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            save_quantized(model, path, group_size=128, bits=bits)

            fresh = make_model()
            fresh = load_quantized(fresh, path)
            assert isinstance(fresh.fc1, QuantLinear)
            assert fresh.fc1.bits == bits
