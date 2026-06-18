"""Save/load quantized model weights and swap nn.Linear → QuantLinear."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from liteweight.quantlinear import QuantLinear


def swap_linears(
    model: nn.Module,
    group_size: int = 128,
    bits: int = 8,
    skip: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """Recursively replace nn.Linear → QuantLinear (quantizes live weights in place)."""
    _swap_children(model, group_size, bits, skip, quantize=True)
    return model


def swap_linears_empty(
    model: nn.Module,
    group_size: int = 128,
    bits: int = 8,
    skip: tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """Recursively replace nn.Linear → QuantLinear with zero-filled buffers.

    Used before load_quantized to get the right dtype/shape without quantizing.
    """
    _swap_children(model, group_size, bits, skip, quantize=False)
    return model


def _swap_children(
    module: nn.Module,
    group_size: int,
    bits: int,
    skip: tuple[str, ...],
    quantize: bool,
) -> None:
    # Use named_children() so we hold a reference to the parent for setattr.
    # Never mutate via named_modules() (flat iteration — no parent reference).
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in skip:
            if quantize:
                setattr(module, name, QuantLinear.from_linear(child, group_size, bits))
            else:
                setattr(
                    module,
                    name,
                    QuantLinear(
                        child.in_features,
                        child.out_features,
                        group_size=group_size,
                        bits=bits,
                        bias=child.bias is not None,
                    ),
                )
        else:
            _swap_children(child, group_size, bits, skip, quantize)


def save_quantized(
    model: nn.Module,
    out_path: str | Path,
    group_size: int = 128,
    bits: int = 8,
    skip: tuple[str, ...] = ("lm_head",),
) -> None:
    """Quantize (if needed) and save weights to <out_path>.safetensors + .meta.json.

    Accepts both raw nn.Linear models and already-swapped models.  If the model
    has no QuantLinear modules yet, swap_linears is called in-place first.
    """
    out_path = Path(out_path)

    # Quantize in-place if the model hasn't been swapped yet
    if not any(isinstance(m, QuantLinear) for m in model.modules()):
        model = swap_linears(model, group_size=group_size, bits=bits, skip=skip)

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            tensors[f"{name}.qweight"] = module.qweight.cpu()
            tensors[f"{name}.scales"] = module.scales.cpu()
            if module.bias is not None:
                tensors[f"{name}.bias"] = module.bias.cpu()
            layers[name] = {
                "in": module.in_features,
                "out": module.out_features,
                "has_bias": module.bias is not None,
            }

    save_file(tensors, out_path)

    meta = {
        "format_version": 1,
        "bits": bits,
        "group_size": group_size,
        "skip": list(skip),
        "layers": layers,
    }
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))


def load_quantized(model: nn.Module, path: str | Path) -> nn.Module:
    """Load a quantized model saved by save_quantized.

    Rebuilds the module tree from metadata, then copies saved tensors by name.
    """
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())

    if meta["format_version"] != 1:
        raise ValueError(f"Unknown format version: {meta['format_version']}")

    model = swap_linears_empty(
        model,
        group_size=meta["group_size"],
        bits=meta["bits"],
        skip=tuple(meta["skip"]),
    )

    tensors = load_file(path)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.qweight.copy_(tensors[f"{name}.qweight"])
            module.scales.copy_(tensors[f"{name}.scales"])
            if module.bias is not None:
                module.bias.copy_(tensors[f"{name}.bias"])

    return model
