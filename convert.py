"""CLI: quantize a HuggingFace Llama model and write to disk.

Usage:
    python convert.py -i <model_dir> -o <out_file> [--bits 8|4] [--group-size 128]
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteweight.serialize import save_quantized


def main():
    parser = argparse.ArgumentParser(description="Quantize a HF model to liteweight format")
    parser.add_argument("-i", "--input", required=True, help="HF model directory or hub name")
    parser.add_argument("-o", "--output", required=True, help="Output .safetensors path")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Weight bits (default: 8)")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size (default: 128)")
    args = parser.parse_args()

    print(f"Loading model from: {args.input}")
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch.float16,
        device_map="cpu",   # quantize on CPU to avoid OOM; move to GPU after
    )
    model.eval()

    # Measure fp16 size
    fp16_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"FP16 weight size: {fp16_bytes / 1e9:.2f} GB")

    print(f"Quantizing to INT{args.bits} (group_size={args.group_size})...")
    save_quantized(
        model,
        out_path=args.output,
        group_size=args.group_size,
        bits=args.bits,
    )

    quant_bytes = Path(args.output).stat().st_size
    print(f"Quantized size:   {quant_bytes / 1e9:.2f} GB")
    print(f"Compression:      {fp16_bytes / quant_bytes:.2f}x")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
