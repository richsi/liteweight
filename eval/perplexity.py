"""Quality + speed headline table: FP16 vs INT8 vs INT4.

perplexity() computes token-level NLL over a text sample.
decode_tokens_per_sec() measures decode throughput (M=1 regime).
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteweight.serialize import swap_linears


WIKITEXT_SAMPLE = (
    "The transformer architecture has become the dominant paradigm in natural language "
    "processing. Attention mechanisms allow the model to weigh the relevance of different "
    "input tokens when producing each output token. This enables the model to capture "
    "long-range dependencies that recurrent architectures struggle with. Modern large "
    "language models are trained on trillions of tokens and contain billions of parameters. "
    "The inference cost of these models is substantial, motivating research into quantization "
    "and other compression techniques. Weight-only quantization reduces the memory footprint "
    "and, at batch size one, the memory bandwidth required to load weights is the bottleneck, "
    "so smaller weights directly translate to faster token generation."
)


def perplexity(model, tokenizer, text: str) -> float:
    """Compute token-level perplexity (exp of mean NLL) over `text`."""
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)

    return float(torch.exp(out.loss).item())


def decode_tokens_per_sec(
    model, tokenizer, prompt: str, n: int = 50
) -> float:
    """Measure decode throughput: tokens generated per second at batch 1."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_id  = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n):
        with torch.no_grad():
            out = model(next_id, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id  = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n / elapsed


def run_table(model_path: str, prompt: str = "The capital of France is"):
    """Print the headline quality/speed table."""
    print(f"Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    configs = [
        ("FP16", None, None),
        ("INT8", 8,    128),
        ("INT4", 4,    128),
    ]

    print(f"\n{'Format':>8}  {'Perplexity':>12}  {'Tokens/sec':>12}")
    print("-" * 40)

    for label, bits, group_size in configs:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model.eval()

        if bits is not None:
            swap_linears(model, group_size=group_size, bits=bits)

        ppl  = perplexity(model, tokenizer, WIKITEXT_SAMPLE)
        tps  = decode_tokens_per_sec(model, tokenizer, prompt)
        print(f"{label:>8}  {ppl:>12.2f}  {tps:>12.1f}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="HF model dir or hub name")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    run_table(args.model_path)
