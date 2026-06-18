# liteweight

Weight-only quantization (INT8/INT4) for Llama-style models, with Triton kernels.

## Setup

```bash
conda activate lw
pip install -r requirements.txt
make test
```

## Quantize a model

```bash
python convert.py -i meta-llama/Llama-3.2-1B -o llama-1b-int8.safetensors --bits 8
python convert.py -i meta-llama/Llama-3.2-1B -o llama-1b-int4.safetensors --bits 4
```

## Generate text

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from liteweight.serialize import load_quantized
from liteweight.generate import generate

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16)
model = load_quantized(model, "llama-1b-int8.safetensors").cuda().eval()

print(generate(model, tokenizer, "The capital of France is", max_new_tokens=50))
```

To quantize on-the-fly without saving:

```python
from liteweight.serialize import swap_linears
model = swap_linears(model, bits=8)
```

## Benchmark

```bash
python bench/benchmark.py   # latency, bandwidth, % peak for each shape/format
python bench/crossover.py   # speedup vs batch size plot → crossover.png
```

## Evaluate quality

```bash
python eval/perplexity.py meta-llama/Llama-3.2-1B
```

## Tests

```bash
make test            # all tests
make test-quant      # quantize round-trip (no GPU needed)
make test-kernels    # Triton kernel correctness (GPU needed)
make test-serialize  # save/load round-trip (no GPU needed)
```
