.PHONY: test test-quant test-kernels test-serialize bench crossover install

install:
	pip install -r requirements.txt

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

test-quant:
	python -m pytest tests/test_quantize.py -v

test-kernels:
	python -m pytest tests/test_kernels.py -v

test-serialize:
	python -m pytest tests/test_serialize.py -v

# ── Kernels sanity check (not pytest) ─────────────────────────────────────────

check-device:
	python liteweight/device.py

check-perf-model:
	python liteweight/perf_model.py

check-triton:
	python tests/test_triton.py

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench:
	python bench/benchmark.py

crossover:
	python bench/crossover.py

# ── Convert ───────────────────────────────────────────────────────────────────
# Usage: make convert MODEL=/path/to/llama OUT=quantized.safetensors BITS=8
MODEL ?= meta-llama/Llama-3.2-1B
OUT   ?= llama-3.2-1b-int8.safetensors
BITS  ?= 8

convert:
	python convert.py -i $(MODEL) -o $(OUT) --bits $(BITS)

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python eval/perplexity.py $(MODEL)
