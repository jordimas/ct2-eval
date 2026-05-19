# ct2 vs vLLM Benchmark

Benchmarks CTranslate2 and vLLM inference on `google/gemma-3-1b-it` by running "What is the capital of France?" 10 times and reporting tokens/s (avg, min, max, std).

## Setup

```bash
uv sync
```

Convert the model to CTranslate2 format (one-time):

```bash
uv run ct2-transformers-converter --model google/gemma-3-1b-it \
    --output_dir google/gemma-3-1b-it-ct2 --quantization int8 --trust_remote_code
```

## Usage

```bash
# Both backends on CPU (default)
uv run python benchmark.py

# CUDA
uv run python benchmark.py --device cuda

# Single backend
uv run python benchmark.py --backend ct2
uv run python benchmark.py --backend vllm
```

## Options

| Flag | Values | Default |
|------|--------|---------|
| `--device` | `cpu`, `cuda` | `cpu` |
| `--backend` | `ct2`, `vllm`, `both` | `both` |
