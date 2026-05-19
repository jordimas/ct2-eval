"""
Micro-benchmark isolating the decode kernel (single token forward pass).
Generates exactly 1 token per call to remove prefill and sampling overhead,
leaving only the GEMM/attention kernels that dominate autoregressive decode.
"""
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)
import time
import statistics
import ctranslate2
import transformers
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_ID = "google/gemma-3-1b-it"
GGUF_REPO = "ggml-org/gemma-3-1b-it-GGUF"
GGUF_FILE = "gemma-3-1b-it-Q8_0.gguf"
PROMPT = "What is the capital of France?"
WARMUP = 20
N = 100


def build_prompt(tokenizer):
    messages = [{"role": "user", "content": PROMPT}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def bench_ct2():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    ct2_model_dir = f"{MODEL_ID}-ct2-int8"
    if not os.path.isdir(ct2_model_dir):
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(ct2_model_dir, quantization="int8")
    generator = ctranslate2.Generator(ct2_model_dir, device="cpu",
                                      compute_type="int8",
                                      inter_threads=1,
                                      intra_threads=os.cpu_count() // 2)
    prompt = build_prompt(tokenizer)
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False))

    for _ in range(WARMUP):
        generator.generate_batch([input_tokens], max_length=1, include_prompt_in_result=False, beam_size=1)

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        generator.generate_batch([input_tokens], max_length=1, include_prompt_in_result=False, beam_size=1)
        times.append(time.perf_counter() - t0)

    ms = [t * 1000 for t in times]
    print(f"CT2   int8  — avg={statistics.mean(ms):.1f}ms  min={min(ms):.1f}ms  "
          f"max={max(ms):.1f}ms  std={statistics.stdev(ms):.1f}ms  "
          f"→ {1000/statistics.mean(ms):.1f} tok/s")


def bench_llama_cpp():
    model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    llm = Llama(model_path=model_path, n_threads=os.cpu_count() // 2,
                n_ctx=512, verbose=False)
    messages = [{"role": "user", "content": PROMPT}]

    for _ in range(WARMUP):
        llm.create_chat_completion(messages, max_tokens=1, temperature=0)

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        llm.create_chat_completion(messages, max_tokens=1, temperature=0)
        times.append(time.perf_counter() - t0)

    ms = [t * 1000 for t in times]
    print(f"llama Q8_0 — avg={statistics.mean(ms):.1f}ms  min={min(ms):.1f}ms  "
          f"max={max(ms):.1f}ms  std={statistics.stdev(ms):.1f}ms  "
          f"→ {1000/statistics.mean(ms):.1f} tok/s")


if __name__ == "__main__":
    print(f"Decode kernel micro-benchmark ({WARMUP} warmup, {N} timed runs, 1 token each)\n")
    bench_ct2()
    bench_llama_cpp()
