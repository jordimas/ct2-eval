import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import statistics

import transformers
from vllm import LLM, SamplingParams

MODEL_ID = "google/gemma-3-1b-it"
GGUF_REPO = "ggml-org/gemma-3-1b-it-GGUF"
GGUF_FILE = "gemma-3-1b-it-Q8_0.gguf"
PROMPT = "What is the capital of France?"
N = 5
CPU_COUNT = os.cpu_count()

batch_sizes = [1, 2, 4, 8]
configs = [
    ("float32", None),
    ("bfloat16", None),
    ("gguf/Q8_0", "gguf"),
]


def build_prompt(tokenizer):
    messages = [{"role": "user", "content": PROMPT}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run(llm, prompt, batch_size):
    params = SamplingParams(max_tokens=256, temperature=0)
    batch = [prompt] * batch_size
    throughput_list = []
    latency_list = []
    for _ in range(N):
        t0 = time.perf_counter()
        outputs = llm.generate(batch, params)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput_list.append(total_tokens / elapsed)
        latency_list.append(elapsed / batch_size)
    return statistics.mean(throughput_list), statistics.mean(latency_list)


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = build_prompt(tokenizer)

    gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

    all_results = []

    for label, quant in configs:
        print(f"\n--- {label} ---")
        print(f"{'batch':>6}  {'throughput tok/s':>18}  {'latency s/req':>14}")
        print("-" * 44)

        if quant == "gguf":
            llm = LLM(model=gguf_path, tokenizer=MODEL_ID, max_model_len=512,
                      kv_cache_memory_bytes=2 * 1024 ** 3)
        else:
            dtype = label
            llm = LLM(model=MODEL_ID, max_model_len=512, dtype=dtype,
                      kv_cache_memory_bytes=2 * 1024 ** 3)

        sanity = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0))
        print(f"  Q: {PROMPT}\n  A: {sanity[0].outputs[0].text}\n")

        for batch_size in batch_sizes:
            throughput, latency = run(llm, prompt, batch_size)
            print(f"{batch_size:>6}  {throughput:>18.1f}  {latency:>14.3f}")
            all_results.append((throughput, latency, label, batch_size))

        del llm

    print()
    best = max(all_results, key=lambda x: x[0])
    print(f"Best throughput: {best[2]}, batch={best[3]}  → {best[0]:.1f} tok/s, {best[1]:.3f} s/req")
    fastest = min(all_results, key=lambda x: x[1])
    print(f"Best latency:    {fastest[2]}, batch={fastest[3]}  → {fastest[0]:.1f} tok/s, {fastest[1]:.3f} s/req")
