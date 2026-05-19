import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)
import time
import statistics
import ctranslate2
import transformers

MODEL_ID = "google/gemma-3-1b-it"
PROMPT = "What is the capital of France?"
N = 5
CPU_COUNT = os.cpu_count()

ct2_model_dir = f"{MODEL_ID}-ct2-int8"


def build_prompt(tokenizer):
    messages = [{"role": "user", "content": PROMPT}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run(inter_threads, intra_threads, input_tokens, batch_size):
    generator = ctranslate2.Generator(
        ct2_model_dir, device="cpu",
        compute_type="int8",
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )
    batch = [input_tokens] * batch_size
    throughput_list = []
    latency_list = []
    for _ in range(N):
        t0 = time.perf_counter()
        results = generator.generate_batch(batch, max_length=256, include_prompt_in_result=False, beam_size=1)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(r.sequences_ids[0]) for r in results)
        throughput_list.append(total_tokens / elapsed)
        latency_list.append(elapsed / batch_size)
    return statistics.mean(throughput_list), statistics.mean(latency_list)


if __name__ == "__main__":
    if not os.path.isdir(ct2_model_dir):
        print("Converting model to int8...")
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(ct2_model_dir, quantization="int8")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = build_prompt(tokenizer)
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False))

    thread_counts = sorted(set([1, CPU_COUNT // 4, CPU_COUNT // 2, CPU_COUNT]))
    batch_sizes = [1, 2, 4, 8]

    print(f"CPU count: {CPU_COUNT}  thread counts: {thread_counts}  batch sizes: {batch_sizes}\n")

    all_results = []

    for batch_size in batch_sizes:
        print(f"--- batch_size={batch_size} ---")
        print(f"{'inter':>6}  {'intra':>6}  {'throughput tok/s':>18}  {'latency s/req':>14}")
        print("-" * 52)
        for inter in thread_counts:
            for intra in thread_counts:
                throughput, latency = run(inter, intra, input_tokens, batch_size)
                print(f"{inter:>6}  {intra:>6}  {throughput:>18.1f}  {latency:>14.3f}")
                all_results.append((throughput, latency, batch_size, inter, intra))
        print()

    best = max(all_results, key=lambda x: x[0])
    print(f"Best throughput: batch={best[2]}, inter={best[3]}, intra={best[4]}  "
          f"→ {best[0]:.1f} tok/s total, {best[1]:.3f} s/req")

    fastest = min(all_results, key=lambda x: x[1])
    print(f"Best latency:    batch={fastest[2]}, inter={fastest[3]}, intra={fastest[4]}  "
          f"→ {fastest[0]:.1f} tok/s total, {fastest[1]:.3f} s/req")
