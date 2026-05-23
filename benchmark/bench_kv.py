"""
Diagnostic benchmark to isolate CT2's KV-cache scaling weakness.

Two experiments:
  1. Output sweep  – fixed short input (~10 tokens), vary max_tokens [16,32,64,128,192,256]
                     Tests O(T²) concat cost in the generation loop.
  2. Input sweep   – fixed max_tokens=64, vary input length [short → long]
                     Tests whether prompt length alone degrades decode speed.

Run: python bench_kv.py
"""
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)

import time
import statistics
import ctranslate2
import transformers

MODEL_ID = "google/gemma-3-1b-it"
WARMUP = 3
REPS = 5  # repetitions per data point


def load_model(quant="int8"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    ct2_dir = f"{MODEL_ID}-ct2-{quant}"
    if not os.path.isdir(ct2_dir):
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(ct2_dir, quantization=quant)
    generator = ctranslate2.Generator(
        ct2_dir, device="cpu", compute_type=quant,
        inter_threads=1, intra_threads=os.cpu_count() // 2,
    )
    return tokenizer, generator


def tokenize(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False))


def run(generator, tokens, max_tokens, reps=REPS):
    results = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = generator.generate_batch([tokens], max_length=max_tokens,
                                     include_prompt_in_result=False, beam_size=1)
        elapsed = time.perf_counter() - t0
        n = len(r[0].sequences_ids[0])
        if n > 0:
            results.append(n / elapsed)
    return results


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'setting':<30} {'avg':>7} {'p50':>7} {'min':>7} {'max':>7}  tok/s")
    print(f"  {'-'*60}")


def row(label, tps_list):
    avg = statistics.mean(tps_list)
    p50 = statistics.median(tps_list)
    print(f"  {label:<30} {avg:>7.1f} {p50:>7.1f} {min(tps_list):>7.1f} {max(tps_list):>7.1f}")


def experiment_output_sweep(tokenizer, generator):
    """Fix a very short input, sweep max output tokens."""
    header("EXPERIMENT 1: Output length sweep (short input, vary max_tokens)")
    SHORT_Q = "What is 2+2?"
    tokens = tokenize(tokenizer, SHORT_Q)
    input_len = len(tokens)
    print(f"  Input: '{SHORT_Q}'  ({input_len} tokens)\n")

    # warmup
    run(generator, tokens, 64, reps=WARMUP)

    for max_tok in [16, 32, 64, 128, 192, 256]:
        tps = run(generator, tokens, max_tok)
        row(f"max_tokens={max_tok}", tps)


def experiment_input_sweep(tokenizer, generator):
    """Fix short output, sweep input length using progressively longer questions."""
    header("EXPERIMENT 2: Input length sweep (fix max_tokens=64, vary input)")

    questions = [
        ("~10 tok",  "What is 2+2?"),
        ("~20 tok",  "What is the capital of France and what is it known for?"),
        ("~40 tok",  "Explain in one sentence the main idea behind supervised machine learning."),
        ("~80 tok",  "Explain the main differences between supervised learning and unsupervised learning in machine learning, focusing on how each approach uses data."),
        ("~150 tok", "You are an expert in machine learning. Describe in technical detail the differences between supervised, unsupervised, and reinforcement learning. Include examples of algorithms for each paradigm and explain when you would choose one over another in a real-world project."),
        ("~300 tok", "You are a professor of computer science. Write a comprehensive explanation of transformer neural networks, covering: (1) the attention mechanism and why it was introduced, (2) the encoder and decoder architecture, (3) positional encodings and why they are necessary, (4) multi-head attention and its advantages, (5) applications in NLP and beyond such as vision transformers. Explain each point clearly for a graduate student audience."),
    ]

    # warmup
    tokens0 = tokenize(tokenizer, questions[0][1])
    run(generator, tokens0, 64, reps=WARMUP)

    for label, q in questions:
        tokens = tokenize(tokenizer, q)
        n_in = len(tokens)
        tps = run(generator, tokens, max_tokens=64)
        row(f"{label} (in={n_in})", tps)


def experiment_output_sweep_incremental(tokenizer, generator):
    """Same as exp 1 but report total time and per-token time to surface quadratic growth."""
    header("EXPERIMENT 3: Per-token cost at different output lengths")
    SHORT_Q = "What is 2+2?"
    tokens = tokenize(tokenizer, SHORT_Q)

    # warmup
    run(generator, tokens, 64, reps=WARMUP)

    print(f"  {'max_tokens':<15} {'avg_total_s':>12} {'avg_tok/s':>10} {'est_ms/tok':>12}")
    print(f"  {'-'*55}")
    for max_tok in [16, 32, 64, 96, 128, 160, 192, 224, 256]:
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = generator.generate_batch([tokens], max_length=max_tok,
                                         include_prompt_in_result=False, beam_size=1)
            elapsed = time.perf_counter() - t0
            n = len(r[0].sequences_ids[0])
            times.append((elapsed, n))
        avg_t = statistics.mean(t for t, _ in times)
        avg_n = statistics.mean(n for _, n in times)
        avg_tps = avg_n / avg_t if avg_t > 0 else 0
        ms_per_tok = (avg_t / avg_n * 1000) if avg_n > 0 else 0
        print(f"  {max_tok:<15} {avg_t:>12.3f} {avg_tps:>10.1f} {ms_per_tok:>12.2f}")


if __name__ == "__main__":
    print("Loading model...")
    tokenizer, generator = load_model("int8")
    print("Model loaded.")

    experiment_output_sweep(tokenizer, generator)
    experiment_input_sweep(tokenizer, generator)
    experiment_output_sweep_incremental(tokenizer, generator)
