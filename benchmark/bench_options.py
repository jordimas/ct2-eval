"""
Tests options 1 and 2:
  Option 1 — compute_type sweep: int8, int8_float16, float16
  Option 2 — thread count sweep: vary intra_threads (1..16) with inter_threads=1

Uses a mixed short+long corpus to match the original benchmark conditions.
Runs each config on both short and long questions, reports avg tok/s and drift.
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"  # let CT2 control threading, not OMP

import time
import statistics
import ctranslate2
import transformers

MODEL_ID = "google/gemma-3-1b-it"
MAX_TOKENS = 256
WARMUP = 1
REPS = 2   # repetitions per question for stable measurement
PHYSICAL_CORES = 8
N_QUESTIONS = 5  # subset of corpus per run


def load_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def get_or_convert(quant):
    suffix = quant.replace("_", "-")
    ct2_dir = f"{MODEL_ID}-ct2-{suffix}"
    if not os.path.isdir(ct2_dir):
        print(f"  Converting model for quant={quant}...")
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(
            ct2_dir, quantization=quant if quant != "float32" else None
        )
    return ct2_dir


def make_generator(ct2_dir, compute_type, inter_threads, intra_threads):
    return ctranslate2.Generator(
        ct2_dir, device="cpu",
        compute_type=compute_type,
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )


def tokenize(tok, text):
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.convert_ids_to_tokens(tok.encode(prompt, add_special_tokens=False))


def bench_corpus(gen, tok, questions):
    tps_all = []
    tokens_list = [tokenize(tok, q) for q in questions]
    tokens_list = tokens_list[:N_QUESTIONS]
    # warmup
    for t in tokens_list[:WARMUP]:
        gen.generate_batch([t], max_length=MAX_TOKENS,
                           include_prompt_in_result=False, beam_size=1)
    for t in tokens_list:
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = gen.generate_batch([t], max_length=MAX_TOKENS,
                                   include_prompt_in_result=False, beam_size=1)
            elapsed = time.perf_counter() - t0
            n = len(r[0].sequences_ids[0])
            if n > 0:
                times.append(n / elapsed)
        if times:
            tps_all.append(statistics.median(times))
    return tps_all


def summary(tps_list):
    avg = statistics.mean(tps_list)
    std = statistics.stdev(tps_list) if len(tps_list) > 1 else 0
    first5 = statistics.mean(tps_list[:5])
    last5  = statistics.mean(tps_list[-5:])
    return avg, std, first5, last5, last5 - first5


def section(title):
    print(f"\n{'='*65}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*65}", flush=True)


# ─── Option 1: compute_type sweep ────────────────────────────────────────────

def option1_compute_type(tok, short_q, long_q):
    section("OPTION 1 — compute_type sweep  (intra_threads=8)")

    configs = [
        ("int8",          "int8"),
        ("int8_float16",  "int8_float16"),
        ("float16",       "float16"),
    ]

    print(f"  {'compute_type':<16} {'short avg':>10} {'short std':>10} "
          f"{'long avg':>10} {'long std':>10} {'long drift':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)

    for label, quant in configs:
        ct2_dir = get_or_convert(quant)
        # CT2 uses compute_type as the runtime dtype, model stores int8 weights
        # float32 model can run with float32 compute; int8 model with int8 or int8_float16
        try:
            gen = make_generator(ct2_dir, quant, inter_threads=1, intra_threads=PHYSICAL_CORES)
        except Exception as e:
            print(f"  {label:<16} FAILED: {e}")
            continue

        short_tps = bench_corpus(gen, tok, short_q)
        long_tps  = bench_corpus(gen, tok, long_q)
        del gen

        s_avg, s_std, *_ = summary(short_tps)
        l_avg, l_std, lf5, ll5, drift = summary(long_tps)

        print(f"  {label:<16} {s_avg:>10.1f} {s_std:>10.2f} "
              f"{l_avg:>10.1f} {l_std:>10.2f} {drift:>+11.1f}", flush=True)


# ─── Option 2: intra_threads sweep ───────────────────────────────────────────

def option2_threads(tok, short_q, long_q):
    section("OPTION 2 — intra_threads sweep  (compute_type=int8)")

    ct2_dir = get_or_convert("int8")

    thread_configs = [1, 2, 4, 8, 16]

    print(f"  {'intra_threads':<15} {'short avg':>10} {'short std':>10} "
          f"{'long avg':>10} {'long std':>10} {'long drift':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)

    for n in thread_configs:
        try:
            gen = make_generator(ct2_dir, "int8", inter_threads=1, intra_threads=n)
        except Exception as e:
            print(f"  {n:<15} FAILED: {e}")
            continue

        short_tps = bench_corpus(gen, tok, short_q)
        long_tps  = bench_corpus(gen, tok, long_q)
        del gen

        s_avg, s_std, *_ = summary(short_tps)
        l_avg, l_std, lf5, ll5, drift = summary(long_tps)

        print(f"  {n:<15} {s_avg:>10.1f} {s_std:>10.2f} "
              f"{l_avg:>10.1f} {l_std:>10.2f} {drift:>+11.1f}", flush=True)


# ─── Option 2b: inter_threads sweep (batching angle) ─────────────────────────

def option2b_inter_threads(tok, short_q, long_q):
    section("OPTION 2b — inter_threads split  (total=16 threads, compute_type=int8)")

    ct2_dir = get_or_convert("int8")

    # Keep total threads = 16, vary how they are split
    splits = [
        (1, 16), (1, 8), (2, 8), (4, 4), (8, 2), (16, 1),
    ]

    print(f"  {'inter x intra':<16} {'short avg':>10} {'short std':>10} "
          f"{'long avg':>10} {'long std':>10} {'long drift':>11}", flush=True)
    print(f"  {'-'*65}", flush=True)

    for inter, intra in splits:
        label = f"{inter}x{intra}"
        try:
            gen = make_generator(ct2_dir, "int8", inter_threads=inter, intra_threads=intra)
        except Exception as e:
            print(f"  {label:<16} FAILED: {e}")
            continue

        short_tps = bench_corpus(gen, tok, short_q)
        long_tps  = bench_corpus(gen, tok, long_q)
        del gen

        s_avg, s_std, *_ = summary(short_tps)
        l_avg, l_std, lf5, ll5, drift = summary(long_tps)

        print(f"  {label:<16} {s_avg:>10.1f} {s_std:>10.2f} "
              f"{l_avg:>10.1f} {l_std:>10.2f} {drift:>+11.1f}", flush=True)


if __name__ == "__main__":
    print("Loading tokenizer...", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    short_q = load_lines("corpus_short.txt")
    long_q  = load_lines("corpus_long.txt")
    print(f"Corpora: {len(short_q)} short, {len(long_q)} long questions", flush=True)

    # Pre-convert all models so conversion time doesn't pollute results
    print("Pre-converting models...", flush=True)
    for quant in ["int8", "int8_float16", "float16"]:
        try:
            get_or_convert(quant)
        except Exception as e:
            print(f"  {quant}: {e}")

    option1_compute_type(tok, short_q, long_q)
    option2_threads(tok, short_q, long_q)
    option2b_inter_threads(tok, short_q, long_q)

    print("\nDone.", flush=True)
