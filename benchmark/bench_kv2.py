"""
Follow-up diagnostic: does CT2 degrade progressively across sequential requests?

The original report showed long-context CT2 dropping from 27.1 to 21.3 tok/s
across 25 consecutive requests while llama.cpp stayed flat.

This script runs N long requests in sequence and plots tok/s over time.
Also tests whether it's input length OR output length that causes inter-request drift.
"""
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // 2)

import time
import statistics
import ctranslate2
import transformers
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_ID  = "google/gemma-3-1b-it"
GGUF_REPO = "ggml-org/gemma-3-1b-it-GGUF"
GGUF_FILE = "gemma-3-1b-it-Q8_0.gguf"
MAX_TOKENS = 256
N_REQUESTS = 25


def load_ct2(quant="int8"):
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    ct2_dir = f"{MODEL_ID}-ct2-{quant}"
    if not os.path.isdir(ct2_dir):
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(ct2_dir, quantization=quant)
    gen = ctranslate2.Generator(ct2_dir, device="cpu", compute_type=quant,
                                inter_threads=1, intra_threads=os.cpu_count() // 2)
    return tok, gen


def load_llama():
    path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    llm = Llama(model_path=path, n_threads=os.cpu_count() // 2,
                n_ctx=512, verbose=False)
    return llm


def ct2_tokenize(tok, text):
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.convert_ids_to_tokens(tok.encode(prompt, add_special_tokens=False))


def run_section(title, tps_list):
    print(f"\n  {'req':>4}  {'tok/s':>7}")
    print(f"  {'-'*15}")
    for i, tps in enumerate(tps_list):
        print(f"  {i+1:>4}  {tps:>7.1f}")
    avg = statistics.mean(tps_list)
    std = statistics.stdev(tps_list) if len(tps_list) > 1 else 0
    first5 = statistics.mean(tps_list[:5])
    last5  = statistics.mean(tps_list[-5:])
    print(f"  avg={avg:.1f}  std={std:.1f}  first5={first5:.1f}  last5={last5:.1f}  drift={last5-first5:+.1f}")


def experiment_ct2_sequential(ct2_tok, ct2_gen, questions, label):
    print(f"\n{'='*55}")
    print(f"  CT2 — {label}")
    print(f"{'='*55}")
    # warmup
    w = ct2_tokenize(ct2_tok, questions[0])
    ct2_gen.generate_batch([w], max_length=MAX_TOKENS, include_prompt_in_result=False, beam_size=1)

    tps_list = []
    for q in questions:
        tokens = ct2_tokenize(ct2_tok, q)
        t0 = time.perf_counter()
        r = ct2_gen.generate_batch([tokens], max_length=MAX_TOKENS,
                                   include_prompt_in_result=False, beam_size=1)
        elapsed = time.perf_counter() - t0
        n = len(r[0].sequences_ids[0])
        tps_list.append(n / elapsed if elapsed > 0 else 0)
    run_section(label, tps_list)


def experiment_llama_sequential(llm, questions, label):
    print(f"\n{'='*55}")
    print(f"  llama.cpp — {label}")
    print(f"{'='*55}")
    # warmup
    llm.create_chat_completion([{"role":"user","content":questions[0]}],
                               max_tokens=MAX_TOKENS, temperature=0)
    tps_list = []
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        t0 = time.perf_counter()
        out = llm.create_chat_completion(msgs, max_tokens=MAX_TOKENS, temperature=0)
        elapsed = time.perf_counter() - t0
        n = out["usage"]["completion_tokens"]
        tps_list.append(n / elapsed if elapsed > 0 else 0)
    run_section(label, tps_list)


LONG_Q  = "Explain the causes and consequences of the French Revolution in detail."
SHORT_Q = "What is 2+2?"

if __name__ == "__main__":
    import sys

    # Load corpora
    def load_lines(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    short_corpus = load_lines("corpus_short.txt")
    long_corpus  = load_lines("corpus_long.txt")

    # Pad/repeat to N_REQUESTS
    def pad(lst, n):
        return (lst * ((n // len(lst)) + 1))[:n]

    short_q = pad(short_corpus, N_REQUESTS)
    long_q  = pad(long_corpus,  N_REQUESTS)

    # A list of the same long question repeated — isolates thermal/memory drift
    repeated_long = [LONG_Q] * N_REQUESTS
    repeated_short = [SHORT_Q] * N_REQUESTS

    print("Loading CT2 model...")
    ct2_tok, ct2_gen = load_ct2("int8")
    print("Loading llama.cpp model...")
    llama = load_llama()
    print("Models loaded.\n")

    # --- Experiment A: mixed long corpus (mirrors original benchmark) ---
    experiment_ct2_sequential(ct2_tok, ct2_gen, long_q,
                              "long corpus (mixed questions, max_tokens=256)")
    experiment_llama_sequential(llama, long_q,
                                "long corpus (mixed questions, max_tokens=256)")

    # --- Experiment B: same long question repeated ---
    experiment_ct2_sequential(ct2_tok, ct2_gen, repeated_long,
                              f"repeated identical long question x{N_REQUESTS}")
    experiment_llama_sequential(llama, repeated_long,
                                f"repeated identical long question x{N_REQUESTS}")

    # --- Experiment C: same SHORT question repeated (thermal baseline) ---
    experiment_ct2_sequential(ct2_tok, ct2_gen, repeated_short,
                              f"repeated identical SHORT question x{N_REQUESTS}")
    experiment_llama_sequential(llama, repeated_short,
                                f"repeated identical SHORT question x{N_REQUESTS}")
