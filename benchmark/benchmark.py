import os
# 4 threads is the bandwidth sweet spot for int8 decode on this i7-11800H.
# More threads saturate DDR4 bandwidth and cause thermal throttling with no gain.
os.environ["OMP_NUM_THREADS"] = "4"
import time
import argparse
import statistics
import ctranslate2
import transformers

MODEL_ID = "google/gemma-3-1b-it"
GGUF_REPO = "ggml-org/gemma-3-1b-it-GGUF"
GGUF_FILE = "gemma-3-1b-it-Q8_0.gguf"
WARMUP = 3
MAX_TOKENS = 256

CORPUS_SHORT = "corpus_short.txt"
CORPUS_LONG = "corpus_long.txt"


def load_corpus(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def summarize(tps_list, label):
    lines = [
        f"  avg={statistics.mean(tps_list):.1f}  "
        f"p50={statistics.median(tps_list):.1f}  "
        f"min={min(tps_list):.1f}  "
        f"max={max(tps_list):.1f}  "
        f"std={statistics.stdev(tps_list):.1f} tok/s  "
        f"({label})"
    ]
    return lines


def bench_ctranslate2(device, quant, batch_size, corpus):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    suffix = f"-{quant}" if quant else ""
    ct2_model_dir = f"{MODEL_ID}-ct2{suffix}"
    if not os.path.isdir(ct2_model_dir):
        ctranslate2.converters.TransformersConverter(MODEL_ID).convert(
            ct2_model_dir, quantization=quant if quant else None
        )
    generator = ctranslate2.Generator(ct2_model_dir, device=device,
                                      compute_type=quant if quant else "int8",
                                      inter_threads=1,
                                      intra_threads=4)

    def tokenize(question):
        prompt = build_prompt(tokenizer, question)
        return tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt, add_special_tokens=False))

    warmup_tokens = tokenize(corpus[0])
    for _ in range(WARMUP):
        generator.generate_batch([warmup_tokens] * batch_size, max_length=MAX_TOKENS,
                                  include_prompt_in_result=False, beam_size=1)

    label = f"CTranslate2 ({device}, int8, batch={batch_size}, beam=1)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for i, question in enumerate(questions):
            input_tokens = tokenize(question)
            batch = [input_tokens] * batch_size
            t0 = time.perf_counter()
            results = generator.generate_batch(batch, max_length=MAX_TOKENS,
                                               include_prompt_in_result=False, beam_size=1)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(len(r.sequences_ids[0]) for r in results)
            throughput = total_tokens / elapsed
            latency = elapsed / batch_size
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


def bench_vllm(device, batch_size):
    from vllm import LLM, SamplingParams

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, max_model_len=512,
              dtype="float32",
              kv_cache_memory_bytes=2 * 1024 ** 3)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)

    warmup_prompt = build_prompt(tokenizer, load_corpus(CORPUS_SHORT)[0])
    for _ in range(WARMUP):
        llm.generate([warmup_prompt] * batch_size, params)

    label = f"vLLM ({device}, float32, batch={batch_size}, greedy)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for i, question in enumerate(questions):
            prompt = build_prompt(tokenizer, question)
            batch = [prompt] * batch_size
            t0 = time.perf_counter()
            outputs = llm.generate(batch, params)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed
            latency = elapsed / batch_size
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


def bench_sglang(device, batch_size):
    import sglang as sgl

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = "float32" if device == "cpu" else "float16"
    engine = sgl.Engine(model_path=MODEL_ID, context_length=512, dtype=dtype)
    sampling_params = {"max_new_tokens": MAX_TOKENS, "temperature": 0}

    warmup_prompt = build_prompt(tokenizer, load_corpus(CORPUS_SHORT)[0])
    for _ in range(WARMUP):
        engine.generate([warmup_prompt] * batch_size, sampling_params)

    label = f"SGLang ({device}, {dtype}, batch={batch_size}, greedy)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for question in questions:
            prompt = build_prompt(tokenizer, question)
            batch = [prompt] * batch_size
            t0 = time.perf_counter()
            outputs = engine.generate(batch, sampling_params)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
            throughput = total_tokens / elapsed
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


def bench_openvino(device, batch_size, quant="int8"):
    from optimum.intel.openvino import OVModelForCausalLM
    from optimum.intel import OVWeightQuantizationConfig

    ov_model_dir = f"{MODEL_ID}-ov-{quant}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    if not os.path.isdir(ov_model_dir):
        if quant == "int8":
            qconfig = OVWeightQuantizationConfig(bits=8)
        elif quant == "int4":
            qconfig = OVWeightQuantizationConfig(bits=4)
        else:
            qconfig = None
        model = OVModelForCausalLM.from_pretrained(
            MODEL_ID, export=True, quantization_config=qconfig
        )
        model.save_pretrained(ov_model_dir)
        tokenizer.save_pretrained(ov_model_dir)
    else:
        model = OVModelForCausalLM.from_pretrained(ov_model_dir)

    warmup_prompt = build_prompt(tokenizer, load_corpus(CORPUS_SHORT)[0])
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt")
    for _ in range(WARMUP):
        model.generate(**warmup_inputs, max_new_tokens=MAX_TOKENS, do_sample=False)

    label = f"OpenVINO ({device}, {quant}, batch={batch_size}, greedy)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for question in questions:
            prompt = build_prompt(tokenizer, question)
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]
            t0 = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(len(ids) - input_len for ids in output_ids)
            throughput = total_tokens / elapsed
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


def bench_onnxruntime(device, batch_size, quant="int8"):
    from optimum.onnxruntime import ORTModelForCausalLM

    ort_model_dir = f"{MODEL_ID}-ort-{quant}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    def _model_ready(path):
        return any(
            os.path.isfile(os.path.join(path, f))
            for f in ("model.onnx", "model_quantized.onnx")
        )

    if not _model_ready(ort_model_dir):
        if quant == "int8":
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            fp32_dir = f"{MODEL_ID}-ort-fp32"
            if not _model_ready(fp32_dir):
                m = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
                m.save_pretrained(fp32_dir)
                tokenizer.save_pretrained(fp32_dir)
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name="model.onnx")
            quantizer.quantize(save_dir=ort_model_dir, quantization_config=qconfig)
            tokenizer.save_pretrained(ort_model_dir)
        else:
            m = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
            m.save_pretrained(ort_model_dir)
            tokenizer.save_pretrained(ort_model_dir)

    quant_file = "model_quantized.onnx" if quant == "int8" else "model.onnx"
    model = ORTModelForCausalLM.from_pretrained(ort_model_dir, file_name=quant_file)

    warmup_prompt = build_prompt(tokenizer, load_corpus(CORPUS_SHORT)[0])
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt")
    for _ in range(WARMUP):
        model.generate(**warmup_inputs, max_new_tokens=MAX_TOKENS, do_sample=False)

    label = f"ONNX Runtime ({device}, {quant}, batch={batch_size}, greedy)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for question in questions:
            prompt = build_prompt(tokenizer, question)
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]
            t0 = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
            elapsed = time.perf_counter() - t0
            total_tokens = sum(len(ids) - input_len for ids in output_ids)
            throughput = total_tokens / elapsed
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


def bench_llama_cpp(batch_size):
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    llm = Llama(model_path=model_path, n_threads=os.cpu_count() // 2,
                n_ctx=512, verbose=False)

    warmup_msg = [{"role": "user", "content": load_corpus(CORPUS_SHORT)[0]}]
    for _ in range(WARMUP):
        llm.create_chat_completion(warmup_msg, max_tokens=MAX_TOKENS, temperature=0)

    label = f"llama.cpp (cpu, Q8_0, batch={batch_size} sequential, greedy)"
    lines = [f"\n=== {label} ==="]
    tps_short, tps_long = [], []

    short_corpus = load_corpus(CORPUS_SHORT)
    long_corpus = load_corpus(CORPUS_LONG)

    for tag, questions, tps_list in [("short", short_corpus, tps_short), ("long", long_corpus, tps_long)]:
        lines.append(f"\n  -- {tag} questions --")
        for i, question in enumerate(questions):
            messages = [{"role": "user", "content": question}]
            t0 = time.perf_counter()
            total_tokens = 0
            for _ in range(batch_size):
                output = llm.create_chat_completion(messages, max_tokens=MAX_TOKENS, temperature=0)
                total_tokens += output["usage"]["completion_tokens"]
            elapsed = time.perf_counter() - t0
            throughput = total_tokens / elapsed
            latency = elapsed / batch_size
            tps_list.append(throughput)
        lines += summarize(tps_list, tag)

    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--backend", choices=["ct2", "vllm", "llama", "sglang", "ov", "ort", "all"], default="all")
    parser.add_argument("--quant", choices=["int8", "int8_float16", "float16", "int4"], default="int8")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    results = []
    if args.backend in ("ct2", "all"):
        try:
            results += bench_ctranslate2(args.device, args.quant, args.batch_size,
                                         load_corpus(CORPUS_SHORT))
        except Exception as e:
            print(f"[ct2] skipped: {e}")
    if args.backend in ("vllm", "all"):
        try:
            results += bench_vllm(args.device, args.batch_size)
        except Exception as e:
            print(f"[vllm] skipped: {e}")
    if args.backend in ("llama", "all"):
        try:
            results += bench_llama_cpp(args.batch_size)
        except Exception as e:
            print(f"[llama] skipped: {e}")
    if args.backend in ("sglang", "all"):
        try:
            results += bench_sglang(args.device, args.batch_size)
        except Exception as e:
            print(f"[sglang] skipped: {e}")
    if args.backend in ("ov", "all"):
        try:
            ov_quant = args.quant if args.quant in ("int8", "int4") else "int8"
            results += bench_openvino(args.device, args.batch_size, ov_quant)
        except Exception as e:
            print(f"[openvino] skipped: {e}")
    if args.backend in ("ort", "all"):
        try:
            ort_quant = args.quant if args.quant in ("int8",) else "fp32"
            results += bench_onnxruntime(args.device, args.batch_size, ort_quant)
        except Exception as e:
            print(f"[ort] skipped: {e}")

    output = "\n" + "=" * 40 + " RESULTS " + "=" * 40 + "\n" + "\n".join(results)
    print(output)

    report_file = f"report_{args.device}.txt"
    with open(report_file, "w") as f:
        f.write(output)
    print(f"\nReport written to {report_file}")
