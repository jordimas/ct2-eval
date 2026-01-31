from transformers import AutoTokenizer
import ctranslate2
import time
from sacrebleu.metrics import BLEU

# ------------------------
# Load Gemma 3 tokenizer
# ------------------------
model_id = "gemma-3-270m"
tok = AutoTokenizer.from_pretrained(f"google/{model_id}")

# ------------------------
# Read FLORES-200 files (limited to 50 sentences)
# ------------------------
with open("flores200.eng", "r", encoding="utf-8") as f:
    english_sentences = [line.strip() for line in f if line.strip()][:50]
with open("flores200.cat", "r", encoding="utf-8") as f:
    catalan_references = [line.strip() for line in f if line.strip()][:50]

assert len(english_sentences) == len(
    catalan_references
), f"Mismatch: {len(english_sentences)} English sentences vs {len(catalan_references)} Catalan references"

print(f"Loaded {len(english_sentences)} sentence pairs")
print(f"CTranslate2 version: {ctranslate2.__version__}")
print("=" * 60)

# ------------------------
# Initialize BLEU scorer
# ------------------------
bleu = BLEU()

# ------------------------
# Count tokens (done once, shared across benchmarks)
# ------------------------
total_tokens = sum(len(tok.encode(sentence)) for sentence in english_sentences)


# ------------------------
# Translation function using Gemma 3
# ------------------------
def translate_with_gemma(gen, english_text):
    prompt = f"<start_of_turn>user\nTranslate the following English text to Catalan:\n{english_text}<end_of_turn>\n<start_of_turn>model\n"
    tokens = tok.convert_ids_to_tokens(tok.encode(prompt))
    res = gen.generate_batch(
        [tokens],
        max_length=512,
        sampling_temperature=0.1,
        include_prompt_in_result=False,
    )
    return tok.convert_tokens_to_string(res[0].sequences[0])


# ------------------------
# Benchmark function for a specific device
# ------------------------
def run_benchmark(device):
    print(f"Running {model_id} translation benchmark on {device.upper()}")
    print("=" * 60)

    # Load model for this device
    gen = ctranslate2.Generator(f"{model_id}.ct2", device=device)

    # Translate all sentences
    start_time = time.time()
    translations = [
        translate_with_gemma(gen, sentence) for sentence in english_sentences
    ]
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute BLEU score
    bleu_score = bleu.corpus_score(translations, [catalan_references])

    print(f"BLEU: {bleu_score.score:.2f}")
    print(f"Time: {elapsed_time:.2f}s")
    print(f"Tokens: {total_tokens}")
    print(f"Tokens/sec: {total_tokens / elapsed_time:.2f}")

    print(f"\nSample translations:")
    for i in range(min(3, len(translations))):
        print(f"  EN: {english_sentences[i][:60]}...")
        print(f"  CA: {translations[i][:60]}...")
        print()

    return {
        "device": device,
        "bleu_score": bleu_score.score,
        "elapsed_time": elapsed_time,
        "tokens_per_sec": total_tokens / elapsed_time,
    }


# ------------------------
# Run benchmarks on both devices
# ------------------------
results = []

# CPU benchmark
results.append(run_benchmark("cpu"))

# CUDA benchmark (check if available)
if ctranslate2.get_cuda_device_count() > 0:
    results.append(run_benchmark("cuda"))
else:
    print("\n" + "=" * 60)
    print("CUDA not available - skipping GPU benchmark")
    print("=" * 60)

# ------------------------
# Summary
# ------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Sentences: {len(english_sentences)}")
print(f"Total Tokens: {total_tokens}")
print(f"CTranslate2 version: {ctranslate2.__version__}")
print()

print(f"{'Device':<10} {'BLEU':<10} {'Time (s)':<12} {'Tokens/sec':<12}")
print("-" * 44)
for r in results:
    print(
        f"{r['device'].upper():<10} {r['bleu_score']:<10.2f} {r['elapsed_time']:<12.2f} {r['tokens_per_sec']:<12.2f}"
    )
