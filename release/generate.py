import argparse
import time
from transformers import AutoTokenizer
import ctranslate2
from sacrebleu.metrics import BLEU
import numpy as np

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="BLEU benchmark for Gemma 3 translation across devices and compute types"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="gemma-3-270m",
    help="Gemma model path or ID (e.g., gemma-3-270m)",
)
parser.add_argument(
    "--num_sentences",
    type=int,
    default=10,
    help="Number of sentences to translate from FLORES-200 dataset.",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=3,
    help="Number of times to run each device for statistical analysis.",
)
parser.add_argument(
    "--warmup_runs",
    type=int,
    default=1,
    help="Number of warm-up runs before timed runs (not included in statistics).",
)
parser.add_argument(
    "--no_verbose",
    action="store_true",
    dest="no_verbose",
    help="Disable verbose output.",
)
args = parser.parse_args()

verbose = not args.no_verbose
model_id = args.model_path
num_runs = args.num_runs
warmup_runs = args.warmup_runs
num_sentences = args.num_sentences
devices = ["cpu", "cuda"]

# ------------------------
# Load Gemma 3 tokenizer
# ------------------------
tok = AutoTokenizer.from_pretrained(f"google/{model_id}")

# ------------------------
# Read FLORES-200 files (limited to num_sentences)
# ------------------------
with open("flores200.eng", "r", encoding="utf-8") as f:
    english_sentences = [line.strip() for line in f if line.strip()][:num_sentences]
with open("flores200.cat", "r", encoding="utf-8") as f:
    catalan_references = [line.strip() for line in f if line.strip()][:num_sentences]

assert len(english_sentences) == len(
    catalan_references
), f"Mismatch: {len(english_sentences)} English sentences vs {len(catalan_references)} Catalan references"

if verbose:
    print(f"Loaded {len(english_sentences)} sentence pairs")
    print(f"Model: {model_id}")
    print(f"Warm-up runs: {warmup_runs}")
    print(f"Number of timed runs per compute type: {num_runs}")
    print(f"CTranslate2 version: {ctranslate2.__version__}")
    print("=" * 70)

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
# Benchmark for each device and compute type
# ------------------------
results = []

for device in devices:
    # Check if CUDA is available
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        if verbose:
            print("\n" + "=" * 70)
            print("CUDA not available - skipping GPU benchmark")
            print("=" * 70)
        continue

    if verbose:
        print(f"\nDEVICE: {device.upper()}")
        print("=" * 70)

    # Get supported compute types for this device
    supported_compute_types = ctranslate2.get_supported_compute_types(device)

    if verbose:
        print(f"Supported compute types: {sorted(supported_compute_types)}")

    for compute_type in sorted(supported_compute_types):
        if verbose:
            print(
                f"\nTesting compute_type: {compute_type} ({warmup_runs} warm-up + {num_runs} timed runs)"
            )

        # Load model for this device and compute type
        gen = ctranslate2.Generator(
            f"{model_id}.ct2", device=device, compute_type=compute_type
        )

        # ------------------------
        # Warm-up runs
        # ------------------------
        for warmup_idx in range(warmup_runs):
            warmup_start = time.time()
            for sentence in english_sentences:
                _ = translate_with_gemma(gen, sentence)
            warmup_elapsed = time.time() - warmup_start
            if verbose:
                print(f"  Warm-up {warmup_idx + 1}/{warmup_runs}: {warmup_elapsed:.2f}s")

        # ------------------------
        # Timed runs
        # ------------------------
        run_bleus = []
        run_times = []
        run_tokens_per_sec = []

        for run_idx in range(num_runs):
            start_time = time.time()
            translations = [
                translate_with_gemma(gen, sentence) for sentence in english_sentences
            ]
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Compute BLEU score
            bleu_score = bleu.corpus_score(translations, [catalan_references])
            tokens_per_sec = total_tokens / elapsed_time

            # Store run metrics
            run_bleus.append(bleu_score.score)
            run_times.append(elapsed_time)
            run_tokens_per_sec.append(tokens_per_sec)

            if verbose:
                print(
                    f"  Run {run_idx + 1}/{num_runs}: BLEU: {bleu_score.score:.2f} | "
                    f"Time: {elapsed_time:.2f}s | Tokens/sec: {tokens_per_sec:.2f}"
                )

        # Clean up model after all runs for this compute type
        del gen

        # Calculate mean and std for all metrics
        results.append(
            {
                "device": device,
                "compute_type": compute_type,
                "bleu_mean": np.mean(run_bleus),
                "bleu_std": np.std(run_bleus),
                "time_mean": np.mean(run_times),
                "time_std": np.std(run_times),
                "tokens_per_sec_mean": np.mean(run_tokens_per_sec),
                "tokens_per_sec_std": np.std(run_tokens_per_sec),
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            }
        )

        bleu_cv = (
            (np.std(run_bleus) / np.mean(run_bleus) * 100) if np.mean(run_bleus) != 0 else 0
        )
        time_cv = (
            (np.std(run_times) / np.mean(run_times) * 100) if np.mean(run_times) != 0 else 0
        )
        tps_cv = (
            (np.std(run_tokens_per_sec) / np.mean(run_tokens_per_sec) * 100)
            if np.mean(run_tokens_per_sec) != 0
            else 0
        )

        if verbose:
            print(f"  ──────────────────────────────────────────────────────────────")
            print(
                f"  Summary: BLEU: {np.mean(run_bleus):.2f} ± {bleu_cv:.1f}% | "
                f"Time: {np.mean(run_times):.2f}s ± {time_cv:.1f}% | "
                f"Tokens/sec: {np.mean(run_tokens_per_sec):.2f} ± {tps_cv:.1f}%"
            )


# ------------------------
# Summary table
# ------------------------
def cv_percent(mean, std):
    """Calculate coefficient of variation as percentage."""
    return (std / mean * 100) if mean != 0 else 0


print("\n" + "=" * 110)
print("SUMMARY (± values are std as % of mean)")
print("=" * 110)
print(f"{'Device':<10} {'Compute Type':<15} {'BLEU':<22} {'Time (s)':<22} {'Tokens/sec':<22}")
print("-" * 110)
for r in results:
    bleu_cv = cv_percent(r["bleu_mean"], r["bleu_std"])
    time_cv = cv_percent(r["time_mean"], r["time_std"])
    tps_cv = cv_percent(r["tokens_per_sec_mean"], r["tokens_per_sec_std"])

    bleu_str = f"{r['bleu_mean']:.2f} ± {bleu_cv:.1f}%"
    time_str = f"{r['time_mean']:.2f} ± {time_cv:.1f}%"
    tps_str = f"{r['tokens_per_sec_mean']:.2f} ± {tps_cv:.1f}%"
    print(f"{r['device']:<10} {r['compute_type']:<15} {bleu_str:<22} {time_str:<22} {tps_str:<22}")

print(
    f"\nSentences: {len(english_sentences)} | Total tokens: {total_tokens} | "
    f"Warm-up runs per compute type: {warmup_runs} | Timed runs per compute type: {num_runs} | "
    f"CTranslate2 version: {ctranslate2.__version__}"
)
