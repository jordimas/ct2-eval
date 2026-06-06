import argparse
import time
import sentencepiece as spm
import ctranslate2
from sacrebleu.metrics import BLEU
import numpy as np

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="BLEU benchmark for translation across devices and compute types"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="eng-cat/ctranslate2/",
    help="CTranslate2 model path",
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    default="eng-cat/tokenizer/sp_m.model",
    help="SentencePiece tokenizer model path",
)
parser.add_argument(
    "--num_sentences",
    type=int,
    default=None,
    help="Number of sentences to translate from FLORES-200 dataset (default: all).",
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
model_path = args.model_path
tokenizer_path = args.tokenizer_path
num_runs = args.num_runs
warmup_runs = args.warmup_runs
num_sentences = args.num_sentences
devices = ["cuda"]

# ------------------------
# Load tokenizer
# ------------------------
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

# ------------------------
# Read FLORES-200 files
# ------------------------
with open("flores200.eng", "r", encoding="utf-8") as f:
    english_sentences = [line.strip() for line in f if line.strip()]
with open("flores200.cat", "r", encoding="utf-8") as f:
    catalan_references = [line.strip() for line in f if line.strip()]

# Limit sentences if specified
if num_sentences is not None:
    english_sentences = english_sentences[:num_sentences]
    catalan_references = catalan_references[:num_sentences]

assert len(english_sentences) == len(
    catalan_references
), f"Mismatch: {len(english_sentences)} English sentences vs {len(catalan_references)} Catalan references"

if verbose:
    print(f"Loaded {len(english_sentences)} sentence pairs")
    print(f"Model: {model_path}")
    print(f"Warm-up runs: {warmup_runs}")
    print(f"Number of timed runs per compute type: {num_runs}")
    print(f"CTranslate2 version: {ctranslate2.__version__}")
    print("=" * 70)

# ------------------------
# Initialize BLEU scorer
# ------------------------
bleu = BLEU()

# ------------------------
# Tokenize all English sentences (done once, shared across benchmarks)
# ------------------------
tokenized_sentences = [
    tokenizer.encode(sentence, out_type=str) for sentence in english_sentences
]
total_tokens = sum(len(tokens) for tokens in tokenized_sentences)

# ------------------------
# Benchmark translations for each device and compute type
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

        try:
            # Load translator for this device and compute type
            translator = ctranslate2.Translator(
                model_path,
                compute_type=compute_type,
                device=device,
            )
        except RuntimeError as e:
            if verbose:
                print(f"  WARNING: Skipping {compute_type} - {e}")
            continue

        # ------------------------
        # Warm-up runs
        # ------------------------
        for warmup_idx in range(warmup_runs):
            warmup_start = time.time()
            _ = translator.translate_batch(tokenized_sentences, max_batch_size=32)
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
            translated_batches = translator.translate_batch(tokenized_sentences, max_batch_size=32)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Decode results
            translations = [
                tokenizer.decode(t.hypotheses[0]) for t in translated_batches
            ]

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

        # Clean up translator after all runs for this compute type
        del translator

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
