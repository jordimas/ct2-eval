import argparse
import time
import ctranslate2
import numpy as np
from faster_whisper import WhisperModel

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="Benchmark for Whisper transcription across devices and compute types"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="whisper-tiny-ct2",
    help="CTranslate2 Whisper model path",
)
parser.add_argument(
    "--audio",
    type=str,
    default="short_audio.mp3",
    help="Path to audio file to transcribe.",
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
audio_path = args.audio
num_runs = args.num_runs
warmup_runs = args.warmup_runs
devices = ["cpu", "cuda"]

if verbose:
    print(f"Audio: {audio_path}")
    print(f"Model: {model_path}")
    print(f"Warm-up runs: {warmup_runs}")
    print(f"Number of timed runs per compute type: {num_runs}")
    print(f"CTranslate2 version: {ctranslate2.__version__}")
    print("=" * 70)


def transcribe_once(model, audio_path):
    segments, info = model.transcribe(audio_path)
    text_parts = []
    num_tokens = 0
    for segment in segments:
        text_parts.append(segment.text)
        if segment.tokens is not None:
            num_tokens += len(segment.tokens)
    return "".join(text_parts), num_tokens, info


# ------------------------
# Benchmark transcriptions for each device and compute type
# ------------------------
results = []

for device in devices:
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        if verbose:
            print("\n" + "=" * 70)
            print("CUDA not available - skipping GPU benchmark")
            print("=" * 70)
        continue

    if verbose:
        print(f"\nDEVICE: {device.upper()}")
        print("=" * 70)

    supported_compute_types = ctranslate2.get_supported_compute_types(device)

    if verbose:
        print(f"Supported compute types: {sorted(supported_compute_types)}")

    for compute_type in sorted(supported_compute_types):
        if verbose:
            print(
                f"\nTesting compute_type: {compute_type} ({warmup_runs} warm-up + {num_runs} timed runs)"
            )

        try:
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"  WARNING: Skipping {compute_type} - {e}")
            continue

        language = None
        probability = 0.0

        # ------------------------
        # Warm-up runs
        # ------------------------
        for warmup_idx in range(warmup_runs):
            warmup_start = time.time()
            _, _, info = transcribe_once(model, audio_path)
            warmup_elapsed = time.time() - warmup_start
            language = info.language
            probability = info.language_probability
            if verbose:
                print(f"  Warm-up {warmup_idx + 1}/{warmup_runs}: {warmup_elapsed:.2f}s")

        # ------------------------
        # Timed runs
        # ------------------------
        run_times = []
        run_tokens_per_sec = []
        last_transcription = ""

        for run_idx in range(num_runs):
            start_time = time.time()
            transcription, num_tokens, info = transcribe_once(model, audio_path)
            end_time = time.time()
            elapsed_time = end_time - start_time

            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0

            language = info.language
            probability = info.language_probability
            last_transcription = transcription

            run_times.append(elapsed_time)
            run_tokens_per_sec.append(tokens_per_sec)

            if verbose:
                print(
                    f"  Run {run_idx + 1}/{num_runs}: Tokens: {num_tokens} | "
                    f"Time: {elapsed_time:.2f}s | Tokens/sec: {tokens_per_sec:.2f}"
                )

        del model

        results.append(
            {
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "language_prob": probability,
                "time_mean": np.mean(run_times),
                "time_std": np.std(run_times),
                "tokens_per_sec_mean": np.mean(run_tokens_per_sec),
                "tokens_per_sec_std": np.std(run_tokens_per_sec),
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            }
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
                f"  Summary: Time: {np.mean(run_times):.2f}s ± {time_cv:.1f}% | "
                f"Tokens/sec: {np.mean(run_tokens_per_sec):.2f} ± {tps_cv:.1f}%"
            )
            print(f"  Language: {language} ({probability:.2f})")
            print(f"  Transcription: {last_transcription[:100]}")


# ------------------------
# Summary table
# ------------------------
def cv_percent(mean, std):
    """Calculate coefficient of variation as percentage."""
    return (std / mean * 100) if mean != 0 else 0


print("\n" + "=" * 90)
print("SUMMARY (± values are std as % of mean)")
print("=" * 90)
print(f"{'Device':<10} {'Compute Type':<15} {'Time (s)':<22} {'Tokens/sec':<22}")
print("-" * 90)
for r in results:
    time_cv = cv_percent(r["time_mean"], r["time_std"])
    tps_cv = cv_percent(r["tokens_per_sec_mean"], r["tokens_per_sec_std"])

    time_str = f"{r['time_mean']:.2f} ± {time_cv:.1f}%"
    tps_str = f"{r['tokens_per_sec_mean']:.2f} ± {tps_cv:.1f}%"
    print(f"{r['device']:<10} {r['compute_type']:<15} {time_str:<22} {tps_str:<22}")

print(
    f"\nAudio: {audio_path} | "
    f"Warm-up runs per compute type: {warmup_runs} | Timed runs per compute type: {num_runs} | "
    f"CTranslate2 version: {ctranslate2.__version__}"
)
