import argparse
import json
import os
import time
from datasets import load_dataset
from jiwer import wer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import ctranslate2
from faster_whisper import WhisperModel
import numpy as np

from datasets import disable_progress_bars

disable_progress_bars()

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="WER benchmark across devices and compute types"
)
parser.add_argument(
    "--audio_numb",
    type=int,
    default=20,
    help="Specify the number of validation audio files in the dataset."
    " Set to None to retrieve all audio files.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="small",
    help="Whisper model path or size (e.g., tiny, base, small, medium, large-v3)",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=3,
    help="Number of times to run each compute type for statistical analysis.",
)
parser.add_argument(
    "--warmup_runs",
    type=int,
    default=1,
    help="Number of warm-up runs before timed runs (not included in statistics).",
)

parser.add_argument(
    "--no_verbose", action="store_true", dest="no_verbose", help="Usa verbose output."
)
args = parser.parse_args()

verbose = not args.no_verbose

model_path = args.model_path
num_runs = args.num_runs
warmup_runs = args.warmup_runs
devices = ["cpu", "cuda"]

# ------------------------
# Load normalizer
# ------------------------
with open(os.path.join(os.path.dirname(__file__), "normalizer.json"), "r") as f:
    normalizer = EnglishTextNormalizer(json.load(f))

# ------------------------
# Load dataset with streaming and pre-fetch samples
# ------------------------
if verbose:
    print("Loading dataset with streaming...")
dataset_stream = load_dataset(
    "librispeech_asr", "clean", split="validation", streaming=True
)

# Pre-fetch and buffer samples to avoid streaming overhead during timing
if verbose:
    print("Pre-fetching audio samples...")
samples = []
for i, sample in enumerate(dataset_stream):
    samples.append(
        {
            "audio_array": sample["audio"]["array"],
            "sampling_rate": sample["audio"]["sampling_rate"],
            "text": sample["text"],
        }
    )
    if args.audio_numb and i + 1 >= args.audio_numb:
        break

if verbose:
    print(f"Buffered {len(samples)} audio samples")
    print(f"Model: {model_path}")
    print(f"Warm-up runs: {warmup_runs}")
    print(f"Number of timed runs per compute type: {num_runs}")
    print("=" * 70)

# ------------------------
# Benchmark for each device and compute type
# ------------------------
results = []

for device in devices:
    if verbose:
        print(f"DEVICE: {device.upper()}")
        print(f"{'='*70}")

    # Get supported compute types for this device
    supported_compute_types = ctranslate2.get_supported_compute_types(device)

    if verbose:
        print(f"Supported compute types: {sorted(supported_compute_types)}")

    for compute_type in sorted(supported_compute_types):
        if verbose:
            print(
                f"\nTesting compute_type: {compute_type} ({warmup_runs} warm-up + {num_runs} timed runs)"
            )

        # Load model with current device and compute type
        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        # ------------------------
        # Warm-up runs
        # ------------------------
        for warmup_idx in range(warmup_runs):
            warmup_start = time.time()
            for sample in samples:
                audio_array = sample["audio_array"]
                segments, info = model.transcribe(audio_array, language="en")
                # Consume the generator to ensure inference completes
                _ = list(segments)
            warmup_elapsed = time.time() - warmup_start
            if verbose:
                print(
                    f"  Warm-up {warmup_idx + 1}/{warmup_runs}: {warmup_elapsed:.2f}s"
                )

        # ------------------------
        # Timed runs
        # ------------------------
        # Store metrics for each run
        run_wers = []
        run_times = []
        run_rtfs = []
        run_speeds = []
        total_audio_duration = 0.0

        for run_idx in range(num_runs):
            all_transcriptions = []
            all_references = []
            run_audio_duration = 0.0

            start_time = time.time()

            # Iterate over the pre-fetched samples and run inference
            for sample in samples:
                audio_array = sample["audio_array"]
                sampling_rate = sample["sampling_rate"]

                # Calculate audio duration
                audio_duration = len(audio_array) / sampling_rate
                run_audio_duration += audio_duration

                # Transcribe
                segments, info = model.transcribe(audio_array, language="en")
                transcription = "".join([segment.text for segment in segments])

                all_transcriptions.append(transcription)
                all_references.append(sample["text"])

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Normalize predictions and references
            all_transcriptions_norm = [normalizer(t) for t in all_transcriptions]
            all_references_norm = [normalizer(r) for r in all_references]

            # Compute WER
            word_error_rate = 100 * wer(
                hypothesis=all_transcriptions_norm, reference=all_references_norm
            )

            # Calculate real-time factor (RTF)
            rtf = elapsed_time / run_audio_duration if run_audio_duration > 0 else 0
            speed = 1 / rtf if rtf > 0 else 0

            # Store run metrics
            run_wers.append(word_error_rate)
            run_times.append(elapsed_time)
            run_rtfs.append(rtf)
            run_speeds.append(speed)
            total_audio_duration = run_audio_duration  # Same for all runs

            if verbose:

                print(
                    f"  Run {run_idx + 1}/{num_runs}: WER: {word_error_rate:.3f}% | "
                    f"Time: {elapsed_time:.2f}s | RTF: {rtf:.4f} | Speed: {speed:.2f}x"
                )

        # Clean up model after all runs for this compute type
        del model

        # Calculate mean and std for all metrics
        results.append(
            {
                "device": device,
                "compute_type": compute_type,
                "wer_mean": np.mean(run_wers),
                "wer_std": np.std(run_wers),
                "time_mean": np.mean(run_times),
                "time_std": np.std(run_times),
                "rtf_mean": np.mean(run_rtfs),
                "rtf_std": np.std(run_rtfs),
                "speed_mean": np.mean(run_speeds),
                "speed_std": np.std(run_speeds),
                "audio_duration": total_audio_duration,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            }
        )

        wer_cv = (
            (np.std(run_wers) / np.mean(run_wers) * 100)
            if np.mean(run_wers) != 0
            else 0
        )
        time_cv = (
            (np.std(run_times) / np.mean(run_times) * 100)
            if np.mean(run_times) != 0
            else 0
        )
        speed_cv = (
            (np.std(run_speeds) / np.mean(run_speeds) * 100)
            if np.mean(run_speeds) != 0
            else 0
        )
        if verbose:
            print(f"  ──────────────────────────────────────────────────────────────")
            print(
                f"  Summary: WER: {np.mean(run_wers):.3f}% ± {wer_cv:.1f}% | "
                f"Time: {np.mean(run_times):.2f}s ± {time_cv:.1f}% | "
                f"Speed: {np.mean(run_speeds):.2f}x ± {speed_cv:.1f}%"
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
print(
    f"{'Device':<10} {'Compute Type':<15} {'WER (%)':<22} {'Time (s)':<22} {'RTF':<22} {'Speed (x)':<22}"
)
print("-" * 110)
for r in results:
    wer_cv = cv_percent(r["wer_mean"], r["wer_std"])
    time_cv = cv_percent(r["time_mean"], r["time_std"])
    rtf_cv = cv_percent(r["rtf_mean"], r["rtf_std"])
    speed_cv = cv_percent(r["speed_mean"], r["speed_std"])

    wer_str = f"{r['wer_mean']:.3f} ± {wer_cv:.1f}%"
    time_str = f"{r['time_mean']:.2f} ± {time_cv:.1f}%"
    rtf_str = f"{r['rtf_mean']:.4f} ± {rtf_cv:.1f}%"
    speed_str = f"{r['speed_mean']:.2f} ± {speed_cv:.1f}%"
    print(
        f"{r['device']:<10} {r['compute_type']:<15} {wer_str:<22} {time_str:<22} {rtf_str:<22} {speed_str:<22}"
    )

print(
    f"\nTotal audio duration: {results[0]['audio_duration']:.2f} seconds | Number of samples: {len(samples)} | Warm-up runs per compute type: {warmup_runs} | Timed runs per compute type: {num_runs}|CTranslate2 version: {ctranslate2.__version__}"
)
