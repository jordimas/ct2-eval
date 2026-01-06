import argparse
import json
import os
import time
from datasets import load_dataset
from jiwer import wer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import ctranslate2
from faster_whisper import WhisperModel

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="WER benchmark across devices and compute types"
)
parser.add_argument(
    "--audio_numb",
    type=int,
    default=10,
    help="Specify the number of validation audio files in the dataset."
    " Set to None to retrieve all audio files.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="small",
    help="Whisper model path or size (e.g., tiny, base, small, medium, large-v3)",
)
args = parser.parse_args()

model_path = args.model_path
devices = ["cpu", "cuda"]

# ------------------------
# Load normalizer
# ------------------------
with open(os.path.join(os.path.dirname(__file__), "normalizer.json"), "r") as f:
    normalizer = EnglishTextNormalizer(json.load(f))

# ------------------------
# Load dataset with streaming and pre-fetch samples
# ------------------------
print("Loading dataset with streaming...")
dataset_stream = load_dataset(
    "librispeech_asr", "clean", split="validation", streaming=True
)

# Pre-fetch and buffer samples to avoid streaming overhead during timing
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

print(f"Buffered {len(samples)} audio samples")
print(f"Model: {model_path}")
print("=" * 70)

# ------------------------
# Benchmark for each device and compute type
# ------------------------
results = []

for device in devices:
    print(f"\n{'='*70}")
    print(f"DEVICE: {device.upper()}")
    print(f"{'='*70}")

    # Get supported compute types for this device
    supported_compute_types = ctranslate2.get_supported_compute_types(device)
    print(f"Supported compute types: {sorted(supported_compute_types)}")

    for compute_type in sorted(supported_compute_types):
        print(f"\nTesting compute_type {compute_type}")

        # Load model with current device and compute type
        model = WhisperModel(model_path, device=device, compute_type=compute_type)

        all_transcriptions = []
        all_references = []
        total_audio_duration = 0.0

        start_time = time.time()

        # Iterate over the pre-fetched samples and run inference
        for sample in samples:
            audio_array = sample["audio_array"]
            sampling_rate = sample["sampling_rate"]

            # Calculate audio duration
            audio_duration = len(audio_array) / sampling_rate
            total_audio_duration += audio_duration

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
        rtf = elapsed_time / total_audio_duration if total_audio_duration > 0 else 0

        # Store results
        results.append(
            {
                "device": device,
                "compute_type": compute_type,
                "wer": word_error_rate,
                "time": elapsed_time,
                "audio_duration": total_audio_duration,
                "rtf": rtf,
                "speed": 1 / rtf if rtf > 0 else 0,  # x times real-time
            }
        )

        print(
            f"WER: {word_error_rate:.3f}% | Time: {elapsed_time:.2f}s | Audio: {total_audio_duration:.2f}s | RTF: {rtf:.4f} (lower=faster) | Speed: {1/rtf:.2f}x"
        )
        print(f"  Sample transcription: {all_transcriptions[0][:80]}...")
        del model

# ------------------------
# Summary table
# ------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(
    f"{'Device':<10} {'Compute Type':<15} {'WER (%)':<12} {'Time (s)':<12} {'RTF':<12} {'Speed':<12}"
)
print("-" * 80)
for r in results:
    print(
        f"{r['device']:<10} {r['compute_type']:<15} {r['wer']:<12.3f} {r['time']:<12.2f} {r['rtf']:<12.4f} {r['speed']:<10.2f}x"
    )

print(f"\nTotal audio duration: {results[0]['audio_duration']:.2f} seconds")
print(f"Number of samples: {len(samples)}")
