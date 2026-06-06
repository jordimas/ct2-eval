import time
import ctranslate2
import librosa
import transformers
import numpy as np

# -------------------------
# Load and resample the audio
# -------------------------
audio, _ = librosa.load("inaguracio2011.mp3", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# -------------------------

device = "cuda"
for compute_type in sorted(ctranslate2.get_supported_compute_types(device)):
    model = ctranslate2.models.Whisper(
        "whisper-tiny-ct2", compute_type=compute_type, device=device
    )

    # Detect language
    results = model.detect_language(features)
    language, probability = results[0][0]

    prompt = processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            language,
            "<|transcribe|>",
            "<|notimestamps|>",
        ]
    )

    # -------------------------
    # Run transcription + timing
    # -------------------------
    start_time = time.time()

    num_tokens = 0
    tps_values = []
    for i in range(0, 10):

        iter_start = time.time()
        results = model.generate(features, [prompt])
        iter_end = time.time()

        end_time = time.time()
        output_tokens = results[0].sequences_ids[0]
        num_tokens += len(output_tokens)

        # 1 iteraction
        iter_tokens = len(output_tokens)
        iter_elapsed = iter_end - iter_start
        iter_tps = iter_tokens / iter_elapsed if iter_elapsed > 0 else 0
        tps_values.append(iter_tps)


    elapsed = end_time - start_time
    tps = num_tokens / elapsed if elapsed > 0 else 0
    tps_std = np.std(tps_values)

    # -------------------------
    # Decode transcription
    # -------------------------
    transcription = processor.decode(output_tokens)

    print(f"--- compute_type : {compute_type} -----")
    print(f"Transcription: {transcription[0:100]}")
    print(f"Total output tokens: {num_tokens}")
    print(f"Total time: {elapsed:.3f} seconds")
    print(f"Tokens per second: {tps:.2f} ± {tps_std:.2f} tokens/sec\n")

print(f"device: {device}")
print(f"ctranslate2: {ctranslate2.__version__}")
