import time
import ctranslate2
import librosa
import transformers

# -------------------------
# Load and resample the audio
# -------------------------
audio, _ = librosa.load("audio/15GdH1.mp3", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# -------------------------

device = "cpu"
for compute_type in ["int8", "float32"]:
    model = ctranslate2.models.Whisper(
        "whisper-medium-ct2", compute_type=compute_type, device=device
    )

    # Detect language
    # -------------------------
    results = model.detect_language(features)
    language, probability = results[0][0]
    #    print("Detected language %s with probability %f" % (language, probability))

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

    results = model.generate(features, [prompt])

    end_time = time.time()
    elapsed = end_time - start_time

    # -------------------------
    # Extract metrics
    # -------------------------
    output_tokens = results[0].sequences_ids[0]
    num_tokens = len(output_tokens)

    tps = num_tokens / elapsed if elapsed > 0 else 0

    # -------------------------
    # Decode transcription
    # -------------------------
    transcription = processor.decode(output_tokens)

    print(f"--- compute_type : {compute_type} -----")
    print("Transcription:")
    print(transcription[0:100])

    print(f"Total output tokens: {num_tokens}")
    print(f"Total time: {elapsed:.3f} seconds")
    print(f"Tokens per second: {tps:.2f} tokens/sec")

print(f"device: {device}")
print(f"ctranslate2: {ctranslate2.__version__}")
