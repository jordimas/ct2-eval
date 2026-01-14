import torchaudio
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_7B", device="cpu")

# Load MP3 with torchaudio and resample to 16kHz
waveform, sample_rate = torchaudio.load("../../release-v2/audio/inaguracio2011.mp3")

# Resample to 16kHz if needed (required by the model)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Pass as audio data dict
audio_data = [{"waveform": waveform.squeeze(0), "sample_rate": 16000}]
transcription = pipeline.transcribe(audio_data, lang=["eng_Latn"], batch_size=1)
print(transcription[0])
