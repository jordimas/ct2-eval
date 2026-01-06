import ctranslate2
from pathlib import Path

# Path to the converted model
model_path = "llama3-8b"

# Initialize the CTranslate2 model
translator = ctranslate2.Translator(model_path)

# Prompt for generation
prompt = "Write a short poem about space exploration."

# Generate text
results = translator.generate_batch(
    [{"text": prompt}],
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Extract generated text
generated_text = results[0].sequences[0].text
print("Generated Text:\n", generated_text)

