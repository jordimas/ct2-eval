import time
from transformers import AutoTokenizer
import ctranslate2

# Load tokenizer from original model
tokenizer = AutoTokenizer.from_pretrained("meta/llama3-8b")

# Load CTranslate2 generator
generator = ctranslate2.Generator("llama3-8b", device="cpu")

#prompt = "<start_of_turn>user\nQuina és la capital d'Alemania?<end_of_turn>\n<start_of_turn>model\n"

#prompt = "<start_of_turn>user\nGenera un text en català de 200 paraules que parli d'Antoni Gaudí.<end_of_turn>\n<start_of_turn>model\n"
prompt = "Genera un text en català de 200 paraules que parli d'Antoni Gaudí."


# Encode prompt to IDs, then convert to tokens (strings)
token_ids = tokenizer.encode(prompt)
token_strings = tokenizer.convert_ids_to_tokens(token_ids)

batch = [token_strings]

# ------------ Measure inference time -------------
start_time = time.time()

results = generator.generate_batch(
    batch,
    max_length=2048,
    sampling_temperature=0.1,
    sampling_topk=1,
    sampling_topp=0.1,
    include_prompt_in_result=False
)

end_time = time.time()
inference_time = end_time - start_time
# -------------------------------------------------

# Convert generated token strings back to text
generated_tokens = results[0].sequences[0]
text = tokenizer.convert_tokens_to_string(generated_tokens)

print(text)
print(f"\nInference time: {inference_time:.4f} seconds")
