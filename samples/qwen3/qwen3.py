import time
import argparse
from transformers import AutoTokenizer
import ctranslate2
import os
import psutil  # new import

# ------------------ CLI Arguments ------------------
parser = argparse.ArgumentParser(description="Run CTranslate2 generation with a specified model")
parser.add_argument("--hf_model", type=str, help="Hugging Face model name (e.g., Qwen/Qwen3-4B)", default="Qwen/Qwen3-4B")
parser.add_argument("--ct2_model", type=str, help="CTranslate2 model path (e.g., Qwen3-4B.ct2/)", default="Qwen3_4B.ct2")
args = parser.parse_args()
# ---------------------------------------------------

# Load tokenizer from original model
tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

# Load CTranslate2 generator
generator = ctranslate2.Generator(args.ct2_model, device="cpu")

prompt = """<|im_start|>user
Explica'm en 300 paraules com a màxim:
- que és Softcatalà.
- els 3 projectes principals
- els 3 col·laboradors principals
<|im_end|>
<|im_start|>assistant
"""

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
print(f"Tokenizer: {args.hf_model}")
print(f"CT2: {args.ct2_model}")

# ---------------- Model size ----------------
model_bin_path = os.path.join(args.ct2_model, "model.bin")
if os.path.exists(model_bin_path):
    size_bytes = os.path.getsize(model_bin_path)
    size_gb = size_bytes / (1024 ** 3)
    print(f"Model size on disk: {size_gb:.2f} GB")
else:
    print("model.bin not found in the CT2 directory")

# ---------------- Memory usage ----------------
process = psutil.Process(os.getpid())
mem_bytes = process.memory_info().rss
mem_gb = mem_bytes / (1024 ** 3)
print(f"Memory used by Python process: {mem_gb:.2f} GB")

