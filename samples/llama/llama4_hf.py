from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model directly on CPU (no 8-bit, no GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",          # Ensures model is on CPU
    torch_dtype=torch.float32  # Optional: can also use torch.float16 for lower memory
)

# Prompt text
prompt = "Genera un text en català de 200 paraules que parli d'Antoni Gaudí."

# Encode prompt to tensor on CPU
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text on CPU
output_ids = model.generate(
    input_ids,
    max_length=2048,
    do_sample=True,
    temperature=0.1,
    top_k=1,
    top_p=0.1,
    pad_token_id=tokenizer.eos_token_id
)

# Decode generated tokens to text
generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print(generated_text)

