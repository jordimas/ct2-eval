from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model from Hugging Face
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to CPU (or GPU if available)
device = "cpu"
model.to(device)

# Prompt text
#prompt = "<start_of_turn>user\nHello, how are you today?<end_of_turn>\n<start_of_turn>model\n"
prompt = "<start_of_turn>user\nGenera un text en català de 200 paraules que parli d'Antoni Gaudí.<end_of_turn>\n<start_of_turn>model\n"

# Encode prompt to tensor
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate text
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
