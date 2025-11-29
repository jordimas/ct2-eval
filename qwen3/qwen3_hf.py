from transformers import AutoModelForCausalLM, AutoTokenizer
import time

#model_name = "Qwen/Qwen3-4B"
model_name = "Qwen3-4B-Thinking-2507"


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Explica'm quins són els tres monuments més importants d'arquitectura catalana. Màxim 200 paraules."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# ---- measure inference time ----
start_time = time.time()

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

end_time = time.time()
inference_time = end_time - start_time
# --------------------------------

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
print(f"\nInference time: {inference_time:.2f} seconds")

