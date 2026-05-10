from transformers import AutoProcessor, AutoModelForCausalLM

model_name = "google/gemma-4-31B-it"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Genera un text en català de 200 paraules que parli d'Antoni Gaudí."},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=1.0,
    top_k=64,
    top_p=0.95,
)

response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
print(response)
