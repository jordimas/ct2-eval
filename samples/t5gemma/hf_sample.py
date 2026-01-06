from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor

# Load the tokenizer and model

model_id = "google/t5gemma-b-b-prefixlm-it"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model = model.eval()

prompts ="Question: Why is the sky blue? Answer:"

model_inputs = processor(text=prompts, return_tensors="pt")

generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False, eos_token_id=[1, 108])

decoded = processor.batch_decode(generation)
print(decoded[0])


