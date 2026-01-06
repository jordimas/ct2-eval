from llama_cpp import Llama

# Load the GGML model (converted for llama.cpp)
# Make sure you have a GGML version of your model, e.g., gemma-3-1b-it.ggmlv3.q4_0.bin
model_path = "/home/jordi/sc/llama/llama.cpp/download/google_gemma-3-4b-it-Q8_0.gguf"
llm = Llama(model_path=model_path)

# Prompt text
prompt = "<start_of_turn>user\nGenera un text en català de 200 paraules que parli d'Antoni Gaudí.<end_of_turn>\n<start_of_turn>model\n"

# Generate text
response = llm(
    prompt,
    max_tokens=2048,
    temperature=0.1,
    top_p=0.1,
    top_k=1,
    echo=False  # Only return the generated text, not the prompt
)

# Print generated text
text = response['choices'][0]['text']
print(text)

