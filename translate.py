import sentencepiece as spm
import ctranslate2
import re
import time

# Load tokenizer and translator
tokenizer = spm.SentencePieceProcessor(model_file="eng-cat/tokenizer/sp_m.model")

# Read file
with open("translation_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into sentences
sentences = re.split(r"(?<=[.!?])\s+", text.strip())

device = "cpu"
for compute_type in sorted(ctranslate2.get_supported_compute_types(device)):

    translator = ctranslate2.Translator(
        "eng-cat/ctranslate2/",
        compute_type=compute_type,
        device=device,
    )
    start_time = time.time()

    total_tokens = 0
    for i in range(0, 10):
        # Tokenize each sentence
        tokenized_sentences = [
            tokenizer.encode(sentence, out_type=str) for sentence in sentences
        ]

        # Count total tokens
        total_tokens += sum(len(tokens) for tokens in tokenized_sentences)

        # Translate batch
        translated_batches = translator.translate_batch(tokenized_sentences)

        # Decode each translated sentence
        translations = [tokenizer.decode(t[0]["tokens"]) for t in translated_batches]

        end_time = time.time()
        elapsed_time = end_time - start_time

    # Join translated sentences
    final_translation = " ".join(translations)

    print(f"--- compute_type : {compute_type} -----")
    print(f"Translation: {final_translation[0:100]}")
    print(f"Time used: {elapsed_time:.2f} seconds")
    print(f"Tokens processed: {total_tokens}")
    print(f"Tokens per second: {total_tokens / elapsed_time:.2f}\n")
print(f"device: {device}")
print(f"ctranslate2: {ctranslate2.__version__}")
