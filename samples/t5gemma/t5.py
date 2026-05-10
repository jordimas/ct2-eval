import argparse
import ctranslate2
import transformers

def main():
    parser = argparse.ArgumentParser(description="Translate text using CTranslate2 and HuggingFace tokenizer")
    parser.add_argument("--ct2-model", required=True, help="Path to CTranslate2 model directory")
    parser.add_argument("--hf-model", required=True, help="HuggingFace tokenizer model name or path")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, auto)")
    args = parser.parse_args()

    translator = ctranslate2.Translator(args.ct2_model, device=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_model)
    
    sentences = ["Question: Why is the sky blue? Answer:"]
    
    # Tokenize each sentence
    tokenized_sentences = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence)) 
        for sentence in sentences
    ]
    
    print(tokenized_sentences)
    
    # Translate batch
    translated_batches = translator.translate_batch(
        tokenized_sentences,
        beam_size=1,
        repetition_penalty=1.2,
        max_decoding_length=50
    )
    
    # Decode results
    translations = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(t.hypotheses[0])) 
        for t in translated_batches
    ]
    
    final_translation = " ".join(translations)
    print(final_translation)

if __name__ == "__main__":
    main()
