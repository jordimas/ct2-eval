import sentencepiece as spm
import ctranslate2
import time
import argparse
from sacrebleu.metrics import BLEU

# ------------------------
# Parse CLI arguments
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    help="Device: cpu, cuda")
parser.add_argument("--eng-file", type=str, default="flores200.eng",
                    help="Path to English source file")
parser.add_argument("--cat-file", type=str, default="flores200.cat",
                    help="Path to Catalan reference file")
args = parser.parse_args()

device = args.device

# ------------------------
# Load tokenizer
# ------------------------
tokenizer = spm.SentencePieceProcessor(model_file="eng-cat/tokenizer/sp_m.model")

# ------------------------
# Read FLORES-200 files
# ------------------------
with open(args.eng_file, "r", encoding="utf-8") as f:
    english_sentences = [line.strip() for line in f if line.strip()]

with open(args.cat_file, "r", encoding="utf-8") as f:
    catalan_references = [line.strip() for line in f if line.strip()]

assert len(english_sentences) == len(catalan_references), \
    f"Mismatch: {len(english_sentences)} English sentences vs {len(catalan_references)} Catalan references"

print(f"Loaded {len(english_sentences)} sentence pairs")
print(f"Selected device: {device}")
print(f"CTranslate2 version: {ctranslate2.__version__}")
print("=" * 60)

# ------------------------
# Initialize BLEU scorer
# ------------------------
bleu = BLEU()

# ------------------------
# Benchmark translations for each compute type
# ------------------------
results = []

for compute_type in sorted(ctranslate2.get_supported_compute_types(device)):
    print(f"\nTesting compute_type: {compute_type}")
    
    translator = ctranslate2.Translator(
        "eng-cat/ctranslate2/",
        compute_type=compute_type,
        device=device,
    )
    
    # Tokenize all English sentences
    tokenized_sentences = [
        tokenizer.encode(sentence, out_type=str) for sentence in english_sentences
    ]
    
    # Count tokens
    total_tokens = sum(len(tokens) for tokens in tokenized_sentences)
    
    # Translate batch
    start_time = time.time()
    translated_batches = translator.translate_batch(tokenized_sentences)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # Decode results
    translations = [tokenizer.decode(t.hypotheses[0]) for t in translated_batches]
    
    # Compute BLEU score
    bleu_score = bleu.corpus_score(translations, [catalan_references])
    
    # Store results
    results.append({
        "compute_type": compute_type,
        "bleu": bleu_score.score,
        "time": elapsed_time,
        "tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_time
    })
    
    print(f"  BLEU score: {bleu_score.score:.2f}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Tokens: {total_tokens}")
    print(f"  Tokens/sec: {total_tokens / elapsed_time:.2f}")
    print(f"  Sample translation: {translations[0][:80]}...")

# ------------------------
# Summary table
# ------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Compute Type':<15} {'BLEU':<10} {'Time (s)':<12} {'Tokens/sec':<12}")
print("-" * 60)
for r in results:
    print(f"{r['compute_type']:<15} {r['bleu']:<10.2f} {r['time']:<12.2f} {r['tokens_per_sec']:<12.2f}")
