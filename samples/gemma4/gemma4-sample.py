
import time
import argparse
from transformers import AutoTokenizer
import ctranslate2

total_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to the CTranslate2 model directory")
parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)
gen = ctranslate2.Generator(args.model)

prompt = "<start_of_turn>user\nGenerate a 20 word text talking about George Orwell.<end_of_turn>\n<start_of_turn>model\n"
tokens = tok.convert_ids_to_tokens(tok.encode(prompt))

res = gen.generate_batch([tokens], max_length=128, sampling_temperature=0.1, sampling_topk=1, sampling_topp=0.1, include_prompt_in_result=False)
print(tok.convert_tokens_to_string(res[0].sequences[0]))
print(f"\nTotal execution time: {time.time() - total_start:.4f} seconds")
