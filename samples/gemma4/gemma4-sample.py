import time
import argparse
from transformers import AutoTokenizer
import ctranslate2

total_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="models/google/gemma-4-31b-it",
    help="Path to the CTranslate2 model directory",
)
parser.add_argument(
    "--tokenizer", default="google/gemma-4-31b-it", help="Tokenizer name or path"
)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)
gen = ctranslate2.Generator(args.model)

messages = [
    {"role": "user", "content": "Generate a 20 word text talking about George Orwell."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokens = tok.convert_ids_to_tokens(tok.encode(prompt))

res = gen.generate_batch(
    [tokens],
    max_length=128,
    sampling_temperature=0.1,
    sampling_topk=1,
    sampling_topp=0.1,
    include_prompt_in_result=False,
#    end_token="<turn|>",
)
print(tok.convert_tokens_to_string(res[0].sequences[0]))
print(f"\nTotal execution time: {time.time() - total_start:.4f} seconds")
