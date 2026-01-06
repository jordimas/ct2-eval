
from transformers import AutoTokenizer
import ctranslate2, time

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gen = ctranslate2.Generator("gemma-3-1b-it.ct2")

prompt = "<start_of_turn>user\nGenerate a 200 word text talking about George Orwell.<end_of_turn>\n<start_of_turn>model\n"
tokens = tok.convert_ids_to_tokens(tok.encode(prompt))

res = gen.generate_batch([tokens], max_length=2048, sampling_temperature=0.1, sampling_topk=1, sampling_topp=0.1, include_prompt_in_result=False)
print(tok.convert_tokens_to_string(res[0].sequences[0]))
