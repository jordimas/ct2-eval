import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ctranslate2
from transformers import AutoTokenizer

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
DEVICE = os.environ.get("DEVICE", "cpu")

cpu_count = os.cpu_count()
inter_threads = max(1, cpu_count // 4)
intra_threads = 4

_cache: dict[str, tuple] = {}


def _load(model_id: str):
    if model_id not in _cache:
        ct2 = MODELS_DIR / model_id
        if not (ct2 / "model.bin").exists():
            raise RuntimeError(
                f"Model {model_id} not found at {ct2}. Convert it first."
            )
        logger.info("Loading model %s...", model_id)
        ct2_kwargs = (
            {}
            if DEVICE == "cuda"
            else {"inter_threads": inter_threads, "intra_threads": intra_threads}
        )
        _cache[model_id] = (
            ctranslate2.Generator(
                str(ct2),
                device=DEVICE,
                compute_type="int8" if DEVICE == "cuda" else "int8",
                **ct2_kwargs,
            ),
            AutoTokenizer.from_pretrained(model_id),
        )
        logger.info("Model %s loaded.", model_id)
    return _cache[model_id]


def _end_tokens(tok):
    candidates = ["<eos>", "<end_of_turn>", "<turn|>", "<|end_of_turn|>", "<|eot_id|>"]
    added = tok.get_added_vocab()
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in candidates:
        if t in added:
            ids.add(added[t])
    return [tok.convert_ids_to_tokens([i])[0] for i in ids if i is not None]


def _generate(model_id, token_ids, max_tokens, temperature, top_p, stop):
    gen, tok = _load(model_id)
    tokens = tok.convert_ids_to_tokens(token_ids)
    result = gen.generate_batch(
        [tokens],
        max_length=max_tokens,
        sampling_temperature=max(temperature, 1e-6),
        sampling_topp=top_p,
        include_prompt_in_result=False,
        end_token=_end_tokens(tok),
    )
    text = tok.decode(result[0].sequences_ids[0], skip_special_tokens=True)
    finish = "length"
    for s in stop or []:
        if s in text:
            text, finish = text[: text.index(s)], "stop"
            break
    return text, finish


def chat(model_id, messages, max_tokens, temperature, top_p, stop, reasoning):
    gen, tok = _load(model_id)
    is_qwen = "qwen" in model_id.lower()
    is_gemma4 = "gemma-4" in model_id.lower()
    disable_thinking = reasoning not in ("low", "medium", "high")
    is_gemma = "gemma" in model_id.lower()
    if tok.chat_template is not None:
        extra = (
            {"enable_thinking": False}
            if ((is_qwen or is_gemma4) and disable_thinking)
            else {}
        )
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra
        )
        logger.info("Prompt: extra: %s - %s", extra, prompt)
        token_ids = tok.encode(prompt, add_special_tokens=False)
    elif is_gemma:
        parts = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        prompt = "\n".join(parts)
        logger.info("Prompt (gemma manual template): %s", prompt)
        token_ids = tok.encode(prompt, add_special_tokens=False)
    else:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        token_ids = tok.encode("\n".join(parts))
    return _generate(model_id, token_ids, max_tokens, temperature, top_p, stop)


def main():
    model_id = "google/gemma-4-31b-it"
    messages = [
        {
            "role": "system",
            "content": "Translate the following English text to Catalan. Output only the translation.",
        },
        {
            "role": "user",
            "content": "Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.",
        },
    ]
    t0 = time.time()
    text, finish = chat(
        model_id,
        messages,
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        stop=[],
        reasoning=None,
    )
    elapsed = time.time() - t0
    print(text)
    logger.info("Finish reason: %s | elapsed: %.2fs", finish, elapsed)


if __name__ == "__main__":
    main()
