# translator_qwen.py
# Caption → English translation using Qwen2.5-7B-Instruct
# Usage env: conda activate translator   (or your VLM env)

from typing import List, Tuple, Optional
import re

from langdetect import detect_langs, LangDetectException
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================================================
# 1. Caption filtering
# ================================================================

MIN_LETTERS = 3          # minimum alphabetic characters
MIN_LANG_PROB = 0.75     # minimum langdetect confidence
MIN_EN_PROB = 0.90       # if English >= 90% confidence → keep as-is

def is_valid_caption(text: str) -> bool:
    """
    Decide if a caption is valid enough to be translated.
    Filters out junk / extremely short / random symbol strings.
    Language-agnostic: only uses unicode alphabetic chars + langdetect.
    """
    if text is None:
        return False

    t = text.strip()
    if not t:
        return False

    # Remove hashtags to reduce noise like "###"
    t_clean = t.replace("#", "").strip()
    if not t_clean:
        return False

    # Count alphabetic characters (works for all unicode scripts)
    letters = [ch for ch in t_clean if ch.isalpha()]
    if len(letters) < MIN_LETTERS:
        # Too few letters → likely garbage
        return False

    # Language detection confidence
    try:
        langs = detect_langs(t_clean)
        best = max(langs, key=lambda l: l.prob)
        if best.prob < MIN_LANG_PROB:
            return False
    except LangDetectException:
        return False

    return True


# ================================================================
# 2. Qwen2.5-7B-Instruct setup
# ================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None

SYSTEM_PROMPT = (
    "You are a professional machine translation engine. "
    "Given any text in any language, you only output its translation into "
    "natural, fluent English. Do NOT explain, do NOT add comments, "
    "do NOT say anything else, just output English words."
)

USER_PROMPT_TEMPLATE = (
    "Translate the following text into English. "
    "Only output the English translation, with no commentary.\n\n"
    "Text:\n{input}"
)


def _load_qwen() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Lazy-load Qwen2.5-7B-Instruct (tokenizer + model), keep in global cache.
    """
    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    dtype = "auto"  # let transformers pick bf16/fp16/fp32 as appropriate

    _TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )

    if DEVICE != "cuda":
        # On pure CPU, make sure model is on CPU
        _MODEL = _MODEL.to("cpu")

    _MODEL.eval()
    return _TOKENIZER, _MODEL


# ================================================================
# 3. Language detection helper (only to detect English)
# ================================================================

def _detect_lang_code(text: str) -> Tuple[str, float]:
    """
    Detect language using langdetect.
    Returns (language code, probability).
    """
    try:
        langs = detect_langs(text)
        best = max(langs, key=lambda l: l.prob)
        return best.lang.lower(), best.prob
    except LangDetectException:
        return "en", 0.0


# ================================================================
# 4. Post-processing: clean Qwen output
# ================================================================

def _clean_translation(raw: str) -> str:
    """
    Clean the raw LLM output and return a single, clean English sentence/phrase.

    Heuristics:
      - strip whitespace
      - drop leading 'assistant:' / 'Assistant:' style prefixes
      - keep the first non-empty line
      - remove wrapping quotes / backticks
    """
    text = raw.strip()

    # Common pattern: "assistant\n\nThis is ...", remove leading 'assistant' line
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]  # drop empty lines

    if not lines:
        return ""

    first = lines[0]

    # Remove leading 'assistant' / 'Assistant' label if present
    first = re.sub(
        r'^(assistant|Assistant|ASSISTANT)\s*[:,\-]*\s*',
        '',
        first
    )

    # If still multiple lines, just take first non-empty after cleaning
    first = first.strip()

    # Remove surrounding backticks / quotes
    # e.g. ```This is ramen.``` or "This is ramen."
    if first.startswith("```") and first.endswith("```"):
        first = first.strip("`").strip()
    if (first.startswith('"') and first.endswith('"')) or \
       (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()

    return first.strip()


# ================================================================
# 5. Single-caption translation
# ================================================================

def translate_to_english(text: str, max_new_tokens: int = 64) -> str:
    """
    Translate a single caption into English using Qwen2.5-7B-Instruct.
    - If invalid caption → return empty string.
    - If high-confidence English → return original text.
    - Else → call Qwen with translation prompt.
    """
    if not is_valid_caption(text):
        return ""

    t = text.strip()
    if not t:
        return ""

    lang_code, prob = _detect_lang_code(t)
    if lang_code == "en" and prob >= MIN_EN_PROB:
        # High-confidence English → keep as-is
        return t

    tokenizer, model = _load_qwen()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(input=t),
        },
    ]

    # Use chat template → single input string
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [chat_text],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Remove the prompt tokens from each output
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
    ]

    decoded = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _clean_translation(decoded)


# ================================================================
# 6. Batch translation
# ================================================================

def translate_batch_to_english(texts: List[str], max_new_tokens: int = 64) -> List[str]:
    """
    Translate a batch of captions into English using Qwen2.5-7B-Instruct.

    Pipeline:
      1. Filter invalid captions → empty string.
      2. High-confidence English → keep as-is.
      3. Other languages → translate with Qwen.
    """
    if not texts:
        return []

    outputs = ["" for _ in texts]

    # Step 1 & 2: decide per caption whether to skip / keep / translate
    indices_to_translate: list[int] = []
    user_prompts: list[str] = []

    for i, text in enumerate(texts):
        if not is_valid_caption(text):
            outputs[i] = ""
            continue

        t = text.strip()
        if not t:
            outputs[i] = ""
            continue

        lang_code, prob = _detect_lang_code(t)
        if lang_code == "en" and prob >= MIN_EN_PROB:
            # High-confidence English → keep
            outputs[i] = t
            continue

        # Need translation
        indices_to_translate.append(i)
        user_prompts.append(
            USER_PROMPT_TEMPLATE.format(input=t)
        )

    if not indices_to_translate:
        return outputs

    tokenizer, model = _load_qwen()

    # Step 3: build chat messages for all items that need translation
    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": up},
        ]
        for up in user_prompts
    ]

    # Convert each chat into a single string via chat template
    chat_texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in messages_batch
    ]

    model_inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Remove prompt part for each sample
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, gen_ids)
    ]

    decoded_list = tokenizer.batch_decode(trimmed, skip_special_tokens=True)

    for idx, raw in zip(indices_to_translate, decoded_list):
        outputs[idx] = _clean_translation(raw)

    return outputs


# ================================================================
# 7. Simple self-test
# ================================================================

if __name__ == "__main__":
    samples = [
        "これはおいしいラーメンです",                      # Japanese
        "这是一碗很好吃的拉面",                          # Chinese
        "Una deliciosa sopa de pollo",                 # Spanish
        "Frisch zubereitet BLT-Sandwich ...",          # German
        "Japan Food Of Grilled Chicken",               # English → should remain unchanged
        "Сосиски от фуде с рисом",                    # Russian
        "辣白菜豆腐汤——迷迭香",
        "美味又可口的石锅拌饭,做法简单到你也会,在家吃省钱又实惠的做法",
    ]

    print("[VALID FLAGS] ", [is_valid_caption(s) for s in samples])
    translated = translate_batch_to_english(samples)
    for src, trg in zip(samples, translated):
        print("-" * 40)
        print("SRC:", src)
        print("TRG:", trg)