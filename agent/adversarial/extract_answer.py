"""
Shared MCQ answer letter extraction utilities.

Two-step pipeline:
  Step 1: Quick regex — check if last line is a bare letter (A/B/C/D)
  Step 2: LLM extractor — send response to lightweight LLM for extraction
"""

import re
from typing import Optional


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    if "<think>" not in text:
        return text
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def extract_letter_from_response(response: str) -> Optional[str]:
    """Quick regex: extract letter if it's clearly stated.

    Checks last line (bare letter) and common patterns like
    'The answer is B' or '<answer>C</answer>'. Returns None
    if not obvious — let the LLM extractor handle it.
    """
    if not response or not response.strip():
        return None

    text = strip_thinking(response).strip()

    # Last line is a bare letter (most common with our prompt)
    last_line = text.split('\n')[-1].strip()
    m = re.match(r'^([A-D])\.?\s*$', last_line)
    if m:
        return m.group(1)

    # Entire response is a single letter
    if len(text) <= 2:
        m = re.match(r'^([A-D])\.?$', text)
        if m:
            return m.group(1)

    # <answer>B</answer> tag
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text)
    if m:
        return m.group(1)

    return None


def extract_with_confirmation(
    response: str,
    question: str,
    extractor_client=None,
    extractor_model: str = None,
    options: dict = None,
) -> Optional[str]:
    """LLM-based extraction: send response to extractor for letter extraction."""
    # Quick regex first
    letter = extract_letter_from_response(response)
    if letter:
        return letter

    # LLM extractor
    if not extractor_client or not extractor_model:
        return None

    try:
        opts = ""
        if options:
            opts = "\n".join(f"{l}. {options[l]}" for l in ("A", "B", "C", "D")
                             if l in options)
        prompt = (
            f"Question:\n{question}\n{opts}\n\n"
            f"Model response:\n{response[-800:]}\n\n"
            "Which letter (A, B, C, or D) did the model choose? "
            "Reply with ONLY the letter. /no_think"
        )
        resp = extractor_client.chat.completions.create(
            model=extractor_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32, temperature=0.0,
        )
        text = strip_thinking(
            (resp.choices[0].message.content or "").strip()).upper()
        m = re.match(r'^([A-D])', text)
        return m.group(1) if m else None
    except Exception:
        return None
