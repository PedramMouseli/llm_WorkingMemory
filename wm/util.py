"""Shared helpers. The chat-template wrapper exists because `enable_thinking`
is not accepted by every tokenizer build; if it is rejected we fall back and
WARN LOUDLY rather than silently generating in the wrong mode."""

from __future__ import annotations

_WARNED = {}


def chat(tok, msgs, thinking: bool = True) -> str:
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        if not _WARNED.get("kw"):
            print("[util] WARNING: tokenizer rejected enable_thinking=%r. "
                  "Falling back to the default template. VERIFY whether the "
                  "model is actually thinking before trusting any result." % thinking)
            _WARNED["kw"] = True
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def assert_thinking(sample_text: str, expect: bool):
    """Call this on a pilot generation. Verifies that the model is generating
    in the intended thinking mode."""
    has = "</think>" in sample_text
    if has != expect:
        raise SystemExit(
            f"[util] thinking mode mismatch: expected thinking={expect}, "
            f"saw </think>={has}. Inspect tok.chat_template before continuing."
        )
    print(f"[util] thinking mode confirmed: {expect}")
