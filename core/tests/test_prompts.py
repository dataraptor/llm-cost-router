"""No-key tests for the canonical prompts (split-01 cases 17–20)."""

from __future__ import annotations

from frugalroute.prompts import GATE_SYSTEM, GEN_SYSTEM, PROMPT_VERSION, gate_user


def test_prompt_version_is_non_empty_str() -> None:
    assert isinstance(PROMPT_VERSION, str)
    assert PROMPT_VERSION


def test_gate_system_has_load_bearing_phrases() -> None:
    # Sanity that the §6 text is present verbatim, not paraphrased.
    assert "strict answer-quality gate" in GATE_SYSTEM
    assert "sufficient = false" in GATE_SYSTEM
    assert "confidence: your probability (0.0-1.0)" in GATE_SYSTEM


def test_gate_user_format() -> None:
    assert gate_user("Q", "A") == "QUESTION:\nQ\n\nANSWER:\nA"


def test_generation_prompts_end_with_answer_format_line() -> None:
    assert GEN_SYSTEM["gsm8k"].endswith('end with exactly one line: "The answer is <number>."')
    assert GEN_SYSTEM["mmlu"].endswith('Respond with exactly one line: "The answer is <A|B|C|D>."')
