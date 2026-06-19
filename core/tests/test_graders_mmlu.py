"""No-key tests for the MMLU letter grader + tolerant extraction (split-02 8–12)."""

from __future__ import annotations

import pytest

from frugalroute.benchmarks import extract_mmlu_answer, grade, grade_mmlu


def test_answer_line_and_case_insensitive() -> None:
    # 8. "The answer is C" with gold C -> True; lowercase "c" -> True.
    assert grade_mmlu("The answer is C", "C") is True
    assert grade_mmlu("the answer is c", "C") is True


@pytest.mark.parametrize("text", ["(C)", "C.", "C)", " C "])
def test_standalone_fallbacks(text: str) -> None:
    # 9. Parenthesized / punctuated standalone letter -> "C".
    assert extract_mmlu_answer(text) == "C"


def test_wrong_letter_is_false() -> None:
    # 10. Wrong letter grades False.
    assert grade_mmlu("The answer is A", "C") is False


def test_unparseable_is_false() -> None:
    # 11. No letter -> None -> False, no exception.
    assert grade_mmlu("It depends.", "C") is False
    assert extract_mmlu_answer("It depends.") is None


def test_prefers_answer_line_over_stray_letter() -> None:
    # 12. A stray earlier letter must not beat the explicit "answer is" line.
    text = "Option A looks plausible at first. The answer is C."
    assert extract_mmlu_answer(text) == "C"
    assert grade_mmlu(text, "C") is True


def test_distractor_letters_in_working() -> None:
    text = "Felony murder... it could be B or possibly D. The answer is B."
    assert extract_mmlu_answer(text) == "B"


def test_grade_dispatch_mmlu() -> None:
    assert grade("mmlu", "The answer is D", "D") is True
    assert grade("mmlu", "The answer is A", "D") is False


def test_letter_glued_to_word_is_not_matched() -> None:
    # A bare "Apple" must not be read as letter "A".
    assert extract_mmlu_answer("Apple bananas cherries") is None


def test_empty_text_extracts_none() -> None:
    assert extract_mmlu_answer("") is None


def test_none_gold_grades_false() -> None:
    assert grade_mmlu("The answer is C", None) is False
