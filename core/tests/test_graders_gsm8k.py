"""No-key tests for the GSM8K numeric grader + tolerant extraction (split-02 1–7, R10)."""

from __future__ import annotations

import pytest

from frugalroute.benchmarks import _parse_number, extract_gsm8k_answer, grade, grade_gsm8k


def test_clean_answer_line() -> None:
    # 1. Clean "The answer is N." line.
    assert grade_gsm8k("Working... The answer is 72.", 72) is True


@pytest.mark.parametrize(
    "text", ["The answer is $1,250", "The answer is 1250.0", "$1,250", "1,250"]
)
def test_formatting_tolerance(text: str) -> None:
    # 2. $, thousands commas, trailing .0 all equal 1250.
    assert grade_gsm8k(text, 1250) is True


def test_fallback_to_last_number() -> None:
    # 3. No "answer is" line -> use the last number in the text.
    assert grade_gsm8k("first 24 then double, so total 72", 72) is True


def test_wrong_answer_is_false() -> None:
    # 4. A parseable but wrong answer grades False.
    assert grade_gsm8k("The answer is 73.", 72) is False


def test_unparseable_is_false_no_exception() -> None:
    # 5. Garbage -> False, never an exception.
    assert grade_gsm8k("I cannot solve this", 72) is False


def test_trailing_noise_tolerance() -> None:
    # 6. Trailing words after the number.
    assert grade_gsm8k("The answer is 72 clips.", 72) is True


def test_float_gold() -> None:
    # 7. Float gold vs integer-style answer and vice versa.
    assert grade_gsm8k("The answer is 10.0", 10) is True
    assert grade_gsm8k("The answer is 10", 10.0) is True


def test_prefers_answer_line_over_working() -> None:
    # The marker number wins over earlier numbers in the working.
    assert extract_gsm8k_answer("48 + 24 steps... The answer is 72.") == 72.0


def test_scientific_notation() -> None:
    assert grade_gsm8k("The answer is 1e3", 1000) is True


# --- R10 adversarial table: each must grade to a definite, stable bool. ---
HOSTILE = [
    "",  # empty
    "abcdefg",  # only letters
    "3 5 7 9 11",  # multiple numbers, no marker
    "The answer is 1.5e2",  # scientific notation -> 150
    "７２",  # full-width unicode digits
    "   \n\t  ",  # only whitespace
    "The answer is .",  # marker but no number
]


@pytest.mark.parametrize("text", HOSTILE)
def test_adversarial_inputs_grade_deterministically(text: str) -> None:
    first = grade_gsm8k(text, 150)
    second = grade_gsm8k(text, 150)
    assert isinstance(first, bool)
    assert first == second  # deterministic


def test_adversarial_multiple_numbers_takes_last() -> None:
    # "3 5 7 9 11" -> last number 11.
    assert extract_gsm8k_answer("3 5 7 9 11") == 11.0


def test_adversarial_empty_extracts_none() -> None:
    assert extract_gsm8k_answer("") is None
    assert extract_gsm8k_answer("no digits here") is None


def test_non_numeric_gold_grades_false() -> None:
    # A gold that can't be coerced to a number -> False (never raises).
    assert grade_gsm8k("The answer is 5", "not-a-number") is False


def test_parse_number_rejects_bare_punctuation() -> None:
    # The normalizer's failure path: a token float() can't parse -> None.
    assert _parse_number("+") is None
    assert _parse_number("$,") is None
    assert _parse_number("12") == 12.0


def test_grade_dispatch_gsm8k_and_unknown() -> None:
    assert grade("gsm8k", "The answer is 72.", 72) is True
    with pytest.raises(ValueError, match="Unknown benchmark"):
        grade("trivia", "x", 1)
