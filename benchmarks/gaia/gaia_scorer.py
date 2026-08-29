"""GAIA official question scorer port.

Ported from the official GAIA ``evaluation.py`` (``question_scorer``), with one
addition: when a numeric ground truth cannot be matched because the model answer
is an English number word (e.g. "forty-one"), a lightweight built-in 0-999
word-to-number converter is tried before giving up (returning ``inf``, which
always scores False). This keeps the module dependency-free.
"""

from __future__ import annotations

import re
import string

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(text: str) -> int:
    """Convert an English number word (0-999, e.g. "forty-one") to an int.

    Raises:
        ValueError: If the text is not a supported English number word.
    """
    words = [w for w in re.split(r"[\s-]+", text.strip().lower()) if w and w != "and"]
    if not words:
        raise ValueError(f"not a number word: {text!r}")
    total = 0
    pending = 0  # accumulates ones/tens before a "hundred"
    for word in words:
        if word in _ONES:
            pending += _ONES[word]
        elif word in _TENS:
            pending += _TENS[word]
        elif word == "hundred":
            total += max(pending, 1) * 100
            pending = 0
        else:
            raise ValueError(f"unknown number word {word!r} in {text!r}")
    return total + pending


def normalize_number_str(number_str: str) -> float:
    """Normalize a number string ($, %, commas removed) to a float.

    Falls back to the built-in English word converter (0-999) when the string
    is not numeric; returns ``inf`` when conversion is impossible (an ``inf``
    comparison is always False, matching official behavior).
    """
    for char in ("$", "%", ","):
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        pass
    try:
        return float(_words_to_number(number_str))
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    """Split on any of the given separator characters (default ',' and ';')."""
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalize a string: remove all whitespace, optionally punctuation, lowercase."""
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _is_float(element: str | None) -> bool:
    try:
        float(element)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def question_scorer(model_answer: str | None, ground_truth: str) -> tuple[bool, str]:
    """Score a model answer against the ground truth (official GAIA rules).

    Three branches, mirroring the official scorer:

    1. Numeric ground truth: prediction is stripped of ``$``/``%``/``,``, and
       compared as a float (English number words supported via the fallback).
    2. List ground truth (contains ``,`` or ``;``): both sides are split on
       ``[;,]``; element counts must match and every element must match
       (numeric elements compare as floats, string elements keep punctuation).
    3. String ground truth: both sides are normalized (all whitespace removed,
       punctuation removed, lowercased) and compared for equality.

    Returns:
        ``(score, detail)`` where ``detail`` names the branch taken.
    """
    model_answer = model_answer or ""

    if _is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth), "number"

    if any(char in ground_truth for char in (",", ";")):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False, f"list: length mismatch ({len(ma_elems)} vs {len(gt_elems)})"
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                if normalize_number_str(ma_elem) != float(gt_elem):
                    return False, "list: numeric element mismatch"
            elif normalize_str(ma_elem, remove_punct=False) != normalize_str(
                gt_elem, remove_punct=False
            ):
                return False, "list: string element mismatch"
        return True, "list"

    score = normalize_str(model_answer) == normalize_str(ground_truth)
    return score, "string"
