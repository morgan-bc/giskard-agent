"""Tests for the GAIA official question scorer port."""

import math

import pytest

from gaia_scorer import _words_to_number, normalize_number_str, normalize_str, question_scorer


class TestWordsToNumber:
    def test_hyphenated_tens(self):
        assert _words_to_number("forty-one") == 41

    def test_hundreds(self):
        assert _words_to_number("one hundred twenty three") == 123

    def test_with_and(self):
        assert _words_to_number("one hundred and five") == 105

    def test_bare_hundred(self):
        assert _words_to_number("hundred") == 100

    def test_unknown_word_raises(self):
        with pytest.raises(ValueError):
            _words_to_number("banana")


class TestNormalizeNumberStr:
    def test_plain(self):
        assert normalize_number_str("41") == 41.0

    def test_commas(self):
        assert normalize_number_str("1,000") == 1000.0

    def test_dollar(self):
        assert normalize_number_str("$100") == 100.0

    def test_percent(self):
        assert normalize_number_str("50%") == 50.0

    def test_english_words(self):
        assert normalize_number_str("forty-one") == 41.0

    def test_garbage_is_inf(self):
        assert math.isinf(normalize_number_str("banana"))


class TestNormalizeStr:
    def test_lowercase(self):
        assert normalize_str("Seagull") == "seagull"

    def test_all_whitespace_removed(self):
        assert normalize_str("sea gull") == "seagull"

    def test_punct_removed(self):
        assert normalize_str("seagull.") == "seagull"

    def test_keep_punct(self):
        assert normalize_str("a.b", remove_punct=False) == "a.b"


class TestQuestionScorer:
    # number branch
    def test_number_exact(self):
        assert question_scorer("41", "41") == (True, "number")

    def test_number_dollar_comma(self):
        assert question_scorer("$1,000", "1000")[0] is True

    def test_number_wrong(self):
        assert question_scorer("40", "41") == (False, "number")

    def test_number_unrounded_differs(self):
        assert question_scorer("40.4", "41")[0] is False

    def test_number_garbage(self):
        assert question_scorer("forty two", "41")[0] is False

    # string branch
    def test_string_case_punct(self):
        assert question_scorer("seagull.", "seagull") == (True, "string")

    def test_string_whitespace_insensitive(self):
        assert question_scorer("sea gull", "seagull")[0] is True

    def test_string_mismatch(self):
        assert question_scorer("crow", "seagull")[0] is False

    def test_empty_prediction_string_branch(self):
        assert question_scorer("", "seagull")[0] is False

    # list branch
    def test_list_exact(self):
        assert question_scorer("34689, 33063", "34689,33063") == (True, "list")

    def test_list_length_mismatch(self):
        score, detail = question_scorer("34689", "34689,33063")
        assert score is False
        assert detail.startswith("list: length mismatch")

    def test_list_element_mismatch(self):
        assert question_scorer("34689, 33064", "34689,33063")[0] is False

    def test_list_mixed_numeric_string_elements(self):
        assert question_scorer("2, Barack Obama", "2,barack obama")[0] is True

    def test_semicolon_gt_comma_prediction(self):
        # gt uses ';', prediction uses ',' — both split on [;,]
        assert question_scorer("34689,33063", "34689;33063")[0] is True

    # None prediction must not crash
    def test_none_prediction(self):
        assert question_scorer(None, "seagull")[0] is False
