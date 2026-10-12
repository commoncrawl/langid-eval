"""Tests for LangStats: the counts for one gold language and the metrics derived
from them. Just arithmetic -- no langcodes, no tag matching.

The numbers are the real ones for Afrikaans and Dutch in test_evaluator.py.
"""

import pytest

from langid_eval import LangStats

AFR = "afr_Latn_afri1274"


def test_a_new_langstats_starts_empty_with_its_own_counters():
    # Guards against the dataclass trap of sharing one mutable default.
    afr, nld = LangStats(AFR), LangStats("nld_Latn_dutc1256")
    afr.wrong_preds.update(["nl"])

    assert (nld.gold_count, nld.precision, nld.wrong_preds) == (0, 0.0, {})


def test_precision():
    stats = LangStats(AFR, true_positive_count=3, false_positive_count=1)
    assert stats.calc_precision() == 0.75


def test_recall():
    stats = LangStats(AFR, true_positive_count=3, false_negative_count=1)
    assert stats.calc_recall() == 0.75


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    # calc_f1 reads .precision and .recall, not the counts.
    stats = LangStats("nld_Latn_dutc1256", precision=1 / 3, recall=0.5)
    assert stats.calc_f1() == pytest.approx(0.4)


def test_false_positive_rate():
    stats = LangStats(AFR, false_positive_count=1, true_negative_count=5)
    assert stats.calc_fpr() == pytest.approx(1 / 6)


def test_zero_counts_give_zero_rather_than_a_divide_by_zero():
    # Precision: the language was never predicted. F1: precision and recall
    # are both zero. FPR: the evaluation set holds only this one language.
    stats = LangStats(AFR)
    assert (stats.calc_precision(), stats.calc_f1(), stats.calc_fpr()) == (0, 0, 0)


def test_recall_raises_if_the_language_has_no_gold_lines():
    # Deliberate: such a language should never reach this code, so the error
    # surfaces instead of hiding a bug behind a plausible-looking 0.0.
    with pytest.raises(ZeroDivisionError):
        LangStats(AFR).calc_recall()
