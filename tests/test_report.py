"""Tests for build_report_dict and format_report.

The averaging maths is tested in test_evaluator.py; these check that the
report puts the right numbers in the right places and lines them up.

Five lines: two Xhosa (both found) and three Zulu (one found, one called
Xhosa, one called something else). Every number in a row is different, and
the three averages differ, so a value in the wrong column or row shows up.
"""

import pytest

from langid_eval import LangStats
from langid_eval.report import build_report_dict, format_report

XHO = LangStats(
    "xho_Latn_xhos1239",
    gold_count=2,
    true_positive_count=2,
    false_positive_count=1,
    false_negative_count=0,
    true_negative_count=2,
    precision=2 / 3,
    recall=1.0,
    f1=0.8,
    fpr=1 / 3,
)
ZUL = LangStats(
    "zul_Latn_zulu1248",
    gold_count=3,
    true_positive_count=1,
    false_positive_count=0,
    false_negative_count=2,
    true_negative_count=2,
    precision=1.0,
    recall=1 / 3,
    f1=0.5,
    fpr=0.0,
)


def test_report_as_dict():
    # Given out of order to check the languages come back sorted.
    report = build_report_dict([ZUL, XHO])

    assert list(report) == [
        "xho_Latn_xhos1239",
        "zul_Latn_zulu1248",
        "macro avg",
        "micro avg",
        "weighted avg",
    ]
    assert report["xho_Latn_xhos1239"] == {
        "precision": 2 / 3,
        "recall": 1.0,
        "f1-score": 0.8,
        "fpr": 1 / 3,
        "support": 2,
    }
    assert report["macro avg"] == pytest.approx(
        {
            "precision": 5 / 6,
            "recall": 2 / 3,
            "f1-score": 0.65,
            "fpr": 1 / 6,
            "support": 5,
        }
    )
    assert report["micro avg"] == pytest.approx(
        {"precision": 0.75, "recall": 0.6, "f1-score": 2 / 3, "fpr": 0.2, "support": 5}
    )
    assert report["weighted avg"] == pytest.approx(
        {
            "precision": 13 / 15,
            "recall": 0.6,
            "f1-score": 0.62,
            "fpr": 2 / 15,
            "support": 5,
        }
    )


def test_printed_report():
    expected = (
        "                   precision    recall  f1-score       fpr   support\n"
        "\n"
        "xho_Latn_xhos1239       0.67      1.00      0.80      0.33         2\n"
        "zul_Latn_zulu1248       1.00      0.33      0.50      0.00         3\n"
        "\n"
        "        macro avg       0.83      0.67      0.65      0.17         5\n"
        "        micro avg       0.75      0.60      0.67      0.20         5\n"
        "     weighted avg       0.87      0.60      0.62      0.13         5\n"
    )

    assert format_report(build_report_dict([ZUL, XHO]), digits=2) == expected


def test_columns_line_up_when_labels_are_shorter_than_weighted_avg():
    # The first column has to fit "weighted avg" as well as the labels.
    zu = LangStats("zu", gold_count=3, true_positive_count=1, false_negative_count=2)

    report = format_report(build_report_dict([zu]), digits=4)

    line_lengths = {len(line) for line in report.splitlines() if line}
    assert len(line_lengths) == 1
