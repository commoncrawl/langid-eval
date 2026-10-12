"""Tests for tag_match, closest_match_with_caching and LangIdEval.

The labels are real: FLORES+ golds in lang_Script_glottocode form and
predictions from the funlid-rust run. langcodes reads the glottocode as a
variant subtag, so "afr_Latn_afri1274" compares equal to "af".

The languages in CASES have different numbers of lines, so macro, micro and
weighted averages all give different answers.
"""

from statistics import StatisticsError

import pytest
from langcodes import Language

from langid_eval import LangIdEval, closest_match_with_caching, tag_match

CASES = [
    ("afr_Latn_afri1274", "af"),  # hit
    ("afr_Latn_afri1274", "af"),  # hit
    ("afr_Latn_afri1274", "af"),  # hit
    ("afr_Latn_afri1274", "nl"),  # miss; false positive charged to Dutch
    ("nld_Latn_dutc1256", "nl"),  # hit
    ("nld_Latn_dutc1256", "af"),  # miss; false positive charged to Afrikaans
    ("arg_Latn_arag1245", "es"),  # miss; Spanish is no gold language here
    ("arg_Latn_arag1245", "es"),  # miss; ditto
    ("arg_Latn_arag1245", "nl"),  # miss; false positive charged to Dutch
    ("arb_Arab_stan1318", "arz"),  # miss; Egyptian Arabic is 10 from MSA, over 5
]
GOLDS = [gold for gold, _ in CASES]
PREDS = [pred for _, pred in CASES]

# Shared because nothing here mutates them.
EVALUATOR = LangIdEval(GOLDS, PREDS)
SUPPORTED = LangIdEval(GOLDS, PREDS, supported_languages=["af", "nl", "ar"])
STATS = EVALUATOR.stats_lookup


# --- Matching tags ---


@pytest.mark.parametrize(
    "pred, gold, ignore_script, max_distance, matches",
    [
        ("af", "afr_Latn_afri1274", True, 5, True),  # same language, other scheme
        ("nl", "afr_Latn_afri1274", True, 5, False),
        ("arz", "arb_Arab_stan1318", True, 10, True),  # the cutoff is inclusive
        ("arz", "arb_Arab_stan1318", True, 9, False),
        ("ace-Arab", "ace_Latn_achi1257", True, 5, True),
        ("ace-Arab", "ace_Latn_achi1257", False, 5, False),
        # langcodes puts und_Arab at distance 0 from MSA, so without a special
        # case "script known, language not" would count as a hit.
        ("und_Arab", "arb_Arab_stan1318", True, 20, False),
        ("arb_Arab_stan1318", "und_Arab", True, 20, False),
    ],
)
def test_tag_match(pred, gold, ignore_script, max_distance, matches):
    assert tag_match(pred, gold, ignore_script, max_distance) == matches


def closest(tag, cache=None):
    golds = ["afr_Latn_afri1274", "nld_Latn_dutc1256"]
    cache = {} if cache is None else cache
    return closest_match_with_caching(
        tag, golds, cache, max_distance=5, ignore_script=True
    )


def test_closest_match_accepts_a_language_object_and_caches_by_tag():
    cache = {}
    assert closest(Language.get("af"), cache) == ("afr_Latn_afri1274", 0)
    assert cache == {"af": ("afr_Latn_afri1274", 0)}


def test_closest_match_reads_the_cache():
    # A planted wrong answer proves the cache was read, not recomputed.
    assert closest("af", {"af": ("nld_Latn_dutc1256", 3)}) == ("nld_Latn_dutc1256", 3)


def test_closest_match_gives_und_no_match():
    assert closest("und_Latn") == ("und", 1000)


# --- Counting ---


def test_counts_per_language():
    # A false positive goes to the gold language closest to the prediction, not
    # the gold language of its line: both stray "nl"s land on Dutch. "es" and
    # "arz" are close to no gold language, so they land nowhere.
    counts = {
        gold: (
            s.gold_count,
            s.true_positive_count,
            s.false_positive_count,
            s.false_negative_count,
            s.true_negative_count,
        )
        for gold, s in STATS.items()
    }
    assert counts == {
        "afr_Latn_afri1274": (4, 3, 1, 1, 5),
        "nld_Latn_dutc1256": (2, 1, 2, 1, 6),
        "arg_Latn_arag1245": (3, 0, 0, 3, 7),
        "arb_Arab_stan1318": (1, 0, 0, 1, 9),
    }


def test_predictions_are_recorded_against_the_gold_language():
    assert STATS["afr_Latn_afri1274"].matching_preds == {"af": 3}
    assert STATS["arg_Latn_arag1245"].wrong_preds == {"es": 2, "nl": 1}


@pytest.mark.parametrize(
    "gold, pred, settings, hit",
    [
        ("arb_Arab_stan1318", "arz", {}, False),  # default distance is 5
        ("arb_Arab_stan1318", "arz", {"max_tag_distance": 10}, True),
        ("ace_Latn_achi1257", "ace-Arab", {}, True),  # scripts ignored by default
        ("ace_Latn_achi1257", "ace-Arab", {"ignore_script": False}, False),
    ],
)
def test_matching_settings_are_used_when_counting(gold, pred, settings, hit):
    stats = LangIdEval([gold], [pred], **settings).stats_lookup[gold]
    assert stats.true_positive_count == hit


def test_an_und_prediction_is_a_miss_charged_to_nobody():
    gold = "arb_Arab_stan1318"
    stats = LangIdEval([gold], ["und_Arab"]).stats_lookup[gold]
    assert (stats.true_positive_count, stats.false_positive_count) == (0, 0)


# --- Metrics ---


def test_metrics_per_language():
    assert EVALUATOR.precision(by_class=True) == pytest.approx(
        {
            "afr_Latn_afri1274": 0.75,
            "nld_Latn_dutc1256": 1 / 3,
            "arg_Latn_arag1245": 0.0,
            "arb_Arab_stan1318": 0.0,
        }
    )
    assert EVALUATOR.recall(by_class=True)["afr_Latn_afri1274"] == 0.75
    assert EVALUATOR.fpr(by_class=True)["afr_Latn_afri1274"] == pytest.approx(1 / 6)
    # Aragonese gets nothing right, so F1 is 0.0 rather than a divide by zero.
    assert EVALUATOR.f1(by_class=True)["arg_Latn_arag1245"] == 0.0


def test_the_three_averages_differ():
    # macro weights every language equally, weighted by its gold lines, and
    # micro pools the raw counts.
    assert EVALUATOR.precision() == pytest.approx(13 / 48)  # macro is the default
    assert EVALUATOR.precision(average="weighted") == pytest.approx(11 / 30)
    assert EVALUATOR.precision(average="micro") == pytest.approx(4 / 7)


def test_micro_recomputes_each_metric_from_the_pooled_counts():
    assert EVALUATOR.recall(average="micro") == 0.4
    assert EVALUATOR.f1(average="micro") == pytest.approx(8 / 17)
    assert EVALUATOR.fpr(average="micro") == pytest.approx(0.1)


def test_by_class_ignores_the_average():
    by_class = EVALUATOR.recall(by_class=True)
    assert EVALUATOR.recall(by_class=True, average="micro") == by_class


def test_an_unknown_average_is_rejected():
    with pytest.raises(NotImplementedError):
        EVALUATOR.precision(average="median")


# --- Restricting to the languages a model claims to support ---


def test_unsupported_gold_languages_are_dropped():
    # A model claiming Afrikaans, Dutch and Arabic is not marked down for Aragonese.
    assert sorted(SUPPORTED.supported_stats_lookup) == [
        "afr_Latn_afri1274",
        "arb_Arab_stan1318",
        "nld_Latn_dutc1256",
    ]
    assert SUPPORTED.precision(supported_languages_only=True) == pytest.approx(5 / 12)
    report = SUPPORTED.classification_report(supported_languages_only=True)
    assert "arg_Latn_arag1245" not in report


def test_supported_languages_only_needs_a_list_of_supported_languages():
    with pytest.raises(ValueError):
        EVALUATOR.precision(supported_languages_only=True)
    with pytest.raises(ValueError):
        EVALUATOR.classification_report(supported_languages_only=True)


# --- Report and bad input ---


def test_classification_report_as_text_or_dict():
    row = "afr_Latn_afri1274       0.75      0.75      0.75      0.17         4\n"
    assert row in EVALUATOR.classification_report(digits=2)
    report = EVALUATOR.classification_report(output_dict=True)
    assert report["afr_Latn_afri1274"]["support"] == 4


def test_mismatched_numbers_of_golds_and_predictions_are_rejected():
    with pytest.raises(ValueError):
        LangIdEval(GOLDS, PREDS[:-1])


def test_no_data_at_all_raises_rather_than_reporting_a_score():
    # StatisticsError is whatever fmean happens to raise, not a deliberate
    # choice. Worth replacing with a clear error, at which point change this.
    with pytest.raises(StatisticsError):
        LangIdEval([], []).precision()
