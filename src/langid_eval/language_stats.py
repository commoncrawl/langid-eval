"""Calculate evaluation statistics for gold language classes."""

from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence


def average_lang_stats(
    metric: str, stats: Sequence[LangStats], average: str = "macro"
) -> float:
    """Calculate various averages for language-level statistics.

    Parameters
    ----------
    metric : str
        Name of the metric to average ("precision", "recall", "f1", or "fpr").
        Must match a LangStats attribute; for micro averaging, only these four
        are supported.
    stats : Sequence[LangStats]
        A sequence of LangStats objects with information about gold language
        labels.
    average : str, optional
        Type of average to calculate ("macro", "micro", "weighted"). Defaults
        to "macro".

    Returns
    -------
    float
        The requested type of average for the requested metric.

    Raises
    ------
    NotImplementedError
        If an unsupported type of average is given.
    """
    if average == "macro":
        return fmean([getattr(s, metric) for s in stats])
    elif average == "weighted":
        total = sum(s.gold_count for s in stats)  # to cover reporting on a subset
        return sum([getattr(s, metric) * s.gold_count for s in stats]) / total
    elif average == "micro":
        # micro requires recalculating totals because each result counts equally
        tp = sum(s.true_positive_count for s in stats)
        fp = sum(s.false_positive_count for s in stats)
        fn = sum(s.false_negative_count for s in stats)
        tn = sum(s.true_negative_count for s in stats)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn)
        values = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall) / (precision + recall)
            if precision + recall
            else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        }
        return values[metric]
    else:
        raise NotImplementedError("Supported averages: {'macro', 'micro', 'weighted'}")


@dataclass
class LangStats:
    """Counts and metrics for a single gold language label.

    Counts are filled in by the evaluator, then metrics are set with the
    `calc_*` methods. All counts and metrics default to zero until then.

    Attributes
    ----------
    gold_label : str
        The gold language label.
    gold_count : int
        Number of items with the gold label (the support).
    true_positive_count : int
        Count of items where the predicted label matched this gold label.
    true_negative_count : int
        Count of items with a different gold label that were not counted as
        false positives for this gold label.
    false_positive_count : int
        Count of wrong predictions on other items whose closest gold label
        (within the evaluator's `max_tag_distance`) is this gold label.
    false_negative_count : int
        Count of items with this gold label but a different predicted label.
    matching_preds : Counter[str]
        Predicted labels that matched this gold label (may differ from
        `gold_label`).
    wrong_preds : Counter[str]
        Non-matching predicted labels for items with this gold label.
    precision : float
        Language class precision (see `calc_precision`).
    recall : float
        Language class recall (see `calc_recall`).
    f1 : float
        Language class F1 score (see `calc_f1`).
    fpr : float
        Language class false positive rate (see `calc_fpr`).
    """

    gold_label: str
    gold_count: int = field(default=0)
    true_positive_count: int = field(default=0)
    true_negative_count: int = field(default=0)
    false_positive_count: int = field(default=0)
    false_negative_count: int = field(default=0)
    matching_preds: Counter[str] = field(default_factory=Counter)
    wrong_preds: Counter[str] = field(default_factory=Counter)
    precision: float = field(default=0.0)
    recall: float = field(default=0.0)
    f1: float = field(default=0.0)
    fpr: float = field(default=0.0)

    def calc_precision(self) -> float:
        """Calculate precision = true positives / (true positives + false positives)."""
        try:
            return self.true_positive_count / (
                self.true_positive_count + self.false_positive_count
            )
        except ZeroDivisionError:
            return 0.0

    def calc_recall(self) -> float:
        """Calculate recall = true positives / (true positives + false negatives).

        Raises
        ------
        ZeroDivisionError
            If `gold_count` is zero.
        """
        # zero division error would indicate something more fundamentally wrong
        return self.true_positive_count / (
            self.true_positive_count + self.false_negative_count
        )

    def calc_f1(self) -> float:
        """Calculate F1, returning zero if precision and recall are both zero."""
        try:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        except ZeroDivisionError:
            return 0.0

    def calc_fpr(self) -> float:
        """Calculate false positive rate = FP / (FP + TN)."""
        try:
            return self.false_positive_count / (
                self.false_positive_count + self.true_negative_count
            )
        except ZeroDivisionError:
            # can happen if only one class and all correct
            return 0.0
