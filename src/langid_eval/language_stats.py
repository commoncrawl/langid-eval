from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence


def average_lang_stats(
    metric: str, stats: Sequence[LangStats], average: str = "macro"
) -> float:
    """Helper to calculate different averages for language-level statistics"""
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
    """Store information and statistics about a particular gold language label"""

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
        """Precision = true positives / (true positives + false positives)"""
        try:
            return self.true_positive_count / (
                self.true_positive_count + self.false_positive_count
            )
        except ZeroDivisionError:
            return 0.0

    def calc_recall(self) -> float:
        """Calculates recall  = true positives / (true positives + false negatives)"""
        # zero division error would indicate something more fundametally wrong
        return self.true_positive_count / (
            self.true_positive_count + self.false_negative_count
        )

    def calc_f1(self) -> float:
        """Calcualtes F1, returning zero if precision and recall are both zero"""
        try:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        except ZeroDivisionError:
            return 0.0

    def calc_fpr(self) -> float:
        """Calculates false positive rate = false positives / (false positives + true negatives)"""
        try:
            return self.false_positive_count / (
                self.false_positive_count + self.true_negative_count
            )
        except ZeroDivisionError:
            # can happen if only one class and all correct
            return 0.0
