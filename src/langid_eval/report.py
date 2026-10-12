from typing import Sequence

from .language_stats import LangStats, average_lang_stats

AVERAGES = ["macro", "micro", "weighted"]
AVG_KEYS = [f"{a} avg" for a in AVERAGES]


def build_report_dict(stats: Sequence[LangStats]) -> dict[str, dict[str, float]]:
    """Build a report showing the main metrics per class"""
    stats = sorted(stats, key=lambda s: s.gold_label)
    num_items = sum(s.gold_count for s in stats)
    # add lang-level stats to report
    report = {
        entry.gold_label: {
            "precision": entry.precision,
            "recall": entry.recall,
            "f1-score": entry.f1,
            "fpr": entry.fpr,
            "support": entry.gold_count,
        }
        for entry in stats
    }
    # add overall averages to report
    for avg, key in zip(AVERAGES, AVG_KEYS):
        report[key] = {
            "precision": average_lang_stats("precision", stats, avg),
            "recall": average_lang_stats("recall", stats, avg),
            "f1-score": average_lang_stats("f1", stats, avg),
            "fpr": average_lang_stats("fpr", stats, avg),
            "support": num_items,
        }
    return report


def format_report(report_dict: dict[str, dict[str, float]], digits: int) -> str:
    """Create printable report, sklearn-style"""
    # get the format right
    headers = ["precision", "recall", "f1-score", "fpr", "support"]
    longest_last_line_heading = "weighted avg"
    gold_label_set = [x for x in report_dict if x not in AVG_KEYS]
    name_width = max(len(cn) for cn in gold_label_set)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 4 + " {:>9}\n"
    # retrieve language stats from the report dict
    for key in gold_label_set:
        row = (
            key,
            report_dict[key]["precision"],
            report_dict[key]["recall"],
            report_dict[key]["f1-score"],
            report_dict[key]["fpr"],
            report_dict[key]["support"],
        )
        report += row_fmt.format(*row, width=width, digits=digits)
    report += "\n"
    # add on averages
    for key in AVG_KEYS:
        avg = (
            report_dict[key]["precision"],
            report_dict[key]["recall"],
            report_dict[key]["f1-score"],
            report_dict[key]["fpr"],
            report_dict[key]["support"],
        )
        report += row_fmt.format(key, *avg, width=width, digits=digits)

    return report
