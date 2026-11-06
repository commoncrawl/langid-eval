"""Define matches and provide language identification evaluation."""

from functools import lru_cache
from langcodes import tag_distance, closest_match, Language
from typing import List, Literal, Sequence, Tuple, overload

from .language_stats import LangStats, average_lang_stats
from .report import build_report_dict, format_report


@lru_cache(maxsize=None)  # fine to have global cache here
def tag_match(
    desired: str, available: str, ignore_script: bool, max_tag_distance: int
) -> bool:
    """Check if two language tags are close enough to count as the same language.

    Parameters
    ----------
    desired : str
        The language which needs a match.
    available : str
        The language proposed as a match.
    ignore_script : bool
        If True, ignore script subtags when matching.
    max_tag_distance : int
        The maximum distance between languages allowed for matches.

    Returns
    -------
    bool
        True if the tags are within `max_tag_distance` of each other. Always
        False if only one of them is undetermined (`und`).
    """
    # handle case of `und_<Script>` being expanded to most common language for that
    # script, which is then marked as correct
    if desired.startswith("und") != available.startswith("und"):
        return False
    return tag_distance(desired, available, ignore_script) <= max_tag_distance


def closest_match_with_caching(
    desired_language: str | Language,
    supported_languages: Sequence[str],
    cache: dict[str, Tuple[str, int]],
    max_distance: int,
    ignore_script: bool,
) -> Tuple[str, int]:
    """Calculate `closest_match` from `langcodes`, storing prior results in `cache`.

    Parameters
    ----------
    desired_language : str | Language
        The language which needs a match.
    supported_languages : Sequence[str]
        The languages available as matches.
    cache : dict[str, Tuple[str, int]]
        A cache to store pre-computed matches and their distances (changed in place).
    max_distance : int
        The maximum distance between languages allowed for matches.
    ignore_script : bool
        If True, ignore script subtags when matching.

    Returns
    -------
    Tuple[str, int]
        The closest match and distance or ("und", 1000) if no match.
    """
    if not isinstance(desired_language, str):
        desired_language = str(desired_language)  # in case it's Language
    if desired_language.startswith("und"):
        return ("und", 1000)  # assume "und" is never in the gold
    if desired_language in cache:
        return cache[desired_language]
    else:
        # note cache may contain ('und', 1000) if no match
        cache[desired_language] = closest_match(
            desired_language,
            supported_languages,
            max_distance,
            ignore_script,
        )
        return cache[desired_language]


class LangIdEval:
    """Evaluate language identification with fuzzy matching of language tags.

    Tags match if their `langcodes` tag distance is at most `max_tag_distance`.

    Parameters
    ----------
    golds : List[str]
        Gold language labels.
    preds : List[str]
        Predicted language labels.
    ignore_script : bool, optional
        Ignore script subtags when matching. Defaults to True.
    max_tag_distance : int, optional
        Maximum tag distance for a match. Defaults to 5 ("a significant but
        unproblematic regional difference").
    supported_languages : Sequence[str], optional
        Languages the model supports. If given, metrics can be calculated for
        these languages only with `supported_languages_only=True`. Defaults to ().

    Raises
    ------
    ValueError
        If `golds` and `preds` have different lengths.
    """

    def __init__(
        self,
        golds: List[str],
        preds: List[str],
        ignore_script: bool = True,
        # from langcodes: "a significant but unproblematic regional difference"
        max_tag_distance: int = 5,
        supported_languages: Sequence[str] = (),
    ) -> None:
        self.ignore_script = ignore_script
        self.max_tag_distance = max_tag_distance
        self.supported_languages = supported_languages
        # calculate both full and filtered lookups because need separate false positives
        self.stats_lookup = self._build_stats_lookup(golds, preds)
        self.supported_stats_lookup = None
        if self.supported_languages:
            supported_golds, supported_preds = self._filter_to_supported(golds, preds)
            self.supported_stats_lookup = self._build_stats_lookup(
                supported_golds, supported_preds
            )

    def _build_stats_lookup(
        self, golds: List[str], preds: List[str]
    ) -> dict[str, LangStats]:
        """Build look-up for language-level statistics from gold and predicted labels.

        Parameters
        ----------
        golds : List[str]
            List of gold language labels.
        preds : List[str]
            List of predicted language labels.

        Returns
        -------
        dict[str, LangStats]
            Per-language statistics keyed by gold label.
        """
        # initialise lookup with gold labels as keys
        lookup: dict[str, LangStats] = dict()
        for g in golds:
            # increment gold count, initialising label entry if absent
            if g not in lookup:
                lookup[g] = LangStats(g)
            lookup[g].gold_count += 1
        gold_labels = sorted(lookup.keys())
        # lookup is initialised, now counts can be calculated
        # local to this call, unlike lru_cache
        closest_match_cache: dict[str, Tuple[str, int]] = dict()
        for g, p in zip(golds, preds, strict=True):
            # true positive: the tags are close enough to match
            if tag_match(p, g, self.ignore_script, self.max_tag_distance):
                lookup[g].true_positive_count += 1
                lookup[g].matching_preds.update([p])

            # false positive: find gold closest to prediction, up to a max tag distance
            # this approach relies heavily on langcodes' idea of closeness
            else:
                closest_gold, dist = closest_match_with_caching(
                    p,
                    gold_labels,
                    closest_match_cache,
                    max_distance=self.max_tag_distance,
                    ignore_script=self.ignore_script,
                )
                if dist <= self.max_tag_distance:
                    lookup[closest_gold].false_positive_count += 1
                lookup[g].wrong_preds.update([p])

        # iterate over stats for each gold class to calculate true and false negatives
        n_items = len(golds)
        for stats in lookup.values():
            # true negatives:
            # (total overall) - (total in class) - (times it shouldn't have predicted)
            stats.true_negative_count = (
                n_items - stats.gold_count - stats.false_positive_count
            )
            # false negatives: (total in class) - (times model was right)
            stats.false_negative_count = stats.gold_count - stats.true_positive_count

        # counts are done, now update each lookup entry with summary statistics
        for stats in lookup.values():
            stats.precision = stats.calc_precision()
            stats.recall = stats.calc_recall()
            stats.f1 = stats.calc_f1()
            stats.fpr = stats.calc_fpr()
        return lookup

    def _filter_to_supported(
        self, golds: List[str], preds: List[str]
    ) -> tuple[List[str], List[str]]:
        """Filter results to those where the gold is close to a supported language.

        Parameters
        ----------
        golds : List[str]
            List of gold language labels.
        preds : List[str]
            List of predicted language labels.

        Returns
        -------
        tuple[List[str], List[str]]
            Equal-length lists of filtered gold and predicted labels.
        """
        filtered_golds, filtered_preds = [], []
        # local to this call, unlike lru_cache
        closest_match_cache: dict[str, Tuple[str, int]] = dict()
        for g, p in zip(golds, preds, strict=True):
            # logic here: we only want gold labels with a supported match
            closest_supported, dist = closest_match_with_caching(
                g,
                self.supported_languages,
                closest_match_cache,
                max_distance=self.max_tag_distance,
                ignore_script=self.ignore_script,
            )
            if dist <= self.max_tag_distance:
                filtered_golds.append(g)
                filtered_preds.append(p)
        assert len(filtered_golds) == len(filtered_preds)  # I am paranoid
        return filtered_golds, filtered_preds

    def _compute_metric(
        self,
        metric: str,
        average: str,
        by_class: bool,
        supported_languages_only: bool,
    ) -> dict[str, float] | float:
        """Resolve the metric to return.

        Parameters
        ----------
        metric : str
            One of "precision", "recall", "f1" or "fpr".
        average : str
            Type of average ("macro", "micro", or "weighted").
        by_class : bool
            Give metrics by class rather than overall.
        supported_languages_only : bool
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction.

        Returns
        -------
        dict[str, float] | float
            Metric by class as a dict, or metric calculated overall.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        NotImplementedError
            If `by_class` is False and `average` is not "macro", "micro" or
            "weighted".
        """
        # returning stats by class means that no average need be calculated
        if supported_languages_only:
            if self.supported_stats_lookup is None:
                raise ValueError(
                    "No supported_languages given when constructing LangIdEval()"
                )
            stats = list(self.supported_stats_lookup.values())
        else:
            stats = list(self.stats_lookup.values())
        if by_class:
            return {s.gold_label: getattr(s, metric) for s in stats}
        return average_lang_stats(metric, stats, average)

    # these overloads are necessary for the type checker because of polymorphism
    @overload
    def precision(
        self,
        average: str = ...,
        by_class: Literal[False] = ...,
        supported_languages_only: bool = ...,
    ) -> float: ...

    @overload
    def precision(
        self,
        average: str = ...,
        *,
        by_class: Literal[True],
        supported_languages_only: bool = ...,
    ) -> dict[str, float]: ...

    def precision(
        self,
        average: str = "macro",
        by_class: bool = False,
        supported_languages_only: bool = False,
    ) -> dict[str, float] | float:
        """Return precision, either by class or averaged.

        Parameters
        ----------
        average : str, optional
            Type of average ("macro", "micro", or "weighted"). Defaults to "macro".
        by_class : bool, optional
            Give metrics by class rather than overall. Defaults to False.
        supported_languages_only : bool, optional
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction. Defaults to False.

        Returns
        -------
        dict[str, float] | float
            Precision by class as a dict or overall precision.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        NotImplementedError
            If `by_class` is False and `average` is not "macro", "micro" or
            "weighted".
        """
        return self._compute_metric(
            "precision", average, by_class, supported_languages_only
        )

    @overload
    def recall(
        self,
        average: str = ...,
        by_class: Literal[False] = ...,
        supported_languages_only: bool = ...,
    ) -> float: ...

    @overload
    def recall(
        self,
        average: str = ...,
        *,
        by_class: Literal[True],
        supported_languages_only: bool = ...,
    ) -> dict[str, float]: ...

    def recall(
        self,
        average: str = "macro",
        by_class: bool = False,
        supported_languages_only: bool = False,
    ) -> dict[str, float] | float:
        """Return recall, either by class or averaged.

        Parameters
        ----------
        average : str, optional
            Type of average ("macro", "micro", or "weighted"). Defaults to "macro".
        by_class : bool, optional
            Give metrics by class rather than overall. Defaults to False.
        supported_languages_only : bool, optional
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction. Defaults to False.

        Returns
        -------
        dict[str, float] | float
            Recall by class as a dict, or overall recall.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        NotImplementedError
            If `by_class` is False and `average` is not "macro", "micro" or
            "weighted".
        """
        return self._compute_metric(
            "recall", average, by_class, supported_languages_only
        )

    @overload
    def f1(
        self,
        average: str = ...,
        by_class: Literal[False] = ...,
        supported_languages_only: bool = ...,
    ) -> float: ...

    @overload
    def f1(
        self,
        average: str = ...,
        *,
        by_class: Literal[True],
        supported_languages_only: bool = ...,
    ) -> dict[str, float]: ...

    def f1(
        self,
        average: str = "macro",
        by_class: bool = False,
        supported_languages_only: bool = False,
    ) -> dict[str, float] | float:
        """Return F1 score, either by class or averaged.

        Parameters
        ----------
        average : str, optional
            Type of average ("macro", "micro", or "weighted"). Defaults to "macro".
        by_class : bool, optional
            Give metrics by class rather than overall. Defaults to False.
        supported_languages_only : bool, optional
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction. Defaults to False.

        Returns
        -------
        dict[str, float] | float
            F1 by class as a dict, or overall F1.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        NotImplementedError
            If `by_class` is False and `average` is not "macro", "micro" or
            "weighted".
        """
        return self._compute_metric("f1", average, by_class, supported_languages_only)

    @overload
    def fpr(
        self,
        average: str = ...,
        by_class: Literal[False] = ...,
        supported_languages_only: bool = ...,
    ) -> float: ...

    @overload
    def fpr(
        self,
        average: str = ...,
        *,
        by_class: Literal[True],
        supported_languages_only: bool = ...,
    ) -> dict[str, float]: ...

    def fpr(
        self,
        average: str = "macro",
        by_class: bool = False,
        supported_languages_only: bool = False,
    ) -> dict[str, float] | float:
        """Return false positive rate, either by class or averaged.

        Parameters
        ----------
        average : str, optional
            Type of average ("macro", "micro", or "weighted"). Defaults to "macro".
        by_class : bool, optional
            Give metrics by class rather than overall. Defaults to False.
        supported_languages_only : bool, optional
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction. Defaults to False.

        Returns
        -------
        dict[str, float] | float
            FPR by class as a dict, or overall FPR.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        NotImplementedError
            If `by_class` is False and `average` is not "macro", "micro" or
            "weighted".
        """
        return self._compute_metric("fpr", average, by_class, supported_languages_only)

    @overload
    def classification_report(
        self,
        supported_languages_only: bool = ...,
        output_dict: Literal[False] = ...,
        digits: int = ...,
    ) -> str: ...

    @overload
    def classification_report(
        self,
        supported_languages_only: bool = ...,
        *,
        output_dict: Literal[True],
        digits: int = ...,
    ) -> dict[str, dict[str, float]]: ...

    def classification_report(
        self,
        supported_languages_only: bool = False,
        output_dict: bool = False,
        digits: int = 4,
    ) -> dict[str, dict[str, float]] | str:
        """Build a report giving the main metrics per class.

        Parameters
        ----------
        supported_languages_only : bool, optional
            Use only items whose gold label matches a supported language.
            Requires `supported_languages` at construction. Defaults to False.
        output_dict : bool, optional
            Return the report as a dict instead of a formatted string.
            Defaults to False.
        digits : int, optional
            Decimal places in the string report. Ignored if `output_dict` is
            True. Defaults to 4.

        Returns
        -------
        dict[str, dict[str, float]] | str
            sklearn-style classification report.

        Raises
        ------
        ValueError
            If `supported_languages_only` is True but no `supported_languages`
            were given when constructing `LangIdEval`.
        """
        lookup = (
            self.supported_stats_lookup
            if supported_languages_only
            else self.stats_lookup
        )
        if lookup is None:
            raise ValueError(
                "No supported_languages given when constructing LangIdEval()"
            )
        report_dict = build_report_dict(list(lookup.values()))
        if output_dict:
            return report_dict
        else:
            return format_report(report_dict, digits=digits)
