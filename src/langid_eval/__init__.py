import argparse

from .language_stats import LangStats
from .evaluator import LangIdEval, tag_match, closest_match_with_caching

__all__ = ["LangStats", "LangIdEval", "tag_match", "closest_match_with_caching"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("golds_file", help="file containing gold labels, one per line")
    parser.add_argument(
        "preds_file", help="file containing predicted labels, one per line"
    )
    parser.add_argument(
        "--supported_langs_file",
        "-s",
        help="file listing the labels that the evaluated model supports, one per line",
    )
    return parser.parse_args()


def main() -> None:
    # an example of usage: read in gold and predicted labels and write classification report
    args = parse_args()
    with open(args.golds_file) as f, open(args.preds_file) as g:
        golds = [x.strip() for x in f.readlines()]
        preds = [x.strip() for x in g.readlines()]
    assert len(golds) == len(preds)
    if args.supported_langs_file:
        with open(args.supported_langs_file) as f:
            supported_list = [x.strip() for x in f.readlines()]
        lideval = LangIdEval(golds, preds, supported_languages=supported_list)
        print(lideval.classification_report(supported_languages_only=True))
    else:
        lideval = LangIdEval(golds, preds)
        print(lideval.classification_report())


if __name__ == "__main__":
    main()
