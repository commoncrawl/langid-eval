# langid-eval

Tuneable fuzzy matching for language identification evaluation.

## Why `langid-eval`?

Evaluating language identification (LID) entails deciding whether a predicted language label matches the gold. However, this task is not as straightforward as it first seems: models and test sets may use different labelling schemes and cover different sets of languages, and which language varieties are matches depends on the application.

`langid-eval` is an open-source Python package for simplifying LID evaluation. It builds on the [`langcodes`](https://pypi.org/project/langcodes) library and the [Unicode CLDR Project](https://cldr.unicode.org) to reconcile language labels automatically. It allows the user to define the granularity of match that they need, from as narrow as the exact variety and script to as broad as the same macrolanguage. It also calculates key summary statistics in a reproducible way, either for all languages in the test set or only for those covered by the tested model.

## Installation

`langid-eval` can be installed from this repository, directly or using tools like `pip` or `uv`. It will be added to PyPI after publication!

Requires Python ≥3.14.

## Usage

### Python

```python
from langid_eval import LangIdEval

# similar labels match without preprocessing
golds = ["en", "en", "es", "ar", "zh", "fr"]
preds = ["en-GB", "fr", "es-MX", "arb", "cmn", "fr"]

lideval = LangIdEval(golds, preds)
print(lideval.classification_report())
lideval.precision()                    # macro average by default
lideval.f1(by_class=True)       # {'en': 0.667, 'es': 1.0, ...}

lideval = LangIdEval(golds, preds, supported_languages=["en", "fr"])
lideval.fpr(supported_languages_only=True)
print(lideval.classification_report(digits=2))
```

### Command Line

Given two equal-length files of gold and predicted labels (one per line), print an sklearn-style classification report. 

```bash
langid-eval golds.txt preds.txt [-s supported_langs.txt]
```

## Parameters

Three constructor arguments to `LangIdEval` affect how matches are defined:

- `ignore_script` (default: True). Ignore script subtages when matching.
- `max_tag_distance` (default: 5). The maximum tag distance (as defined by `langcodes`) at which two language tags will count as a match.
- `supported_languages` (default: ()). The subset of languages to evaluate on, intended to be the languages supported by the model under testing. Setting `supported_languages_only=True` will filter the test set to the supported languages only prior to calculating statistics, so that models are only tested on languages they claim to cover.

### Choosing `max_tag_distance`

## Citation

After publication! Please see the paper for further details on the model and analysis of its working.

## License

`langid-eval` is released under the MIT license.
