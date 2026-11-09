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
lideval.precision()        # macro average over all classes by default
lideval.recall(average="weighted")
lideval.f1(by_class=True)  # {'en': 0.667, 'es': 1.0, ...}

lideval = LangIdEval(golds, preds, supported_languages=["en", "fr"])
lideval.fpr(supported_languages_only=True)
print(lideval.classification_report(digits=2))
```

### Command line

Given two equal-length files of gold and predicted labels (one per line), print an sklearn-style classification report.

```bash
langid-eval golds.txt preds.txt [-s supported_langs.txt]
```

## How matching works

Three constructor arguments to `LangIdEval` affect how matches are defined:

- `ignore_script` (default: True). Ignore script subtags when matching.
- `max_tag_distance` (default: 5). The maximum tag distance (as defined by `langcodes`) at which two language tags will count as a match.
- `supported_languages` (default: ()). The subset of languages to evaluate on, intended to be the languages supported by the model under testing. Setting the method argument `supported_languages_only=True` will filter the test set only to the gold labels which are within `max_tag_distance` of one of the languages in this sequence prior to calculating statistics. The intended use is to evaluate models only on the languages they claim to cover.

### Choosing `max_tag_distance`

`langid-eval` uses the `tag_distance` function from the `langcodes` package to adjust the strictness of the match. It returns a number from 0 to 134 indicating the distance between the language the user desires and a supported language (predicted and gold labels for `langid-eval`). This distance is intended as a measure of likely language acceptability; that is, given a user understands one language, how likely is it that the other will be acceptable to them? The distance data comes from CLDR v38.1 and involves a lot of judgment calls made by the Unicode Consortium. For this reason, it is not a true metric, but rather should be taken as indicative. Note that distances are not necessarily symmetric: for example, `tag_distance("arz", "ar") = 10`, whereas `tag_distance("ar", "arz") = 80`.

We copy the table describing tag distances from the [documentation for `langcodes`](https://github.com/georgkrause/langcodes#comparing-and-matching-languages) below. See the [docstring of `tag_distance`](https://github.com/georgkrause/langcodes/blob/4701cd34cc607c4ffcd2ed675f03b1d6529398da/langcodes/__init__.py#L1655) for more explanation and examples.

| Value | Meaning                                                                                           | Example                                          |
| ----: | :------------------------------------------------------------------------------------------------ | :----------------------------------------------- |
|     0 | These codes represent the same language, possibly after filling in values and normalizing.        | Norwegian Bokmål → Norwegian                     |
|   1-3 | These codes indicate a minor regional difference.                                                 | Australian English → British English             |
|   4-9 | These codes indicate a significant but unproblematic regional difference.                         | American English → British English               |
| 10-24 | A gray area that depends on your use case. There may be problems with understanding or usability. | Afrikaans → Dutch, Wu Chinese → Mandarin Chinese |
| 25-50 | These languages aren't similar, but there are demographic reasons to expect some intelligibility. | Tamil → English, Marathi → Hindi                 |
| 51-79 | There are large barriers to understanding.                                                        | Japanese → Japanese in Hepburn romanization      |
| 80-99 | These are different languages written in the same script.                                         | English → French, Arabic → Urdu                  |
|  100+ | These languages have nothing particularly in common.                                              | English → Japanese, English → Tamil              |

Since `langid-eval` is intended to match similar language varieties, we recommend that `max_tag_distance` is set to 20 at the very most. As a rule of thumb, a value of 0 matches equivalent codes after normalisation, 5 (the default) allows for regional differences, and 10-15 matches individual languages to their macrolanguage. In case of unexpected behaviour, please raise an issue in this repository.

## API reference

Full parameter details are in the docstrings, e.g. `help(LangIdEval.f1)`.

| Name | Description |
| :--- | :---------- |
| `LangIdEval(golds, preds, ignore_script=True, max_tag_distance=5, supported_languages=())` | Evaluate predicted labels against gold labels. See [How matching works](#how-matching-works). |
| `.precision()`, `.recall()`, `.f1()`, `.fpr()` | Overall metric (`average="macro"`, `"micro"` or `"weighted"`) or per language (`by_class=True`). Use `supported_languages_only=True` to evaluate on supported languages only. |
| `.classification_report(supported_languages_only=False, output_dict=False, digits=4)` | sklearn-style per-language report, as a string or a dict (`output_dict=True`). |

## Citation

After publication! Please see the paper for further details and analysis of the package.

## License

`langid-eval` is released under the MIT license.
