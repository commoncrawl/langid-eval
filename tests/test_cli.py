"""Tests for the langid-eval command. The report itself is tested in
test_report.py; these check the files are read and the options passed on."""

import sys

import pytest

from langid_eval import main

FILES = {
    # The stray space must be stripped, or Afrikaans is split in two.
    "golds": "afr_Latn_afri1274\nafr_Latn_afri1274 \narg_Latn_arag1245\n",
    "preds": "af\nnl\nes\n",
    "supported": "af\nnl\n",
}


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """Run the command on FILES, naming each by its key, and return stdout."""
    for name, text in FILES.items():
        (tmp_path / name).write_text(text)

    def run(*args):
        paths = [str(tmp_path / a) if a in FILES else a for a in args]
        monkeypatch.setattr(sys, "argv", ["langid-eval", *paths])
        main()
        return capsys.readouterr().out

    return run


def test_prints_a_row_for_every_gold_language(run):
    report = run("golds", "preds")
    assert report.count("afr_Latn_afri1274") == 1
    assert "arg_Latn_arag1245" in report


@pytest.mark.parametrize("flag", ["--supported_langs_file", "-s"])
def test_supported_langs_file_drops_unsupported_languages(run, flag):
    report = run("golds", "preds", flag, "supported")
    assert "afr_Latn_afri1274" in report
    assert "arg_Latn_arag1245" not in report
