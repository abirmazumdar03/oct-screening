# tests/test_app_i18n.py
from pathlib import Path

import pytest

from octscreen.app.i18n import Translator, placeholders


@pytest.fixture(scope="module")
def tr():
    return Translator(Path("configs/i18n"), ["en", "hi"], "en")


def test_every_language_has_the_same_keys(tr):
    assert set(tr.tables["hi"]) == set(tr.tables["en"])


def test_placeholders_match_across_languages(tr):
    for key, en in tr.tables["en"].items():
        assert placeholders(en) == placeholders(tr.tables["hi"][key]), key


def test_missing_key_falls_back(tr):
    assert tr.t("hi", "does.not.exist", default="X") == "X"
    assert tr.t("xx", "title") == tr.t("en", "title")


def test_both_shows_every_language(tr):
    assert tr.both("analyze_button") == "Analyse · जाँच करें"
