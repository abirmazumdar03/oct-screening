# octscreen/app/i18n.py
"""Tiny translation layer: one YAML string table per language (configs/i18n/<lang>.yaml), nested keys → 'a.b.c'."""
import string
from pathlib import Path

import yaml


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)
    return out


def placeholders(s):
    return {name for _, name, _, _ in string.Formatter().parse(s) if name}


class Translator:
    def __init__(self, i18n_dir, languages, fallback="en"):
        self.languages = list(languages)
        self.fallback = fallback
        self.tables = {}
        for lang in self.languages:
            path = Path(str(i18n_dir)) / f"{lang}.yaml"
            self.tables[lang] = _flatten(yaml.safe_load(path.read_text(encoding="utf-8")))   # utf-8: Hindi on Windows
        if fallback not in self.tables:
            raise ValueError(f"fallback language {fallback!r} must be in {self.languages}")

    def t(self, lang, key, default=None, **kw):
        table = self.tables.get(lang, self.tables[self.fallback])
        s = table.get(key) or self.tables[self.fallback].get(key)
        if s is None:
            return default if default is not None else key
        try:
            return s.format(**kw)
        except (KeyError, IndexError, ValueError):
            return s

    def both(self, key, sep=" · "):
        """'Analyse · जाँच करें': for static labels that must be readable in every language."""
        return sep.join(dict.fromkeys(self.t(lang, key) for lang in self.languages))
