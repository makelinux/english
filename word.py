#!/usr/bin/env python3
"""Look up English word info: frequency, definition, synonyms, CEFR, translation."""

import argparse
import json
import urllib.parse
import urllib.request

from cefrpy import CEFRAnalyzer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from wordfreq import zipf_frequency, top_n_list

_cefr = CEFRAnalyzer()
_lem = WordNetLemmatizer()
_top = {w: i + 1 for i, w in enumerate(top_n_list('en', 1000))}


def _rank(w, freq):
    r = _top.get(w) or _top.get(w.lower())
    if r is not None:
        return r
    if freq <= 0:
        return None
    return max(1, int(10 ** (5.6618 - 0.1667 * freq - 0.0732 * freq**2)))


def translate(w, lang):
    if not lang:
        return ""
    u = (f"https://translate.google.com/translate_a/single"
         f"?client=gtx&sl=en&tl={lang}&dt=t"
         f"&q={urllib.parse.quote(w)}")
    try:
        r = urllib.request.urlopen(u, timeout=3)
        t = json.loads(r.read())[0][0][0]
        return t if t.lower() != w.lower() else ""
    except Exception:
        return ""


def word_header(w, lang=None):
    freq = zipf_frequency(w, 'en')
    rank = _rank(w, freq)
    h = f"{w}  freq={freq:.1f}"
    if rank is not None:
        if rank >= 1000:
            h += f"  rank={rank // 1000}K"
        else:
            h += f"  rank={rank}"
    lvl = _cefr.get_average_word_level_CEFR(w)
    if lvl:
        h += f"  {lvl}"
    t = translate(w, lang)
    if t:
        h += f"  {t}"
    return h


p = argparse.ArgumentParser(description="English word lookup")
p.add_argument("words", nargs="+", help="words to look up")
p.add_argument("--lang", help="translate to language (e.g. ru, de, fr)")
a = p.parse_args()

def find_root(w):
    """Find lemma/root trying all POS."""
    for pos in ('v', 'n', 'a', 'r'):
        r = _lem.lemmatize(w, pos)
        if r != w:
            return r
    return None

for w in a.words:
    print(word_header(w, a.lang))
    ss = wn.synsets(w)
    for s in ss[:5]:
        cat = s.lexname().split('.', 1)[-1]
        defn = s.definition()
        d = f"{cat}: {defn}" if cat not in ("all", "ppl") else defn
        print(f"  {d}")
        for l in s.lemma_names():
            syn = l.replace('_', ' ')
            if syn.lower() != w:
                print(f"    {word_header(syn, a.lang)}")
    if not ss:
        try:
            u = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(w)}"
            req = urllib.request.Request(u, headers={"User-Agent": "word.py"})
            r = json.loads(urllib.request.urlopen(req, timeout=3).read())
            for entry in r:
                for m in entry.get("meanings", []):
                    pos = m["partOfSpeech"]
                    for d in m.get("definitions", []):
                        print(f"  {pos}: {d['definition']}")
                    syns = m.get("synonyms", [])
                    if syns:
                        print(f"    synonyms: {', '.join(syns)}")
            print(f"  (dictionaryapi.dev / Wiktionary)")
        except Exception as e:
            print(f"  no definition found ({e})")
    root = find_root(w)
    if root:
        print(f"  root: {word_header(root, a.lang)}")
