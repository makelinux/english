#!/usr/bin/env python3
"""Estimate English vocabulary size using binary search."""

import json
import os
import random
import sys
import termios
import threading
import time
import tty
import urllib.parse
import urllib.request

import yaml
from cefrpy import CEFRAnalyzer
from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

_cefr = CEFRAnalyzer()
_lang = None


def translate(word):
    if not _lang:
        return ""
    u = (f"https://translate.google.com/translate_a/single"
         f"?client=gtx&sl=en&tl={_lang}&dt=t"
         f"&q={urllib.parse.quote(word)}")
    try:
        r = urllib.request.urlopen(u, timeout=3)
        return json.loads(r.read())[0][0][0]
    except Exception:
        return ""

GRN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RST = "\033[0m"


_CACHE = os.path.expanduser("~/.config/english-pronounce/vocab_cache.yaml")


def _circular(w, defn):
    """Check if definition reuses the word or its stem."""
    stem = w[:max(4, len(w) - 2)]
    if len(stem) < 4:
        return False
    return any(t.startswith(stem) for t in defn.lower().split())


def _load_words():
    """Load lemmatized words sorted by frequency, with definitions."""
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            return [tuple(w) for w in yaml.safe_load(f)]
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    seen = set()
    words = []
    wl = top_n_list('en', 100000)
    for i, w in enumerate(wl):
        if i % 5000 == 0:
            print(f"\rLoading {i*100//len(wl)}%", end="", flush=True)
        if not w.isalpha() or len(w) < 3:
            continue
        b = w
        for pos in ('n', 'v', 'a', 'r'):
            b = lem.lemmatize(b, pos)
        ss = wn.synsets(b)
        if len(b) < 3 or b in seen or not ss:
            continue
        if all(l[0].isupper() for s in ss for l in s.lemma_names()):
            continue
        good = [x for x in ss
                if any(l[0].islower() for l in x.lemma_names())
                and not _circular(b, x.definition())]
        if not good:
            continue
        seen.add(b)
        seen.add(w)
        words.append((i, b))
    # remove derived forms when base exists
    bases = {w for _, w in words}
    _sfx = [('ably', 3), ('ibly', 3), ('ously', 3), ('ively', 3),
            ('fulness', 3), ('lessly', 3), ('ically', 5),
            ('ness', 4), ('ly', 2), ('ment', 4), ('ful', 3),
            ('less', 4), ('ity', 3), ('ism', 3), ('ist', 3),
            ('able', 3), ('ible', 3), ('ive', 3), ('ous', 3),
            ('tion', 3), ('sion', 3), ('er', 3), ('al', 3),
            ('ic', 5), ('ical', 5)]
    _pfx = ['un', 'dis', 'mis', 'non', 'over', 'out', 'sub',
            'super', 'anti', 'semi', 'in', 'im', 'ir', 'il',
            'de', 're', 'pre']
    def has_base(s):
        return s in bases or s + 'e' in bases or s + 'y' in bases \
            or s[:-1] + 'y' in bases or s + 'ate' in bases
    def is_derived(w):
        for sfx, mn in _sfx:
            if w.endswith(sfx) and len(w) - len(sfx) >= mn:
                if has_base(w[:-len(sfx)]):
                    return True
        for pfx in _pfx:
            if w.startswith(pfx) and len(w) - len(pfx) >= 3:
                if w[len(pfx):] in bases:
                    return True
        # combined: prefix + base + suffix
        for pfx in _pfx:
            if not w.startswith(pfx):
                continue
            rest = w[len(pfx):]
            for sfx, mn in _sfx:
                if rest.endswith(sfx) and len(rest) - len(sfx) >= mn:
                    if has_base(rest[:-len(sfx)]):
                        return True
        return False
    words = [(i, w) for i, w in words if not is_derived(w)]
    print("\r\033[K", end="", flush=True)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, 'w') as f:
        yaml.dump([list(w) for w in words], f)
    return words


WORDS = _load_words()
N = len(WORDS)


def read_key():
    """Read single keypress, return it."""
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def word_info(w):
    """Get definition, CEFR level, and frequency for a word."""
    parts = []
    s = wn.synsets(w)
    if s:
        # prefer non-circular synsets with lowercase lemmas
        good = [x for x in s
                if any(l[0].islower() for l in x.lemma_names())
                and not _circular(w, x.definition())]
        syn = good[0] if good else s[0]
        cat = syn.lexname().split('.', 1)[-1]
        defn = syn.definition()
        if cat in ("all", "ppl"):
            cat = ""
        elif cat == "pert":
            cat = ""
            for l in syn.lemmas():
                for p in l.pertainyms():
                    cat = p.synset().lexname().split('.', 1)[-1]
                    break
                if cat:
                    break
        related = [l.replace('_', ' ') for l in syn.lemma_names()
                   if l.lower() != w][:3]
        desc = f"{cat}: {defn}" if cat else defn
        if related:
            desc += f" ({', '.join(related)})"
        parts.append(desc)
    lvl = _cefr.get_average_word_level_float(w)
    freq = zipf_frequency(w, 'en')
    # hide CEFR when it looks wrong (easy level for rare words)
    if lvl and not (lvl <= 2 and freq < 4):
        parts.append(str(_cefr.get_average_word_level_CEFR(w)))
    return "  ".join(parts) or "?"


from pronounce import speak
wn.synsets('test')  # preload


def ask_word(w, lo, hi):
    """Ask about one word. Returns True if known."""
    lp, hp = lo * 100 // N, hi * 100 // N
    print(f"  {lp}-{hp}%  {w} ", end="", flush=True)
    threading.Thread(target=speak, args=(w,), daemon=True).start()
    while True:
        k = read_key()
        if k == '\x1b':
            k2 = read_key()
            if k2 == '[':
                k3 = read_key()
                if k3 == 'D':
                    known = False
                    break
                if k3 == 'C':
                    known = True
                    break
                continue
            print()
            raise KeyboardInterrupt
        if k in ('n', 'h'):
            known = False
            break
        if k in ('y', 'l', ' ', '\r'):
            known = True
            break
        if k == 'q':
            raise KeyboardInterrupt

    clr = GRN + "yes" if known else RED + "no"
    print(f"{clr}{RST}  {DIM}{word_info(w)}{RST}", end="", flush=True)
    t = translate(w)
    if t and t.lower() != w.lower():
        print(f"  {DIM}{t}{RST}", end="")
    print()
    return known


def estimate():
    """Fuzzy bisection search for vocabulary boundary.

    Bounds move halfway to mid instead of jumping fully,
    ~20 questions to converge. Stops when displayed lo%
    and hi% match. Randomized ±1/6 spread reduces bias
    from non-monotonic word difficulty.
    """
    lo, hi = 0, N
    glob = WORDS[-1][0]
    print(f"Vocabulary size estimator ({N:,} normalized, {glob:,} global)")
    print(f"{DIM}y/right = know, n/left = don't, q = quit{RST}\n")

    pts = []
    try:
        # stop when lo% == hi% (displayed percentages converge)
        while lo * 100 // N < hi * 100 // N:
            mid = (lo + hi) // 2
            # jitter ±1/6 of range to avoid always hitting same word
            spread = (hi - lo) // 6
            mid = random.randint(mid - spread, mid + spread)
            rank, w = WORDS[mid]
            if pts:
                time.sleep(1)
            known = ask_word(w, lo, hi)
            pts.append((mid, known))
            # move bounds halfway to mid (not fully) — tolerates
            # ~20% wrong answers without ruining the estimate
            if known:
                lo = (lo + mid + 1) // 2
            else:
                hi = (hi + mid) // 2
    except KeyboardInterrupt:
        if not pts:
            print("\nAborted.")
            return
        print()

    # robust estimate: skip outliers by taking 2nd extremes
    yes_pts = sorted((p for p, k in pts if k), reverse=True)
    no_pts = sorted(p for p, k in pts if not k)
    y = yes_pts[min(1, len(yes_pts) - 1)] if yes_pts else 0
    n = no_pts[min(1, len(no_pts) - 1)] if no_pts else N
    if not yes_pts:
        n = no_pts[0]
    if not no_pts:
        y = yes_pts[0]
    result = (y + n) // 2
    pct = result * 100 // N
    for threshold, lvl in [(10000, 'C2'), (6500, 'C1'), (4000, 'B2'),
                           (2500, 'B1'), (1000, 'A2'), (0, 'A1')]:
        if result >= threshold:
            break
    print(f"\nEstimated vocabulary: ~{result:,} / {N:,}, {pct}%, level {lvl}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Vocabulary size estimator")
    p.add_argument("--lang", help="translate to language (e.g. ru, de, fr)")
    a = p.parse_args()
    _lang = a.lang
    estimate()
