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
        if not w.isalpha() or len(w) < 4:
            continue
        b = w
        for pos in ('n', 'v', 'a', 'r'):
            b = lem.lemmatize(b, pos)
        ss = wn.synsets(b)
        if len(b) < 4 or b in seen or not ss:
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


wn.synsets('test')  # preload
_silent = False


def ask_word(w, lo, hi):
    """Ask about one word. Returns True if known."""
    lp, hp = lo * 100 // N, hi * 100 // N
    print(f"  {lp}-{hp}%  {w} ", end="", flush=True)
    if not _silent:
        from pronounce import speak
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


def _search(ask_fn):
    """Fuzzy bisection search for vocabulary boundary.

    Bounds move halfway to mid instead of jumping fully,
    ~20 questions to converge. Stops when displayed lo%
    and hi% match. Randomized ±1/6 spread reduces bias
    from non-monotonic word difficulty.

    ask_fn(word, lo, hi) -> bool (known or not)
    """
    lo, hi = 0, N
    pts = []
    # stop when displayed percentages converge or gap < 1%
    while lo * 100 // N < hi * 100 // N and hi - lo > N // 100:
        mid = (lo + hi) // 2
        # jitter ±1/6 of range to avoid always hitting same word
        spread = (hi - lo) // 6
        mid = random.randint(mid - spread, mid + spread)
        rank, w = WORDS[mid]
        known = ask_fn(w, lo, hi)
        pts.append((mid, known))
        # move bounds halfway to mid (not fully) — tolerates
        # ~20% wrong answers without ruining the estimate
        if known:
            lo = (lo + mid + 1) // 2
        else:
            hi = (hi + mid) // 2
    # edge cases: all yes → knows everything, all no → knows nothing
    if not any(not k for _, k in pts):
        result = N
    elif not any(k for _, k in pts):
        result = 0
    else:
        result = (lo + hi) // 2
    return result, len(pts)


def estimate():
    print(f"Vocabulary size estimator ({N // 1000}K words)")
    print(f"{DIM}y/right = know, n/left = don't, q = quit{RST}\n")

    def ask(w, lo, hi):
        if hasattr(estimate, '_first'):
            time.sleep(1)
        estimate._first = True
        return ask_word(w, lo, hi)

    try:
        result, _ = _search(ask)
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    pct = result * 100 // N
    for threshold, lvl in [(10000, 'C2'), (6500, 'C1'), (4000, 'B2'),
                           (2500, 'B1'), (1000, 'A2'), (0, 'A1')]:
        if result >= threshold:
            break
    rk = (result + 500) // 1000
    print(f"\nEstimated vocabulary: ~{rk}K / {N // 1000}K, {pct}%, level {lvl}")
    return result


def _test():
    ok = True

    r, n = _search(lambda w, lo, hi: True)
    if r != N:
        print(f"FAIL all-yes: expected {N}, got {r}")
        ok = False
    if n > 30:
        print(f"FAIL all-yes: {n} iterations > 30")
        ok = False

    r, n = _search(lambda w, lo, hi: False)
    if r != 0:
        print(f"FAIL all-no: expected 0, got {r}")
        ok = False
    if n > 30:
        print(f"FAIL all-no: {n} iterations > 30")
        ok = False

    boundary = N // 2
    r, n = _search(lambda w, lo, hi: WORDS[(lo + hi) // 2][0] < WORDS[boundary][0])
    pct = r * 100 // N
    if not 40 <= pct <= 60:
        print(f"FAIL 50% boundary: got {pct}%")
        ok = False
    if n > 30:
        print(f"FAIL 50% boundary: {n} iterations > 30")
        ok = False

    if ok:
        print(f"OK  all-yes=100% all-no=0% mid={pct}% iterations={n}")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Vocabulary size estimator")
    p.add_argument("--lang", help="translate to language (e.g. ru, de, fr)")
    p.add_argument("-s", "--silent", action="store_true", help="no audio")
    p.add_argument("--test", action="store_true", help="run self-test")
    a = p.parse_args()
    _lang = a.lang
    _silent = a.silent
    if a.test:
        sys.exit(0 if _test() else 1)
    estimate()
