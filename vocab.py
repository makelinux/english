#!/usr/bin/env python3
"""Estimate English vocabulary size using binary search."""

import json
import os
import random
import sys
import termios
import tty

BATCH = 5
ROUNDS = 14
_CACHE = os.path.expanduser("~/.english-pronounce/vocab_cache.json")


def _load_words():
    """Load word list, using cache if available."""
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            return json.load(f)

    from wordfreq import top_n_list
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer

    print("Building word list (first run)...", end="", flush=True)
    lem = WordNetLemmatizer()
    seen = set()
    words = []
    for w in top_n_list('en', 100000):
        if not w.isalpha() or len(w) < 3 or not wn.synsets(w):
            continue
        b = w
        for pos in ('n', 'v', 'a', 'r'):
            b = lem.lemmatize(b, pos)
        if len(b) < 3 or b in seen:
            continue
        seen.add(b)
        seen.add(w)
        words.append(b)

    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, 'w') as f:
        json.dump(words, f)
    print(f" {len(words):,} words cached.")
    return words


WORDS = _load_words()
N = len(WORDS)

GRN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RST = "\033[0m"


def read_key():
    """Read single keypress, return it."""
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def word_info(word):
    """Get definition and synonyms from WordNet."""
    from nltk.corpus import wordnet as wn
    s = wn.synsets(word)[0]
    defn = s.definition()
    related = [l.replace('_', ' ') for l in s.lemma_names() if l != word][:3]
    return defn, related


def ask_word(w):
    """Ask about one word. Returns True if known."""
    print(f"  {w} ", end="", flush=True)
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
        if k in ('n', 'h'):
            known = False
            break
        if k in ('y', 'l', ' ', '\r'):
            known = True
            break
        if k == 'q':
            raise KeyboardInterrupt

    if known:
        print(f"{GRN}yes{RST}")
    else:
        defn, related = word_info(w)
        hint = f" - {defn}"
        if related:
            hint += f" ({', '.join(related)})"
        print(f"{RED}no{RST}  {DIM}{hint}{RST}")
    return known


def ask_batch(words):
    """Ask user about a batch of words. Return fraction known."""
    known = sum(ask_word(w) for w in words)
    return known / len(words)


def estimate():
    """Binary search for vocabulary boundary."""
    lo, hi = 0, N
    print(f"Vocabulary size estimator ({N:,} words)")
    print(f"{DIM}right/left arrow or y/n{RST}\n")

    for r in range(ROUNDS):
        mid = (lo + hi) // 2
        start = max(0, mid - N // 20)
        end = min(N, mid + N // 20)
        sample = random.sample(WORDS[start:end], min(BATCH, end - start))

        print(f"Round {r + 1}/{ROUNDS}  {DIM}({lo:,}-{hi:,}){RST}")
        frac = ask_batch(sample)

        if frac >= 0.5:
            lo = mid
        else:
            hi = mid
        print()

        if hi - lo < 500:
            break

    result = (lo + hi) // 2
    print(f"Estimated vocabulary: ~{result:,} words")
    return result


if __name__ == "__main__":
    try:
        estimate()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
