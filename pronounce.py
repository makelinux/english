#!/usr/bin/env python3
"""English pronunciation trainer - practice words grouped by phoneme."""

import argparse
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import speech_recognition as sr
import yaml
from box import Box

import audio
from audio import (
    SAMPLE_RATE, CONF_DIR, CAL_WORDS,
    VOICES_MALE, VOICES_FEMALE, VOICES_ALL, _BARS,
    cal, DIM, RST,
    load_calibration, save_calibration, quick_calibrate, CAL_FILE,
    status, record_audio, play_raw, speak,
    ensure_ref, get_ref_path, audio_similarity,
    normalize_volume, extract_mfcc, dtw_distance,
    raw_to_sr_audio, _raw_to_wav, _gemini_tts_wav, _ref_raw,
    test_rec,
)

DATA = Path(__file__).parent / "english.yaml"
HIST = CONF_DIR / "history.yaml"
CFG_FILE = CONF_DIR / "config.yaml"

cfg = Box(default_box=True)
data = Box()


def load_cfg():
    if CFG_FILE.exists():
        with open(CFG_FILE) as f:
            c = yaml.safe_load(f)
        if c:
            cfg.update(c)


def load_data():
    with open(DATA) as f:
        raw = yaml.safe_load(f)
    global data
    data = Box(raw)
    data.prompts.sysprompt = data.prompts.sysprompt.format(no_ipa=_no_ipa())
    data.stt_equiv = [set(g) for g in data.stt_equiv or []]


def load_history():
    if HIST.exists():
        with open(HIST) as f:
            h = yaml.safe_load(f)
        if h:
            return h
    return {"words": {}, "groups": {}}


def save_history(h):
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST, "w") as f:
        yaml.dump(h, f)


_term_saved = None


def set_cbreak():
    global _term_saved
    _term_saved = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def restore_term():
    global _term_saved
    if _term_saved:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _term_saved)
        _term_saved = None


def clear_line():
    print("\r\033[K", end="", flush=True)


def wait_key(timeout=2):
    """Returns key pressed or None."""
    if _term_saved:
        key = None
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
        if key:
            return key
        if timeout is None:
            return sys.stdin.read(1)
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        key = None
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
        if key:
            return key
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    return None


# IPA-based homophone lookup
ipa_to_words = {}  # "/raɪt/" -> {"right", "write"}


def build_ipa_map():
    for g in data.phonemes.values():
        for w in g.words:
            for ipa in re.findall(r'/[^/]+/', w.ipa):
                ipa_to_words.setdefault(ipa, set()).add(w.word.lower())


def stt_score(expected, heard):
    """Return 0-100 based on speech-to-text match."""
    if not heard:
        return 0
    e = expected.lower().strip()
    h = heard.lower().strip()
    if e == h:
        return 100
    a = data.stt_aliases
    if a and (a.get(e) == h or a.get(h) == e):
        return 100
    for group in data.stt_equiv:
        if e in group and h in group:
            return 100
    for words in ipa_to_words.values():
        if e in words and h in words:
            return 100
    if e in h.split():
        return 80
    r = SequenceMatcher(None, e, h).ratio()
    if r > 0.9:
        return int(r * 80)
    return 0


def record_word(word, rec, prefix="", duration=5):
    """Returns (heard, pct, sim, peak, dur, raw, key)."""
    bars = []

    def on_chunk(peak):
        if peak > audio._peak_max:
            audio._peak_max = peak
        mx = audio._peak_max if audio._peak_max > 1e-6 else 1.0
        bars.append(_BARS[min(8, int(peak / mx * 8))])
        print(f"\r\033[K{prefix}"
              f"{DIM}[s]kip [f]eedback [q]uit{RST} "
              f"Listening{''.join(bars)}🎤", end="", flush=True)

    try:
        ref = get_ref_path(word)
        key = None
        while True:
            raw, spoke, key = record_audio(
                duration=duration, on_chunk=on_chunk,
                check_keys=bool(_term_saved))
            if key in ('s', 'q', 'f'):
                print()
                return None, 0, 0, 0, 0, None, key
            peak_raw = int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))))
            if spoke and peak_raw > 1000:
                break
        samples = np.frombuffer(raw, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))
        dur = len(samples) / SAMPLE_RATE
        play_raw(raw)
        print(f"\r\033[K{prefix}Processing...", end="", flush=True)
        sim = audio_similarity(ref, raw) if ref.exists() else 0
        try:
            heard = rec.recognize_google(raw_to_sr_audio(raw))
        except sr.UnknownValueError:
            heard = None
        except sr.RequestError:
            heard = None
        stt = stt_score(word, heard)
        return heard, stt, sim, peak, dur, raw, key
    except Exception as e:
        print(f"\r\033[K{prefix}Recording error: {e}")
        return None, 0, 0, 0, 0, None, None


EMA_ALPHA = 0.2


def _ema(entry):
    """Return EMA if present, else fall back to all-time average."""
    if "ema" in entry:
        return entry["ema"]
    return entry["correct"] / entry["attempts"]


def group_accuracy(h, gid):
    """Returns -1 if no history."""
    g = h["groups"].get(gid)
    if not g or g["attempts"] == 0:
        return -1
    return _ema(g)


def pick_words(words, h, count=10):
    scored = []
    for w in words:
        s = h["words"].get(w.word)
        if not s or s["attempts"] == 0:
            scored.append((w, -1))
        else:
            scored.append((w, _ema(s)))
    scored.sort(key=lambda x: x[1])
    return [w for w, _ in scored[:count]]


def update_history(h, gid, word, pct):
    s = pct / 100
    if word not in h["words"]:
        h["words"][word] = {"attempts": 0, "correct": 0, "last": ""}
    w = h["words"][word]
    w["attempts"] += 1
    w["correct"] += s
    w["ema"] = w.get("ema", s) * (1 - EMA_ALPHA) + s * EMA_ALPHA
    w["last"] = str(date.today())
    if gid not in h["groups"]:
        h["groups"][gid] = {"attempts": 0, "correct": 0}
    g = h["groups"][gid]
    g["attempts"] += 1
    g["correct"] += s
    g["ema"] = g.get("ema", s) * (1 - EMA_ALPHA) + s * EMA_ALPHA


def _no_ipa():
    return "" if os.getenv("GOOGLE_API_KEY") else " No IPA symbols."


def _ask_ai(raw, prompt):
    """Send audio + prompt to Gemini or OpenAI, return response text."""
    wav = _raw_to_wav(raw)
    if cfg.openai:
        return _ask_openai(wav, prompt)
    return _ask_gemini(wav, prompt)


def _ask_gemini(wav, prompt):
    import google.generativeai as genai
    status(f"  {DIM}gemini-flash-latest ...{RST}")
    m = genai.GenerativeModel("gemini-flash-latest",
                              system_instruction=data.prompts.sysprompt)
    r = m.generate_content([
        prompt, {"mime_type": "audio/wav", "data": wav},
    ])
    status()
    return r.text.strip()


def _ask_openai(wav, prompt):
    import base64
    from openai import OpenAI
    oc = cfg.openai
    c = OpenAI(
        base_url=oc.base_url or "http://localhost:8321/v1/",
        api_key=oc.api_key or "none",
    )
    b64 = base64.b64encode(wav).decode()
    mdl = oc.model or os.getenv("INFERENCE_MODEL",
                                "gemini/gemini-2.5-flash")
    status(f"  {DIM}{mdl} ...{RST}")
    uri = f"data:audio/wav;base64,{b64}"
    af = oc.audio_format or "image_url"
    if (oc.api_type or "completions") == "responses":
        if af == "input_audio":
            part = {"type": "input_audio",
                    "input_audio": {"data": b64, "format": "wav"}}
        elif af == "input_file":
            part = {"type": "input_file", "filename": "audio.wav",
                    "file_data": uri}
        else:
            part = {"type": "input_image", "image_url": uri}
        r = c.responses.create(
            model=mdl,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, part,
            ]}],
        )
        status()
        return r.output_text.strip()
    if af == "input_audio":
        part = {"type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"}}
    else:
        part = {"type": "image_url", "image_url": {"url": uri}}
    r = c.chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": data.prompts.sysprompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, part,
            ]},
        ],
    )
    status()
    return r.choices[0].message.content.strip()


def get_feedback(raw, word, ipa):
    p = data.prompts.feedback.format(word=word, ipa=ipa)
    fb = _ask_ai(raw, p)
    return re.sub(r'"(\w+)\."', r'"\1".', fb)


def get_assessment(raw, text):
    p = data.prompts.assess.format(text=text, groups=", ".join(data.phonemes.keys()))
    return _ask_ai(raw, p)


def parse_assessment(text):
    if text.strip().upper() == "GOOD":
        return []
    groups = []
    for line in text.strip().splitlines():
        g = line.strip().lower()
        if g in data.phonemes:
            groups.append(g)
    return groups


RED = "\033[31m"
YEL = "\033[33m"
GRN = "\033[32m"

sim_threshold = 60


def sim_color(pct):
    if pct >= sim_threshold:
        return GRN
    if pct >= sim_threshold // 2:
        return YEL
    return RED


def show_stats(h):
    if not h["groups"]:
        print("No practice history yet.")
        return
    print("\nPerformance by phoneme group:\n")
    for gid, g in sorted(h["groups"].items()):
        acc = _ema(g)
        print(f"  {gid:20s} {acc:.0%}"
              f"  ({g['attempts']} attempts)")

    print("\nWeakest words:\n")
    weak = []
    for w, s in h["words"].items():
        if s["attempts"] >= 2:
            weak.append((w, _ema(s), s["attempts"]))
    weak.sort(key=lambda x: x[1])
    for w, acc, att in weak[:10]:
        print(f"  {w:20s} {acc:.0%}  ({att} attempts)")
    if not weak:
        print("  Need at least 2 attempts per word to rank.")


def list_groups():
    print("\nPhoneme groups:\n")
    for gid, g in data.phonemes.items():
        n = len(g.words)
        print(f"  {gid:20s} - {g.name} ({n} words)")


def select_group(h):
    keys = {}
    used = set()
    for gid, g in data.phonemes.items():
        k = g.get("key")
        if not k:
            for ch in gid:
                if ch.isalpha() and ch not in used and ch != '?':
                    k = ch
                    break
        if k:
            used.add(k)
            keys.setdefault(k, []).append(gid)

    print("\nPhoneme groups:\n")
    sorted_keys = sorted(keys.keys())
    for k in sorted_keys:
        gids = keys[k]
        names = []
        for gid in gids:
            g = data.phonemes[gid]
            acc = group_accuracy(h, gid)
            tag = f" {acc:.0%}" if acc >= 0 else " new"
            names.append(f"{g.name}{tag}")
        print(f"  {k} - {' / '.join(names)}")
    print(f"  ? - weakest group (auto-select)")
    print()

    while True:
        print(f"{DIM}Pick a group:{RST} ", end="", flush=True)
        c = wait_key(None)
        clear_line()
        if c == '?':
            accs = [(gid, group_accuracy(h, gid)) for gid in data.phonemes]
            accs.sort(key=lambda x: x[1])
            return accs[0][0]
        if c in keys:
            gids = keys[c]
            if len(gids) == 1:
                return gids[0]
            for i, gid in enumerate(gids):
                g = data.phonemes[gid]
                acc = group_accuracy(h, gid)
                tag = f" {acc:.0%}" if acc >= 0 else " new"
                print(f"  {i + 1} - {g.name}{tag}")
            print(f"{DIM}Pick:{RST} ", end="", flush=True)
            c2 = wait_key(None)
            clear_line()
            try:
                n = int(c2)
                if 1 <= n <= len(gids):
                    return gids[n - 1]
            except ValueError:
                pass


def _do_feedback(raw, word, ipa, h=None):
    try:
        fb = get_feedback(raw, word, ipa)
    except Exception as e:
        status()
        print(f"  {DIM}Feedback error: {e}{RST}")
        return
    print(f"  {DIM}{fb}{RST}")
    speak(re.sub(r'[\"\'()"/]', '', fb))
    if h and word in h["words"]:
        h["words"][word]["feedback"] = fb


def practice_word(w, rec, num="", cont=False, debug=False, prev=None, h=None):
    """Practice one word with retries. Returns (best_pct, last_raw).
    best_pct=-1 means quit. prev=(raw, word, ipa) for feedback fallback."""
    ensure_ref(w.word)

    prefix = f"{num}{w.word}  {w.ipa}  "
    best = 0
    last_raw = None
    lb_s = ""
    while True:
        clear_line()
        if debug:
            lb = [None]
            t = threading.Thread(
                target=lambda: lb.__setitem__(0, record_audio(3)[0]))
            print(f"{prefix} selfcheck, Wait! ", end="", flush=True)
            t.start()
            time.sleep(cal["delay"])
            speak(w.word)
            t.join()
            s = audio_similarity(get_ref_path(w.word), lb[0])
            lb_s = f"selfcheck={sim_color(s)}{s}%{RST}  "
            print(f"\r\033[K{prefix}{lb_s}Listening", end="", flush=True)
        else:
            print(f"{prefix}Listening", end="", flush=True)
            speak(w.word)

        heard, pct, sim, peak, dur, raw, key = \
            record_word(w.word, rec, prefix)
        if raw:
            last_raw = raw

        if key == 'q':
            clear_line()
            return -1, last_raw
        if key == 's':
            print(f"\r\033[K{prefix}{DIM}skipped{RST}")
            break
        if key == 'f':
            clear_line()
            if last_raw:
                fb_raw, fb_word, fb_ipa = last_raw, w.word, w.ipa
            elif prev:
                fb_raw, fb_word, fb_ipa = prev
            else:
                continue
            _do_feedback(fb_raw, fb_word, fb_ipa, h)
            continue

        if pct > best:
            best = pct

        heard_s = f"  heard: {heard}" if heard else ""
        dbg = f"  {dur:.1f}s peak={peak * 100 // 32768}%" if debug else ""
        sim_s = f"  {sim_color(sim)}{sim:2d}%{RST}" if sim else ""
        score = f"{heard_s}{sim_s}{dbg}"
        print(f"\r\033[K{prefix}{lb_s}{score}")

        if sim >= sim_threshold and pct > 0 \
                or sim >= sim_threshold // 2 and pct == 100 \
                or not sim and pct >= 80:
            break

        if cont:
            k = wait_key(2)
            if k == 'q':
                return -1, last_raw
            if k == 's':
                break
            if k == 'f' and last_raw:
                _do_feedback(last_raw, w.word, w.ipa, h)
            elif k == 'p':
                print(f"  {DIM}Paused. Any key...{RST}",
                      end="", flush=True)
                wait_key(None)
                clear_line()
        else:
            print(f"  {DIM}[Enter] retry [s]kip [f]eedback{RST}",
                  end="", flush=True)
            c = wait_key(None)
            clear_line()
            if c in ('\r', '\n', None):
                pass
            elif c == "s":
                break
            elif c == "f" and last_raw:
                _do_feedback(last_raw, w.word, w.ipa, h)

    return best, last_raw


def _assess_one(text, ipa):
    """Record and assess one pangram. Returns list of weak group IDs."""
    print(f"  {text}")
    if ipa:
        print(f"  {DIM}{ipa}{RST}")
    print()
    print(f"  {DIM}Listen first...{RST}", end="", flush=True)
    speak(text)
    print("\r\033[K", end="")
    ensure_ref(text)
    rec = sr.Recognizer()
    pw = len(text.split())
    while True:
        heard, pct, sim, peak, dur, raw, key = record_word(
            text, rec, "  ", duration=30)
        if key == 'q' or not raw:
            return []
        hw = len(heard.split()) if heard else 0
        if hw < pw * 0.5:
            print(f"  Incomplete ({hw}/{pw} words). Try again.")
            continue
        break
    clear_line()
    if heard:
        print(f"  Heard: {heard}")
    if sim:
        print(f"  Audio similarity: {sim_color(sim)}{sim}%{RST}")
    try:
        r = get_assessment(raw, text)
    except Exception as e:
        print(f"  Analysis error: {e}")
        return []
    return parse_assessment(r)


def assess(h, cont=False, debug=False):
    print(f"\nPronunciation assessment\n")
    weak = []
    for p in data.pangrams:
        text = p.get("text", p) if isinstance(p, dict) else p
        ipa = p.get("ipa", "") if isinstance(p, dict) else ""
        w = _assess_one(text, ipa)
        for g in w:
            if g not in weak:
                weak.append(g)
        if weak:
            break
    if not weak:
        print("  Your pronunciation sounds good!")
        return
    print(f"\n  Areas to work on:\n")
    for i, gid in enumerate(weak):
        g = data.phonemes[gid]
        acc = group_accuracy(h, gid)
        tag = f" ({acc:.0%})" if acc >= 0 else ""
        print(f"  {i + 1}. {g.name}{tag}")
    print(f"\n  {DIM}[Enter] practice  [q]uit{RST}", end="", flush=True)
    c = wait_key(None)
    print()
    if c == 'q':
        return
    gid = weak[0]
    print(f"\n  Starting practice: {data.phonemes[gid].name}\n")
    practice_phonemes(gid, h, cont, debug)


def practice_twisters(h):
    print(f"\nTongue twisters\n")
    rec = sr.Recognizer()
    for i, tw in enumerate(data.twisters):
        text = tw["text"]
        gid = tw.get("group", "")
        gn = data.phonemes[gid].name if gid in data.phonemes else ""
        lbl = f"  {i + 1}/{len(data.twisters)}"
        if gn:
            lbl += f"  ({gn})"
        print(lbl)
        print(f"  {text}")
        print()
        print(f"  {DIM}Listen...{RST}", end="", flush=True)
        speak(text)
        print("\r\033[K", end="")
        ensure_ref(text)
        pw = len(text.split())
        while True:
            heard, pct, sim, peak, dur, raw, key = record_word(
                text, rec, "  ", duration=15)
            if key == 'q':
                return
            if not raw:
                continue
            hw = len(heard.split()) if heard else 0
            if hw < pw * 0.5:
                print(f"  Too short ({hw}/{pw} words). Try again.")
                continue
            break
        clear_line()
        if heard:
            print(f"  Heard: {heard}")
        if sim:
            print(f"  Audio similarity: {sim_color(sim)}{sim}%{RST}")
        nxt = f"[Enter] next  " if i < len(data.twisters) - 1 else ""
        print(f"\n  {DIM}{nxt}[f]eedback  [q]uit{RST}", end="",
              flush=True)
        c = wait_key(None)
        print()
        if c == 'q':
            break
        if c == 'f':
            try:
                fb = _ask_ai(raw, data.prompts.twister.format(text=text))
            except Exception as e:
                print(f"  Feedback error: {e}")
                continue
            print(f"  {fb}")
            if fb.strip() != "Good":
                speak(re.sub(r'[\"\'()"/]', '', fb))
        print()


def practice_phonemes(gid, h, cont=False, debug=False):
    g = data.phonemes[gid]
    print(f"\n{g.name}")
    print(f"  {g.description}")
    print()

    if cont:
        set_cbreak()

    rec = sr.Recognizer()

    words = pick_words(g.words, h)
    results = []
    prev = None  # (raw, word, ipa) for feedback on previous word

    try:
        for i, w in enumerate(words):
            best, raw = practice_word(
                w, rec, f"{i + 1}/{len(words)}: ", cont, debug, prev, h)
            if raw:
                prev = (raw, w.word, w.ipa)
            if best == -1:
                break
            results.append((w.word, best))
            update_history(h, gid, w.word, best)

            if i < len(words) - 1:
                if cont:
                    k = wait_key(1)
                    if k == 'q':
                        break
                    if k == 'p':
                        print(f"  {DIM}Paused. Any key...{RST}",
                              end="", flush=True)
                        wait_key(None)
                        clear_line()
                else:
                    print(f"  {DIM}[Enter] next [q]uit{RST}",
                          end="", flush=True)
                    c = wait_key(None)
                    clear_line()
                    if c == "q":
                        break
    finally:
        if cont:
            restore_term()

    total = len(results)
    if not total:
        return
    avg = sum(p for _, p in results) / total
    good = sum(1 for _, p in results if p >= 80)
    ok = sum(1 for _, p in results if 40 <= p < 80)
    bad = sum(1 for _, p in results if p < 40)

    print(f"\nSession summary")
    print(f"  Words practiced: {total}")
    print(f"  Average score: {avg:.0f}%")
    print(f"  Good (>=80%): {good}")
    print(f"  Needs work (40-79%): {ok}")
    print(f"  Struggled (<40%): {bad}")

    if bad:
        print("\n  Words to focus on:")
        for w, p in sorted(results, key=lambda x: x[1]):
            if p < 40:
                print(f"    {w} - {p}%")

    save_history(h)


def calibrate():
    print("Calibration - measuring your mic/speaker channel.")
    print("Keep quiet, the app will play and record.\n")

    for w in CAL_WORDS:
        ensure_ref(w)

    # find best top_db by testing trim lengths
    print("  Finding trim threshold...", end=" ", flush=True)
    raw, _, _ = record_audio(duration=2)  # silence sample
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    samples = normalize_volume(samples)
    best_db = 20
    for db in [15, 18, 20, 22, 25]:
        trimmed, _ = librosa.effects.trim(samples, top_db=db)
        if len(trimmed) < len(samples) * 0.3:
            best_db = db
            break
    cal["top_db"] = best_db
    print(f"top_db={best_db}")

    # find best delay
    print("  Finding playback delay...", end=" ", flush=True)
    best_delay = 0.3
    best_score = -1
    delays = [0.1, 0.2, 0.3, 0.5]
    for i, delay in enumerate(delays):
        w = CAL_WORDS[i % len(CAL_WORDS)]
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(delay)
        speak(w)
        t.join()
        ref_path = get_ref_path(w)
        ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
        rec = np.frombuffer(raw[0], dtype=np.int16).astype(np.float32) / 32768.0
        ref = normalize_volume(ref)
        rec = normalize_volume(rec)
        ref, _ = librosa.effects.trim(ref, top_db=cal["top_db"])
        rec, _ = librosa.effects.trim(rec, top_db=cal["top_db"])
        if len(ref) < 1600 or len(rec) < 1600:
            continue
        d = dtw_distance(extract_mfcc(ref), extract_mfcc(rec))
        if best_score < 0 or d < best_score:
            best_score = d
            best_delay = delay
    cal["delay"] = best_delay
    print(f"delay={best_delay}s")

    # measure baseline distances
    print("  Measuring channel bias...\n")
    dists = []
    for w in CAL_WORDS:
        ref_path = get_ref_path(w)
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(cal["delay"])
        speak(w)
        t.join()

        rec = np.frombuffer(raw[0], dtype=np.int16).astype(np.float32) / 32768.0
        ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
        ref = normalize_volume(ref)
        rec = normalize_volume(rec)
        ref, _ = librosa.effects.trim(ref, top_db=cal["top_db"])
        rec, _ = librosa.effects.trim(rec, top_db=cal["top_db"])
        if len(ref) < 1600 or len(rec) < 1600:
            print(f"    {w:10s} skipped (too short after trim)")
            continue
        d = dtw_distance(extract_mfcc(ref), extract_mfcc(rec))
        dists.append(d)
        print(f"    {w:10s} dist={d:.1f}")

    if len(dists) < 3:
        print("\n  Not enough samples. Check mic/speaker.")
        return

    # cross-word distances for mismatch baseline
    print("  Measuring mismatch distances...")
    refs = {}
    for w in CAL_WORDS:
        rp = get_ref_path(w)
        r, _ = librosa.load(str(rp), sr=SAMPLE_RATE)
        r = normalize_volume(r)
        r, _ = librosa.effects.trim(r, top_db=cal["top_db"])
        if len(r) >= 1600:
            refs[w] = extract_mfcc(r)
    cross = []
    words = list(refs.keys())
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            cross.append(dtw_distance(refs[words[i]], refs[words[j]]))

    bias = np.mean(dists) * 0.9
    cal["bias"] = round(float(bias), 1)
    if cross:
        d_mis = float(np.median(cross))
        cal["scale"] = round((d_mis - bias) / 3, 1)
    else:
        cal["scale"] = round(float(np.mean(dists) * 0.6), 1)

    save_calibration()

    print(f"\n  bias: {cal['bias']}")
    print(f"  scale: {cal['scale']}")
    print(f"  top_db: {cal['top_db']}")
    print(f"  delay: {cal['delay']}s")

    # verify
    print("\n  Verifying...\n")
    for w in CAL_WORDS[:3]:
        ref_path = get_ref_path(w)
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(cal["delay"])
        speak(w)
        t.join()
        sim = audio_similarity(ref_path, raw[0])
        print(f"    {w:10s} {sim_color(sim)}{sim:3d}%{RST}")

    print(f"\n  Saved to {CAL_FILE}")


def _test_bad():
    # near-homophones: forward and reverse pairs
    pairs = [
        ("sit", "seat", "/siːt/"),
        ("seat", "sit", "/sɪt/"),
        ("leaf", "leave", "/liːv/"),
        ("leave", "leaf", "/liːf/"),
        ("bat", "bet", "/bɛt/"),
        ("bet", "bat", "/bæt/"),
        ("ship", "chip", "/tʃɪp/"),
        ("chip", "ship", "/ʃɪp/"),
        ("thin", "this", "/ðɪs/"),
        ("this", "thin", "/θɪn/"),
        ("fan", "van", "/væn/"),
        ("van", "fan", "/fæn/"),
        ("hat", "hot", "/hɑːt/"),
        ("hot", "hat", "/hæt/"),
        ("price", "prize", "/praɪz/"),
        ("prize", "price", "/praɪs/"),
        ("pool", "pull", "/pʊl/"),
        ("pull", "pool", "/puːl/"),
        ("wine", "vine", "/vaɪn/"),
        ("vine", "wine", "/waɪn/"),
    ]
    ok = 0
    for said, expected, ipa in pairs:
        raw = _ref_raw(said)
        fb = get_feedback(raw, expected, ipa)
        caught = not fb.lower().startswith("good")
        ok += caught
        mark = GRN + "pass" + RST if caught else RED + "FAIL" + RST
        print(f"  {mark}  said: {said}  labeled: {expected} {ipa} -> {fb}")
    print(f"\n  error detection: {ok}/{len(pairs)} ({ok * 100 // len(pairs)}%)\n")
    return ok, len(pairs)


def test_feedback():
    _test_bad()


def test_services():
    raw = _ref_raw("hello")
    p = data.prompts.feedback.format(word="hello", ipa="/hɛˈloʊ/")
    svcs = [
        ("gemini-flash-latest feedback", lambda:
         _ask_gemini(_raw_to_wav(raw), p)),
        ("gemini-3.1-flash-tts TTS", lambda:
         _gemini_tts_wav("test")),
    ]
    if cfg.openai:
        m = cfg.openai.model or "gemini/gemini-2.5-flash"
        svcs.append((f"openai {m} feedback", lambda:
                     _ask_openai(_raw_to_wav(raw), p)))
    svcs.append(("gTTS", lambda: speak("test")))

    for name, fn in svcs:
        print(f"  {name} ... ", end="", flush=True)
        try:
            r = fn()
            if isinstance(r, str):
                print(f"\r  {name} ... {GRN}ok{RST}  {DIM}{r[:60]}{RST}")
            else:
                print(f"\r  {name} ... {GRN}ok{RST}")
        except Exception as e:
            print(f"\r  {name} ... {RED}fail{RST}  {e}")


def main():
    p = argparse.ArgumentParser(
        description="English pronunciation trainer")
    p.add_argument("--list", action="store_true",
                   help="list phoneme groups")
    p.add_argument("--stats", action="store_true",
                   help="show performance stats")
    p.add_argument("--calibrate", action="store_true",
                   help="calibrate mic/speaker channel")
    p.add_argument("--group", "-g",
                   help="practice specific group by id")
    p.add_argument("--weak", "-w", action="store_true",
                   help="auto-select weakest group")
    p.add_argument("--continuous", "-c", action="store_true",
                   help="continuous mode, no Enter between words")
    p.add_argument("--debug", "-d", action="store_true",
                   help="show audio debug info")
    p.add_argument("--sim-threshold", type=int, default=60,
                   help="audio similarity threshold %% (default 60)")
    p.add_argument("--text", "-t",
                   help="practice a specific word or phrase")
    p.add_argument("--test-rec", action="store_true",
                   help="test recording histogram")
    p.add_argument("--test-feedback", action="store_true",
                   help="test AI feedback on mismatched words")
    p.add_argument("--test-services", action="store_true",
                   help="test available services")
    p.add_argument("--assess", action="store_true",
                   help="assess pronunciation with a pangram")
    p.add_argument("--twisters", action="store_true",
                   help="practice tongue twisters with feedback")
    p.add_argument("--voice",
                   help="TTS voice (list, male, female, random, or name)")
    a = p.parse_args()

    if a.voice:
        import random
        v = a.voice.lower()
        if v == "list":
            print("Male:  ", ", ".join(VOICES_MALE))
            print("Female:", ", ".join(VOICES_FEMALE))
            return
        elif v == "random":
            audio.voice = random.choice(VOICES_ALL)
        elif v == "male":
            audio.voice = random.choice(VOICES_MALE)
        elif v == "female":
            audio.voice = random.choice(VOICES_FEMALE)
        else:
            audio.voice = next((x for x in VOICES_ALL if x.lower() == v), v)

    if a.test_rec:
        test_rec()
        return

    global sim_threshold
    sim_threshold = a.sim_threshold
    load_calibration()
    load_cfg()
    load_data()
    build_ipa_map()
    h = load_history()

    if a.test_services:
        test_services()
        return
    if a.test_feedback:
        test_feedback()
        return

    if a.list:
        list_groups()
        return
    if a.stats:
        show_stats(h)
        return
    if a.calibrate:
        calibrate()
        return
    if a.assess:
        try:
            assess(h, a.continuous, a.debug)
        except KeyboardInterrupt:
            pass
        finally:
            restore_term()
            save_history(h)
        return
    if a.twisters:
        try:
            practice_twisters(h)
        except KeyboardInterrupt:
            pass
        return

    if a.text:
        ipa = ""
        for g in data.phonemes.values():
            for wd in g.words:
                if wd.word.lower() == a.text.lower():
                    ipa = wd.ipa
                    break
        w = Box({"word": a.text, "ipa": ipa})
        rec = sr.Recognizer()
        practice_word(w, rec, debug=a.debug)
        return

    print("English pronunciation trainer")
    print("The app will say each word, then you repeat it.")
    if not audio.calibrated:
        print(f"{DIM}Quick calibrating...{RST}", end="", flush=True)
        quick_calibrate()
        print("\r\033[K", end="")
    if a.continuous:
        print(f"{DIM}Keys: [s]kip  [f]eedback  [p]ause  [q]uit{RST}")
        print(f"{DIM}Keys work during recording and between words{RST}")
    else:
        print(f"{DIM}Keys: [Enter] retry  [s]kip  [f]eedback  [q]uit{RST}")
    print()

    if a.group:
        if a.group not in data.phonemes:
            print(f"Unknown group: {a.group}")
            print("Use --list to see available groups.")
            return
        gid = a.group
    elif a.weak:
        accs = [(gid, group_accuracy(h, gid))
                for gid in data.phonemes.keys()]
        accs.sort(key=lambda x: x[1])
        gid = accs[0][0]
    else:
        try:
            gid = select_group(h)
        except KeyboardInterrupt:
            print()
            return

    try:
        practice_phonemes(gid, h, a.continuous, a.debug)
    except KeyboardInterrupt:
        pass
    finally:
        restore_term()
        save_history(h)
        print("\nProgress saved.")


if __name__ == "__main__":
    main()
