#!/usr/bin/env python3
"""English pronunciation trainer - practice words grouped by phoneme."""

import argparse
import atexit
import os
import random
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
    SAMPLE_RATE, SAMPLE_WIDTH, CONF_DIR, CAL_WORDS, REF_DIR,
    VOICES_MALE, VOICES_FEMALE, VOICES_ALL, _BARS,
    cal, DIM, RST,
    load_calibration, save_calibration, quick_calibrate, CAL_FILE,
    status, record_audio, play_raw, speak, speak_tutor,
    ensure_ref, get_ref_path, audio_similarity,
    normalize_volume, extract_mfcc, dtw_distance,
    raw_to_sr_audio, _raw_to_wav, _gemini_tts_wav, _ref_raw,
    test_rec,
)

DATA = Path(__file__).parent / "english.yaml"
HIST = CONF_DIR / "history.yaml"
CFG_FILE = CONF_DIR / "config.yaml"
FB_LOG = CONF_DIR / "feedback.log"

cfg = Box(default_box=True)
data = Box()


def load_cfg():
    if CFG_FILE.exists():
        with open(CFG_FILE) as f:
            c = yaml.safe_load(f)
        if c:
            cfg.update(c)
    else:
        cfg.sim_voice = audio.sim_voice
        cfg.tutor_voice = audio.tutor_voice
        save_cfg()
    if cfg.get("voice") and not cfg.get("sim_voice"):
        cfg.sim_voice = cfg.pop("voice")
    if cfg.get("sim_voice"):
        audio.sim_voice = cfg.sim_voice
    if cfg.get("tutor_voice"):
        audio.tutor_voice = cfg.tutor_voice


def save_cfg():
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_FILE, "w") as f:
        yaml.dump(cfg.to_dict(), f)


def _parse_word(s):
    i = s.index(' /')
    return Box(word=s[:i], ipa=s[i + 1:])


def load_data():
    with open(DATA) as f:
        raw = yaml.safe_load(f)
    global data
    data = Box(raw)
    data.prompts.sysprompt = data.prompts.sysprompt.format(extra=_sysprompt_extra())
    aliases = {}
    for line in (data.stt_aliases or "").strip().split('\n'):
        if line:
            k, v = line.split(None, 1)
            aliases[k] = v
    data.stt_aliases = aliases
    data.homophones = [set(line.split())
                       for line in (data.homophones or "").strip().split('\n')
                       if line]
    for g in data.phonemes.values():
        g.words = [_parse_word(w) for w in g.words.strip().split('\n')]
    data.twisters = []
    for gid, g in data.phonemes.items():
        for t in (g.twisters or "").strip().split('\n'):
            if t:
                data.twisters.append({"text": t, "group": gid})


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


def hide_cursor():
    print("\033[?25l", end="", flush=True)


def show_cursor():
    print("\033[?25h", end="", flush=True)


atexit.register(show_cursor)


def wait_key(timeout=2):
    """Returns key pressed or None."""
    show_cursor()
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
    for group in data.homophones:
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


def record_word(word, rec, prefix="", duration=5, pause=0.5,
                hint="[s]kip [q]uit"):
    """Returns (heard, pct, sim, peak, dur, raw, key)."""
    bars = []

    def on_chunk(peak):
        if peak > audio._peak_max:
            audio._peak_max = peak
        mx = audio._peak_max if audio._peak_max > 1e-6 else 1.0
        bars.append(_BARS[min(8, int(peak / mx * 8))])
        print(f"\r\033[K{prefix}"
              f"{DIM}{hint}{RST} "
              f"Listening{''.join(bars)}🎤", end="", flush=True)

    try:
        sim_ref = get_ref_path(word)
        key = None
        while True:
            show_cursor()
            raw, spoke, key = record_audio(
                duration=duration, pause=pause,
                on_chunk=on_chunk,
                check_keys=bool(_term_saved))
            if key in ('s', 'q'):
                clear_line()
                print()
                return None, 0, 0, 0, 0, None, key
            peak_raw = int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))))
            if spoke and peak_raw > 1000:
                break
        hide_cursor()
        print(f"\r\033[K{prefix}", end="", flush=True)
        samples = np.frombuffer(raw, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))
        dur = len(samples) / SAMPLE_RATE
        play_raw(raw)
        print(f"\r\033[K{prefix}Processing...", end="", flush=True)
        sim = audio_similarity(sim_ref, raw) if sim_ref.exists() else 0
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
    g["last"] = str(date.today())


def _sysprompt_extra():
    # gTTS can't render IPA and inserts pauses after ~15 words
    if audio.tutor_voice == "gTTS":
        return " No IPA symbols. No more than 15 words per sentence."
    return ""


def _load_audio_file(path):
    """Load any audio file, return raw PCM bytes (16kHz mono 16-bit)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
    return seg.raw_data


def _ask_ai(raw, prompt):
    """Send audio + prompt to Gemini or OpenAI, return response text."""
    wav = _raw_to_wav(raw)
    if cfg.openai:
        fb = _ask_openai(wav, prompt)
    else:
        fb = _ask_gemini(wav, prompt)
    return re.sub(r'"(\w+)([.,])"', r'"\1"\2', fb)


_last_gemini = 0
_genai_client = None


def _ask_gemini(wav, prompt):
    global _last_gemini, _genai_client
    from google import genai
    from google.genai import types
    now = time.time()
    gap = now - _last_gemini if _last_gemini else 0
    status(f"  {DIM}gemini-flash-latest ...{RST}")
    if _genai_client is None:
        _genai_client = genai.Client()
    c = _genai_client
    cfg = types.GenerateContentConfig(system_instruction=data.prompts.sysprompt)
    for attempt in range(3):
        try:
            _last_gemini = time.time()
            r = c.models.generate_content(
                model="gemini-flash-latest",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                ],
                config=cfg,
            )
            status()
            return r.text.strip()
        except Exception as e:
            if attempt < 2 and ("400" in str(e) or "500" in str(e)):
                status(f"  {DIM}retry {attempt + 1}, {len(wav)}B"
                       f" gap={time.time() - now:.1f}s, {e}{RST}")
                time.sleep(1)
                continue
            status()
            raise Exception(
                f"{e}, wav={len(wav)}B"
                f" gap={gap:.1f}s prompt={prompt[:60]}")


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
    return _ask_ai(raw, p)


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

def sim_color(pct):
    if pct >= 60:
        return GRN
    if pct >= 30:
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
        words = ' '.join(w.word for w in g.words)
        print(f"  {gid:20s} - {g.name}")
        print(f"  {' ' * 20}   {DIM}{words}{RST}")


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
        words = ' '.join(w.word for gid in gids for w in data.phonemes[gid].words)
        print(f"  {k} - {' / '.join(names)}")
        print(f"      {DIM}{words}{RST}")
    print(f"  ? - weakest group")
    print(f"  / - least recent group")
    print()

    while True:
        print(f"{DIM}Pick a group:{RST} ", end="", flush=True)
        c = wait_key(None)
        clear_line()
        if c == '?':
            accs = [(gid, group_accuracy(h, gid))
                    for gid in data.phonemes]
            accs.sort(key=lambda x: x[1])
            floor = accs[0][1] + 0.01
            pool = [g for g, a in accs if a <= floor]
            random.shuffle(pool)
            pool.sort(key=lambda g: h["groups"].get(g, {}).get("last", ""))
            return pool[0]
        if c == '/':
            by_date = [(gid, h["groups"].get(gid, {}).get("last", ""))
                       for gid in data.phonemes]
            by_date.sort(key=lambda x: x[1])
            return random.choice([g for g, _ in by_date[:3]])
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


def _log_feedback(word, fb):
    with open(FB_LOG, "a") as f:
        f.write(f"{date.today()} {word}: {fb}\n")


_CLEAN_RE = re.compile(r'[\"\'()"/,.]')


def _speak_feedback(fb):
    for s in re.split(r'(?<=[.!]) ', fb):
        s = s.strip()
        if not s:
            continue
        print(f"  {DIM}{s}{RST}")
        speak_tutor(_CLEAN_RE.sub('', s))


def _do_feedback(raw, word, ipa, info, h=None):
    """Returns (good, feedback_text). Info line already printed."""
    try:
        fb = get_feedback(raw, word, ipa)
    except Exception as e:
        status()
        print(f"  {DIM}Feedback error: {e}{RST}")
        return False, ""
    good = fb.strip().lower().startswith("good")
    status()
    if good:
        print(f"\033[A\r\033[K{info},  {DIM}{fb}{RST}")
    else:
        _speak_feedback(fb)
    _log_feedback(word, fb)
    if h and word in h["words"]:
        h["words"][word]["feedback"] = fb
    return good, fb


def practice_word(w, rec, num="", cont=False, debug=False, prev=None, h=None):
    """Practice one word with retries. Returns (best_pct, last_raw).
    best_pct=-1 means quit. prev=(raw, word, ipa) for feedback fallback."""
    ensure_ref(w.word)

    prefix = f"{num}{w.word}  {w.ipa}  "
    best = 0
    last_raw = None
    lb_s = ""
    tries = 0
    while True:
        tries += 1
        clear_line()
        hide_cursor()
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

        if pct > best:
            best = pct

        heard_s = f"  heard: {heard}" if heard else ""
        dbg = f"  {dur:.1f}s peak={peak * 100 // 32768}%" if debug else ""
        info = f"{prefix}{lb_s}{heard_s}{dbg}"
        print(f"\r\033[K{info}")

        good, fb = _do_feedback(last_raw, w.word, w.ipa, info, h)
        if good:
            best = 100
            break

        if cont:
            k = wait_key(2)
            if k == 'q':
                return -1, last_raw
            if k == 's':
                break
            if k == 'p':
                print(f"  {DIM}Paused. Any key...{RST}",
                      end="", flush=True)
                wait_key(None)
                clear_line()
        else:
            print(f"  {DIM}[Enter] retry [s]kip{RST}",
                  end="", flush=True)
            c = wait_key(None)
            clear_line()
            if c in ('\r', '\n', None):
                pass
            elif c == "s":
                break

    return best, last_raw, tries


def _assess_one(text, ipa):
    """Record and assess one pangram. Returns (weak_groups, raw)."""
    print(f"  {text}")
    if ipa:
        print(f"  {DIM}{ipa}{RST}")
    print()
    hide_cursor()
    print(f"  {DIM}Listen first...{RST}", end="", flush=True)
    speak(text)
    print("\r\033[K", end="")
    ensure_ref(text)
    rec = sr.Recognizer()
    pw = len(text.split())
    while True:
        heard, pct, sim, peak, dur, raw, key = record_word(
            text, rec, "  ", duration=30, pause=2.0)
        if key == 'q' or not raw:
            return [], None
        hw = len(heard.split()) if heard else 0
        if hw < pw * 0.5:
            print(f"  Incomplete ({hw}/{pw} words). Try again.")
            continue
        break
    clear_line()
    if heard:
        print(f"  Heard: {heard}")
    try:
        r = get_assessment(raw, text)
    except Exception as e:
        print(f"  Analysis error: {e}")
        return [], raw
    return parse_assessment(r), raw



def assess(h, cont=False, debug=False):
    print(f"\nPronunciation assessment\n")
    weak = []
    pangrams = list(data.pangrams)
    random.shuffle(pangrams)
    for p in pangrams:
        text = p.get("text", p) if isinstance(p, dict) else p
        ipa = p.get("ipa", "") if isinstance(p, dict) else ""
        w, raw = _assess_one(text, ipa)
        for g in w:
            if g not in weak:
                weak.append(g)
        if weak:
            break
    if not weak:
        print("\n  Your pronunciation sounds good!")
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


def practice_twisters(h, cont=False):
    print(f"\nTongue twisters\n")
    if cont:
        print(f"{DIM}Keys: [s]kip  [r]epeat  [p]ause  [q]uit{RST}\n")
    set_cbreak()
    rec = sr.Recognizer()
    if "twisters" not in h:
        h["twisters"] = {}
    twisters = sorted(data.twisters,
                      key=lambda t: h["twisters"].get(t["text"], ""))
    try:
        for i, tw in enumerate(twisters):
            text = tw["text"]
            gid = tw.get("group", "")
            gn = data.phonemes[gid].name if gid in data.phonemes else ""
            lbl = f"  {i + 1}/{len(twisters)}"
            if gn:
                lbl += f"  ({gn})"
            print(lbl)
            ensure_ref(text)
            pw = len(text.split())
            while True:
                print(f"  {text}")
                hide_cursor()
                print(f"  {DIM}Listen...{RST}", end="", flush=True)
                speak(text)
                clear_line()
                while True:
                    heard, pct, sim, peak, dur, raw, key = record_word(
                        text, rec, "  ", duration=15, pause=1.5,
                        hint="[n]ext [q]uit")
                    if key == 'q':
                        return
                    if key in ('s', 'n'):
                        raw = None
                        break
                    if not raw:
                        continue
                    hw = len(heard.split()) if heard else 0
                    if hw < pw * 0.5:
                        print(f"  Too short ({hw}/{pw} words). Try again.")
                        continue
                    break
                if not raw:
                    break
                clear_line()
                if heard:
                    print(f"  Heard: {heard}")
                if sim:
                    print(f"  Audio similarity: {sim_color(sim)}{sim}%{RST}")
                h["twisters"][text] = str(date.today())
                try:
                    fb = _ask_ai(raw,
                                 data.prompts.twister.format(text=text))
                except Exception as e:
                    print(f"  Feedback error: {e}")
                    fb = None
                if fb:
                    _log_feedback(text, fb)
                    if fb.strip() == "Good":
                        print(f"  {DIM}{fb}{RST}")
                    else:
                        _speak_feedback(fb)
                if cont:
                    print()
                    k = wait_key(1)
                    if k == 'q':
                        return
                    if k == 'p':
                        print(f"  {DIM}Paused. Any key...{RST}",
                              end="", flush=True)
                        wait_key(None)
                        clear_line()
                    if k != 'r':
                        break
                else:
                    nxt = "[n]ext  " if i < len(twisters) - 1 else ""
                    print(f"\n  {DIM}{nxt}[r]epeat  [q]uit{RST}",
                          end="", flush=True)
                    c = wait_key(None)
                    clear_line()
                    if c == 'q':
                        return
                    if c != 'r':
                        break
            print()
    finally:
        restore_term()


_VOWEL_CHART = f"""\
{DIM}        Front  Central  Back
Close    iː  ɪ            ʊ  uː
Mid      eɪ       ə        oʊ
Open-mid ɛ                    ɔː
Open     æ        ʌ        ɑː{RST}"""

_VOWEL_GROUPS = {"vowel_pairs", "diphthongs", "schwa", "ei_ai", "r_vowels"}


def practice_phonemes(gid, h, cont=False, debug=False):
    g = data.phonemes[gid]
    if gid not in h["groups"]:
        h["groups"][gid] = {"attempts": 0, "correct": 0}
    h["groups"][gid]["last"] = str(date.today())
    print(f"\n{g.name}")
    print(f"  {g.description}")
    if gid in _VOWEL_GROUPS:
        print()
        print(_VOWEL_CHART)
    print()

    if cont:
        set_cbreak()

    rec = sr.Recognizer()

    words = pick_words(g.words, h)
    results = []
    prev = None  # (raw, word, ipa) for feedback on previous word

    try:
        for i, w in enumerate(words):
            best, raw, tries = practice_word(
                w, rec, f"{i + 1}/{len(words)}: ", cont, debug, prev, h)
            if raw:
                prev = (raw, w.word, w.ipa)
            if best == -1:
                break
            results.append((w.word, best, tries))
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

    hard = [(w, t) for w, _, t in results if t > 1]
    if hard:
        hard.sort(key=lambda x: -x[1])
        print()
        for w, t in hard:
            print(f"  {w} - {t} tries")

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


def test_anthropic_api():
    models = [m.strip() for m in
              os.getenv("ANTHROPIC_MODELS", "").split(",") if m.strip()]
    if not (os.getenv("ANTHROPIC_API_KEY") and
            os.getenv("ANTHROPIC_BASE_URL") and models):
        return []

    from anthropic import Anthropic
    c = Anthropic()

    def ask(m):
        r = c.messages.create(
            model=m,
            max_tokens=512,
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
        )
        return next(b.text for b in r.content
                    if b.type == "text").strip()

    return [(f"anthropic {m}", lambda m=m: ask(m)) for m in models]


def test_services():
    raw = _ref_raw("hello")
    p = data.prompts.feedback.format(word="hello", ipa="/hɛˈloʊ/")
    def _test_model(model):
        from google import genai
        from google.genai import types
        c = genai.Client()
        cfg = types.GenerateContentConfig(system_instruction=data.prompts.sysprompt)
        r = c.models.generate_content(
            model=model,
            contents=[
                p,
                types.Part.from_bytes(data=_raw_to_wav(raw), mime_type="audio/wav"),
            ],
            config=cfg,
        )
        return r.text.strip()
    svcs = [
        ("gemini-flash-latest feedback", lambda:
         _ask_gemini(_raw_to_wav(raw), p)),
        ("gemini-3.1-flash-lite feedback", lambda:
         _test_model("gemini-3.1-flash-lite")),
        ("gemini-3.1-flash-tts TTS", lambda:
         _gemini_tts_wav("test")),
    ]

    if cfg.openai:
        m = cfg.openai.model or "gemini/gemini-2.5-flash"
        svcs.append((f"openai {m} feedback", lambda:
                     _ask_openai(_raw_to_wav(raw), p)))
    svcs.extend(test_anthropic_api())
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
    p.add_argument("--old", "-o", action="store_true",
                   help="auto-select least recently practiced group")
    p.add_argument("--continuous", "-c", action="store_true",
                   help="continuous mode, no Enter between words")
    p.add_argument("--debug", "-d", action="store_true",
                   help="show audio debug info")
    p.add_argument("--text", "-t",
                   help="practice a specific word or phrase")
    p.add_argument("--test-rec", action="store_true",
                   help="test recording histogram")
    p.add_argument("--test-services", action="store_true",
                   help="test available services")
    p.add_argument("--assess", action="store_true",
                   help="assess pronunciation with a pangram")
    p.add_argument("--twisters", action="store_true",
                   help="practice tongue twisters with feedback")
    p.add_argument("--file", "-f",
                   help="audio file for offline feedback")
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
            audio.sim_voice = random.choice(VOICES_ALL)
        elif v == "male":
            audio.sim_voice = random.choice(VOICES_MALE)
        elif v == "female":
            audio.sim_voice = random.choice(VOICES_FEMALE)
        else:
            audio.sim_voice = next((x for x in VOICES_ALL if x.lower() == v), v)
    if a.test_rec:
        test_rec()
        return

    load_calibration()
    load_cfg()

    if a.voice:
        cfg.sim_voice = audio.sim_voice
        save_cfg()
    load_data()
    build_ipa_map()
    h = load_history()

    if a.test_services:
        test_services()
        return
    if a.list:
        list_groups()
        return
    if a.stats:
        show_stats(h)
        return
    if a.file:
        raw = _load_audio_file(a.file)
        text = a.text
        if not text:
            rec = sr.Recognizer()
            try:
                heard = rec.recognize_google(raw_to_sr_audio(raw))
            except (sr.UnknownValueError, sr.RequestError):
                heard = ""
            if heard:
                print(f"  {DIM}heard: {heard}{RST}")
                if a.twisters:
                    best, best_tw = 0, None
                    for tw in data.twisters:
                        s = SequenceMatcher(None, heard.lower(),
                                            tw["text"].lower()).ratio()
                        if s > best:
                            best, best_tw = s, tw
                    if best >= 0.3:
                        text = best_tw["text"]
                        print(f"  {DIM}matched: {text}{RST}")
            if not text:
                for i, tw in enumerate(data.twisters):
                    print(f"  {i + 1:2d}. {tw['text']}")
                text = data.twisters[int(input("  #: ")) - 1]["text"]
        print(f"\n  {text}")
        fb = _ask_ai(raw, data.prompts.twister.format(text=text))
        status()
        print(f"  {fb}")
        _log_feedback(text, fb)
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
            practice_twisters(h, a.continuous)
        except KeyboardInterrupt:
            pass
        finally:
            save_history(h)
        return

    if a.text:
        ipa = ""
        for g in data.phonemes.values():
            for wd in g.words:
                if wd.word.lower() == a.text.lower():
                    ipa = wd.ipa
                    break
        w = Box(word=a.text, ipa=ipa)
        rec = sr.Recognizer()
        practice_word(w, rec, debug=a.debug)
        return

    if not audio.calibrated:
        print(f"{DIM}Quick calibrating...{RST}", end="", flush=True)
        quick_calibrate()
        print("\r\033[K", end="")

    if a.group or a.weak or a.old:
        try:
            _run_phonemes(a, h)
        except KeyboardInterrupt:
            pass
        finally:
            restore_term()
            save_history(h)
        return

    print("English pronunciation trainer\n")
    print("  a - assess pronunciation")
    print("  t - tongue twisters")
    print("  p - practice phonemes")
    print()
    try:
        set_cbreak()
        print(f"{DIM}Pick:{RST} ", end="", flush=True)
        c = wait_key(None)
        clear_line()
    except KeyboardInterrupt:
        print()
        return
    finally:
        restore_term()

    try:
        if c == 'a':
            assess(h, a.continuous, a.debug)
        elif c == 't':
            practice_twisters(h, a.continuous)
        elif c == 'p':
            _run_phonemes(a, h)
        else:
            return
    except KeyboardInterrupt:
        pass
    finally:
        restore_term()
        save_history(h)


def _run_phonemes(a, h):
    if a.continuous:
        print(f"{DIM}Keys: [s]kip  [p]ause  [q]uit{RST}")
    else:
        print(f"{DIM}Keys: [Enter] retry  [s]kip  [q]uit{RST}")
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
        # include all groups within 1% of the weakest
        floor = accs[0][1] + 0.01
        pool = [g for g, a in accs if a <= floor]
        random.shuffle(pool)
        pool.sort(key=lambda g: h["groups"].get(g, {}).get("last", ""))
        gid = pool[0]
    elif a.old:
        by_date = [(gid, h["groups"].get(gid, {}).get("last", ""))
                   for gid in data.phonemes.keys()]
        by_date.sort(key=lambda x: x[1])
        gid = random.choice([g for g, _ in by_date[:3]])
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


if __name__ == "__main__":
    main()
