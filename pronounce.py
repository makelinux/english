#!/usr/bin/env python3
"""English pronunciation trainer - practice words grouped by phoneme."""

import argparse
import ctypes
import logging
import os
import re
import select
import subprocess
import sys
import termios
import tty
from datetime import date
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path

logging.disable(logging.CRITICAL)
os.environ["LIBVA_MESSAGING_LEVEL"] = "0"

# suppress ALSA errors printed by C libraries
try:
    asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    asound.snd_lib_error_set_handler(
        ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p)(lambda *_: None))
except OSError:
    pass

import threading
import time
import wave

import librosa
import numpy as np
import pasimple
import speech_recognition as sr
import webrtcvad
import yaml
from box import Box
from gtts import gTTS
from pydub import AudioSegment
from scipy.spatial.distance import cdist

DATA = Path(__file__).parent / "words.yaml"
CONF_DIR = Path.home() / ".english-pronounce"
HIST = CONF_DIR / "history.yaml"
REF_DIR = CONF_DIR / "ref"
CAL_FILE = CONF_DIR / "calibration.yaml"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
N_MFCC = 13
CAL_WORDS = ["test", "sip", "one", "two", "three", "four", "five"]

# calibration state - loaded at startup
cal = {"bias": 0, "scale": 70, "top_db": 20, "delay": 0.3}
calibrated = False


def load_calibration():
    global calibrated, _vu_max
    if CAL_FILE.exists():
        with open(CAL_FILE) as f:
            cal.update(yaml.safe_load(f))
        cal.pop("gain", None)
        calibrated = True
        _vu_max = cal.get("vu_peak", 0.2) * 0.5


def save_calibration():
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_FILE, "w") as f:
        yaml.dump(cal, f)


def load_words():
    with open(DATA) as f:
        raw = yaml.safe_load(f)
    STT_ALIASES.update(raw.pop("stt_aliases", {}))
    for group in raw.pop("stt_equiv", []):
        STT_EQUIV.append(set(group))
    return Box(raw)


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


def get_ref_path(word):
    """Path to cached reference WAV for a word."""
    return REF_DIR / f"{word}.wav"


def ensure_ref(word):
    """Generate and cache gTTS reference audio as 16kHz mono WAV."""
    p = get_ref_path(word)
    if p.exists():
        return p
    REF_DIR.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    gTTS(word, lang="en").write_to_fp(buf)
    buf.seek(0)
    seg = AudioSegment.from_mp3(buf)
    seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
    seg.export(str(p), format="wav")
    return p


def speak(text):
    """Say the word aloud from cached reference, espeak-ng fallback."""
    try:
        ref = ensure_ref(text)
        seg = AudioSegment.from_wav(str(ref))
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK,
            pasimple.PA_SAMPLE_S16LE,
            seg.channels,
            seg.frame_rate,
            app_name="pronounce",
        ) as pa:
            pa.write(seg.raw_data)
            pa.drain()
    except Exception:
        subprocess.run(
            ["espeak-ng", "-s", "130", "-v", "en-us", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def speak_text(text):
    """Speak arbitrary text via gTTS, no caching."""
    try:
        buf = BytesIO()
        gTTS(text, lang="en").write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_mp3(buf)
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
        seg -= 10
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK,
            pasimple.PA_SAMPLE_S16LE,
            1, SAMPLE_RATE,
            app_name="pronounce",
        ) as pa:
            pa.write(seg.raw_data)
            pa.drain()
    except Exception:
        subprocess.run(
            ["espeak-ng", "-s", "150", "-v", "en-us", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_mfcc(samples, sr_rate=SAMPLE_RATE):
    """Extract MFCC + delta features, skip c0, mean-centered."""
    m = librosa.feature.mfcc(y=samples, sr=sr_rate, n_mfcc=N_MFCC)
    w = min(9, m.shape[1])
    if w >= 3:
        d = librosa.feature.delta(m, width=w if w % 2 else w - 1)
        feat = np.vstack([m[1:], d[1:]]).T
    else:
        feat = m[1:].T
    feat = feat - feat.mean(axis=0)
    return feat


def dtw_distance(a, b):
    """DTW distance between two MFCC sequences."""
    cost = cdist(a, b, metric="euclidean")
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = cost[i - 1, j - 1] + min(
                d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[n, m] / (n + m)


def normalize_volume(samples):
    """Normalize audio to peak amplitude."""
    peak = np.max(np.abs(samples))
    if peak < 1e-6:
        return samples
    return samples / peak


def dist_to_pct(dist):
    d = max(0, dist - cal["bias"])
    return max(0, min(100, int(100 * (1 - d / (cal["scale"] * 3)))))


def audio_similarity(ref_path, rec_raw):
    """Compare reference WAV with recorded raw bytes. Returns pct 0-100."""
    ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
    rec = np.frombuffer(rec_raw, dtype=np.int16).astype(np.float32) / 32768.0
    ref = normalize_volume(ref)
    rec = normalize_volume(rec)
    ref, _ = librosa.effects.trim(ref, top_db=cal["top_db"])
    rec, _ = librosa.effects.trim(rec, top_db=cal["top_db"])
    if len(ref) < 1600 or len(rec) < 1600:
        return 0
    return dist_to_pct(dtw_distance(extract_mfcc(ref), extract_mfcc(rec)))


STT_ALIASES = {}
STT_EQUIV = []

# IPA-based homophone lookup, built from words.yaml
ipa_to_words = {}  # "/raɪt/" -> {"right", "write"}


def build_ipa_map(data):
    """Build IPA -> words mapping for homophone detection."""
    for g in data.values():
        for w in g.words:
            for ipa in re.findall(r'/[^/]+/', w.ipa):
                ipa_to_words.setdefault(ipa, set()).add(w.word.lower())


def stt_score(expected, heard):
    """Score based on speech-to-text match, return 0-100."""
    if not heard:
        return 0
    e = expected.lower().strip()
    h = heard.lower().strip()
    if e == h:
        return 100
    if STT_ALIASES.get(e) == h or STT_ALIASES.get(h) == e:
        return 100
    for group in STT_EQUIV:
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


_VU_BLOCKS = ".▁▂▃▄▅▆▇█"


def record_audio(duration=5, pause=0.8, on_chunk=None):
    """Record with VAD - stops after pause seconds of silence.
    on_chunk(peak_0_1) called every ~200ms with peak amplitude.
    Returns (raw_bytes, speech_started, key_pressed)."""
    vad = webrtcvad.Vad(3)
    chunk_ms = 30
    chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * chunk_ms / 1000)
    max_chunks = int(duration * 1000 / chunk_ms)
    pa = pasimple.PaSimple(
        pasimple.PA_STREAM_RECORD,
        pasimple.PA_SAMPLE_S16LE,
        CHANNELS, SAMPLE_RATE,
        app_name="pronounce",
        stream_name="record",
        fragsize=chunk_bytes,
    )
    chunks = []
    speech_started = False
    speech_run = 0
    silence = 0
    key = None
    vu_peak = 0.0
    vu_count = 0
    try:
        for _ in range(max_chunks):
            c = pa.read(chunk_bytes)
            chunks.append(c)
            # VU meter - emit one bar per 300ms (10 chunks)
            if on_chunk:
                p = float(np.max(np.abs(
                    np.frombuffer(c, dtype=np.int16)))) / 32768.0
                if p > vu_peak:
                    vu_peak = p
                vu_count += 1
                if vu_count >= 10:
                    on_chunk(vu_peak)
                    vu_peak = 0.0
                    vu_count = 0
            # check for keypress (works in cbreak mode)
            if _term_saved and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if speech_started:
                    break
            is_speech = vad.is_speech(c, SAMPLE_RATE)
            # reject keyboard clicks: speech has more low-freq energy
            if is_speech:
                s = np.frombuffer(c, dtype=np.int16).astype(np.float32)
                fft = np.abs(np.fft.rfft(s))
                cutoff = len(fft) * 2000 // (SAMPLE_RATE // 2)
                if np.sum(fft[cutoff:]) > np.sum(fft[:cutoff]) * 2:
                    is_speech = False
            if is_speech:
                speech_run += 1
                if speech_run >= 8:
                    speech_started = True
                silence = 0
            elif speech_started:
                silence += chunk_ms
                if silence >= pause * 1000:
                    break
            else:
                speech_run = 0
    finally:
        pa.close()
    return b"".join(chunks), speech_started, key


def normalize_raw(raw):
    """Normalize raw int16 bytes to ~50% peak amplitude."""
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(a))
    if peak < 100:
        return raw
    a = a / peak * 16000
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


def raw_to_sr_audio(raw):
    """Convert raw bytes to sr.AudioData for Google recognition."""
    pad = b'\x00' * (SAMPLE_RATE * SAMPLE_WIDTH)  # 1s silence
    norm = normalize_raw(raw)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pad + norm + pad)
    return sr.AudioData(buf.getvalue(), SAMPLE_RATE, SAMPLE_WIDTH)


def clear_line():
    print("\r\033[K", end="", flush=True)


_term_saved = None


def set_cbreak():
    """Set terminal to cbreak mode for immediate key reads."""
    global _term_saved
    _term_saved = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def restore_term():
    """Restore terminal to original mode."""
    global _term_saved
    if _term_saved:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _term_saved)
        _term_saved = None


def wait_key(timeout=2):
    """Wait up to timeout for keypress. Returns last key pressed or None."""
    if _term_saved:
        # already in cbreak - just read
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
    # set cbreak temporarily
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


def play_raw(raw, volume=0.3):
    """Play back raw recorded audio at normalized volume."""
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(a))
    if peak < 500:
        return
    a = a / peak * 32767 * volume
    out = a.astype(np.int16).tobytes()
    with pasimple.PaSimple(
        pasimple.PA_STREAM_PLAYBACK,
        pasimple.PA_SAMPLE_S16LE,
        CHANNELS, SAMPLE_RATE,
        app_name="pronounce",
    ) as pa:
        pa.write(out)
        pa.drain()


_vu_max = 0.2


def record_word(word, rec, prefix=""):
    """Record, recognize, and score.
    Returns (heard, pct, sim, peak, dur, raw, key).
    key is set if user pressed a key during recording."""
    global _vu_max
    vu = []

    def on_chunk(peak):
        global _vu_max
        if peak > _vu_max:
            _vu_max = peak
        mx = _vu_max if _vu_max > 1e-6 else 1.0
        vu.append(_VU_BLOCKS[min(8, int(peak / mx * 8))])
        print(f"\r\033[K{prefix}Listening{''.join(vu)}🎤",
              end="", flush=True)

    try:
        ref = get_ref_path(word)
        key = None
        while True:
            raw, spoke, key = record_audio(on_chunk=on_chunk)
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
        # audio similarity (only meaningful with calibration)
        if calibrated and ref.exists():
            sim = audio_similarity(ref, raw)
        else:
            sim = 0
        # STT
        try:
            heard = rec.recognize_google(raw_to_sr_audio(raw))
        except sr.UnknownValueError:
            heard = None
        except sr.RequestError:
            heard = None
        pct = max(sim, stt_score(word, heard))
        return heard, pct, sim, peak, dur, raw, key
    except Exception as e:
        print(f"\r\033[K{prefix}Recording error: {e}")
        return None, 0, 0, 0, 0, None, None


def group_accuracy(h, gid):
    """Get accuracy for a phoneme group from history."""
    g = h["groups"].get(gid)
    if not g or g["attempts"] == 0:
        return -1
    return g["correct"] / g["attempts"]


def pick_words(words, h, count=10):
    """Prioritize words with low accuracy or not yet practiced."""
    scored = []
    for w in words:
        s = h["words"].get(w.word)
        if not s or s["attempts"] == 0:
            scored.append((w, -1))
        else:
            scored.append((w, s["correct"] / s["attempts"]))
    scored.sort(key=lambda x: x[1])
    return [w for w, _ in scored[:count]]


def update_history(h, gid, word, pct):
    """Update history with attempt result (percentage 0-100)."""
    if word not in h["words"]:
        h["words"][word] = {"attempts": 0, "correct": 0, "last": ""}
    h["words"][word]["attempts"] += 1
    h["words"][word]["correct"] += pct / 100
    h["words"][word]["last"] = str(date.today())
    if gid not in h["groups"]:
        h["groups"][gid] = {"attempts": 0, "correct": 0}
    h["groups"][gid]["attempts"] += 1
    h["groups"][gid]["correct"] += pct / 100


DIM = "\033[2m"


def get_feedback(raw, word, ipa):
    """Ask Gemini for pronunciation feedback on recorded audio."""
    try:
        import google.generativeai as genai
    except ImportError:
        return "Install google-generativeai: pip install google-generativeai"
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "Set GEMINI_API_KEY environment variable"
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-flash-latest")
    # convert raw to wav bytes
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(normalize_raw(raw))
    wav = buf.getvalue()
    prompt = (
        f"You are a pronunciation coach for American English. "
        f"The speaker is trying to say \"{word}\" (IPA: {ipa}). "
        f"Listen to the audio. Does it sound like \"{word}\"? "
        f"If yes, respond ONLY with the word \"Good\" and nothing "
        f"else. Homophones of \"{word}\" count as correct. "
        f"Accent variations are acceptable. "
        f"Only if a clearly different word is spoken, say what you "
        f"heard and how to fix it in 2-3 sentences. "
        f"Address the speaker as \"you\". No IPA symbols."
    )
    r = model.generate_content([
        prompt,
        {"mime_type": "audio/wav", "data": wav},
    ])
    return r.text.strip()


RED = "\033[31m"
YEL = "\033[33m"
GRN = "\033[32m"
RST = "\033[0m"


def pct_color(pct):
    if pct >= 70:
        return GRN
    if pct >= 40:
        return YEL
    return RED


def pct_bar(pct, width=20):
    """Render percentage as colored #/. bar."""
    n = pct * width // 100
    c = pct_color(pct)
    return f"{c}{'#' * n}{'.' * (width - n)}{RST}"


def pct_block(pct):
    """Single colored unicode block character for percentage."""
    c = pct_color(pct)
    if pct >= 70:
        return f"{c}\u2588{RST}"
    if pct >= 40:
        return f"{c}\u2585{RST}"
    if pct >= 15:
        return f"{c}\u2582{RST}"
    return f"{c} {RST}"


def show_stats(h):
    """Display performance statistics."""
    if not h["groups"]:
        print("No practice history yet.")
        return
    print("\nPerformance by phoneme group:\n")
    for gid, g in sorted(h["groups"].items()):
        acc = g["correct"] / g["attempts"] if g["attempts"] else 0
        print(f"  {gid:20s} {pct_bar(int(acc * 100))} {acc:.0%}"
              f"  ({g['attempts']} attempts)")

    print("\nWeakest words:\n")
    weak = []
    for w, s in h["words"].items():
        if s["attempts"] >= 2:
            acc = s["correct"] / s["attempts"]
            weak.append((w, acc, s["attempts"]))
    weak.sort(key=lambda x: x[1])
    for w, acc, att in weak[:10]:
        print(f"  {w:20s} {acc:.0%}  ({att} attempts)")
    if not weak:
        print("  Need at least 2 attempts per word to rank.")


def list_groups(data):
    """Show available phoneme groups."""
    print("\nPhoneme groups:\n")
    for gid, g in data.items():
        n = len(g.words)
        print(f"  {gid:20s} - {g.name} ({n} words)")


GROUP_KEYS = {
    "th_voiced": "t", "th_voiceless": "t", "sh_ch": "s",
    "r_l": "r", "vowel_pairs": "v", "silent_letters": "i",
    "word_stress": "w", "v_w": "a", "s_z": "z",
    "diphthongs": "d", "consonant_clusters": "c",
    "ed_endings": "e", "schwa": "e", "ng_sound": "n",
    "j_dj": "j", "ei_ai": "y", "ough_spellings": "o",
    "r_vowels": "l", "zh_sound": "k", "h_sound": "h",
    "tion_sion": "g", "f_v": "f",
}


def select_group(data, h):
    """Let user select a phoneme group or pick weakest."""
    groups = list(data.keys())
    # build key -> [gid, ...] mapping (supports submenus)
    keys = {}
    for gid in groups:
        k = GROUP_KEYS.get(gid)
        if not k:
            used = set(GROUP_KEYS.values()) | {'?'}
            for ch in gid:
                if ch.isalpha() and ch not in used:
                    k = ch
                    break
        if k:
            keys.setdefault(k, []).append(gid)

    print("\nPhoneme groups:\n")
    sorted_keys = sorted(keys.keys())
    for k in sorted_keys:
        gids = keys[k]
        names = []
        for gid in gids:
            g = data[gid]
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
            accs = [(gid, group_accuracy(h, gid)) for gid in groups]
            accs.sort(key=lambda x: x[1])
            return accs[0][0]
        if c in keys:
            gids = keys[c]
            if len(gids) == 1:
                return gids[0]
            # submenu
            for i, gid in enumerate(gids):
                g = data[gid]
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


def _do_feedback(raw, word, ipa):
    """Get and display feedback, speak it."""
    try:
        fb = get_feedback(raw, word, ipa)
    except Exception as e:
        print(f"  {DIM}Feedback error: {e}{RST}")
        return
    print(f"  {DIM}{fb}{RST}")
    speak_text(re.sub(r'[\"\'()"/]', '', fb))


def practice_word(w, rec, num="", cont=False, debug=False, prev=None):
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
            lb_s = f"selfcheck={audio_similarity(get_ref_path(w.word), lb[0])}%  "
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
            _do_feedback(fb_raw, fb_word, fb_ipa)
            continue

        if pct > best:
            best = pct

        heard_s = f"  heard: {heard}" if heard else ""
        dbg = f"  {dur:.1f}s peak={peak * 100 // 32768}%" if debug else ""
        sim_s = f"  {pct_block(sim)}{sim:2d}%" if sim else ""
        bar_s = f"{pct_bar(pct)} {pct}%" if sim and not cont else ""
        score = f"{bar_s}{heard_s}{sim_s}{dbg}"
        print(f"\r\033[K{prefix}{lb_s}{score}")

        if pct >= 80:
            break

        if cont:
            k = wait_key(2)
            if k == 'q':
                return -1, last_raw
            if k == 's':
                break
            if k == 'f' and last_raw:
                _do_feedback(last_raw, w.word, w.ipa)
            elif k == 'p':
                print(f"  {DIM}Paused. Any key...{RST}",
                      end="", flush=True)
                wait_key(None)
                clear_line()
        else:
            print(f"  {DIM}[Enter] retry [s] skip [f] feedback{RST}",
                  end="", flush=True)
            c = wait_key(None)
            clear_line()
            if c in ('\r', '\n', None):
                pass
            elif c == "s":
                break
            elif c == "f" and last_raw:
                _do_feedback(last_raw, w.word, w.ipa)

    return best, last_raw


def practice(data, gid, h, cont=False, debug=False):
    """Run a practice session for a phoneme group."""
    g = data[gid]
    print(f"\n--- {g.name} ---")
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
                w, rec, f"{i + 1}/{len(words)}: ", cont, debug, prev)
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
                    print(f"  {DIM}[Enter] next [q] quit{RST}",
                          end="", flush=True)
                    c = wait_key(None)
                    clear_line()
                    if c == "q":
                        break
    finally:
        if cont:
            restore_term()

    # session summary
    total = len(results)
    if not total:
        return
    avg = sum(p for _, p in results) / total
    good = sum(1 for _, p in results if p >= 80)
    ok = sum(1 for _, p in results if 40 <= p < 80)
    bad = sum(1 for _, p in results if p < 40)

    print(f"\n--- Session summary ---")
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
    """Play words through speaker, record via mic, find bias and scale."""
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
    for delay in [0.1, 0.2, 0.3, 0.5]:
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(delay)
        speak(CAL_WORDS[0])
        t.join()
        ref_path = get_ref_path(CAL_WORDS[0])
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
    peak_max = 0.0
    for w in CAL_WORDS:
        ref_path = get_ref_path(w)
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(cal["delay"])
        speak(w)
        t.join()

        rec = np.frombuffer(raw[0], dtype=np.int16).astype(np.float32) / 32768.0
        p = float(np.max(np.abs(rec)))
        if p > peak_max:
            peak_max = p
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

    bias = np.mean(dists) * 0.9
    # scale so that bias distance -> ~95%, and 2x bias -> ~0%
    cal["bias"] = round(float(bias), 1)
    cal["scale"] = round(float(np.mean(dists) * 0.6), 1)
    cal["vu_peak"] = round(float(peak_max), 3)

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
        print(f"    {w:10s} {pct_block(sim)}{sim:3d}%")

    print(f"\n  Saved to {CAL_FILE}")


def test_vu():
    """Test VU meter - show live mic levels."""
    print("VU meter test - speak into mic, Ctrl+C to stop\n")
    chunk_ms = 30
    chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * chunk_ms / 1000)
    pa = pasimple.PaSimple(
        pasimple.PA_STREAM_RECORD,
        pasimple.PA_SAMPLE_S16LE,
        CHANNELS, SAMPLE_RATE,
        app_name="pronounce",
        fragsize=chunk_bytes,
    )
    peak = 0.0
    count = 0
    try:
        while True:
            c = pa.read(chunk_bytes)
            p = float(np.max(np.abs(
                np.frombuffer(c, dtype=np.int16)))) / 32768.0
            if p > peak:
                peak = p
            count += 1
            if count >= 10:
                print(_VU_BLOCKS[min(8, int(peak * 40))], end="", flush=True)
                peak = 0.0
                count = 0
    except KeyboardInterrupt:
        pass
    finally:
        pa.close()
    print()


def _ref_raw(word):
    """Get raw audio bytes from gTTS reference."""
    ref = ensure_ref(word)
    seg = AudioSegment.from_wav(str(ref))
    return seg.set_channels(1).set_frame_rate(
        SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH).raw_data


def _test_good():
    """Test on correct audio, returns (pass, total)."""
    words = [
        ("sit", "/sɪt/"), ("seat", "/siːt/"),
        ("leaf", "/liːf/"), ("leave", "/liːv/"),
        ("bat", "/bæt/"), ("bet", "/bɛt/"),
        ("ship", "/ʃɪp/"), ("chip", "/tʃɪp/"),
        ("thin", "/θɪn/"), ("this", "/ðɪs/"),
        ("fan", "/fæn/"), ("van", "/væn/"),
        ("hat", "/hæt/"), ("hot", "/hɑːt/"),
        ("bird", "/bɜːrd/"), ("work", "/wɜːrk/"),
        ("prize", "/praɪz/"), ("price", "/praɪs/"),
        ("pool", "/puːl/"), ("pull", "/pʊl/"),
    ]
    ok = 0
    for word, ipa in words:
        raw = _ref_raw(word)
        fb = get_feedback(raw, word, ipa)
        good = fb.lower().startswith("good")
        ok += good
        mark = GRN + "pass" + RST if good else RED + "FAIL" + RST
        print(f"  {mark}  {word} {ipa} -> {fb}")
    print(f"\n  etalon accuracy: {ok}/{len(words)} ({ok * 100 // len(words)}%)\n")
    return ok, len(words)


def _test_bad():
    """Test on wrong audio, returns (pass, total)."""
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


def test_feedback(mode):
    """Test Gemini feedback on gTTS reference pronunciation."""
    if mode == "good":
        _test_good()
    elif mode == "bad":
        _test_bad()
    else:
        g_ok, g_n = _test_good()
        b_ok, b_n = _test_bad()
        total = g_ok + b_ok
        n = g_n + b_n
        print(f"  overall: {total}/{n} ({total * 100 // n}%)")


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
    p.add_argument("--text", "-t",
                   help="practice a specific word or phrase")
    p.add_argument("--vu", action="store_true",
                   help="test VU meter (mic level)")
    p.add_argument("--test-feedback", nargs="?", const="both",
                   choices=["good", "bad", "both"],
                   help="test Gemini feedback: good=etalon, bad=wrong word")
    a = p.parse_args()

    if a.vu:
        test_vu()
        return

    load_calibration()
    data = load_words()
    build_ipa_map(data)
    h = load_history()

    if a.test_feedback:
        test_feedback(a.test_feedback)
        return

    if a.list:
        list_groups(data)
        return
    if a.stats:
        show_stats(h)
        return
    if a.calibrate:
        calibrate()
        return

    if a.text:
        # find word in data for IPA, or use empty
        ipa = ""
        for g in data.values():
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
    if not calibrated:
        print(f"{DIM}Run --calibrate for audio similarity scoring{RST}")
    if a.continuous:
        print(f"{DIM}Keys: [s] skip  [f] feedback  [p] pause  [q] quit{RST}")
        print(f"{DIM}Keys work during recording and between words{RST}")
    else:
        print(f"{DIM}Keys: [Enter] retry  [s] skip  [f] feedback  [q] quit{RST}")
    print()

    if a.group:
        if a.group not in data:
            print(f"Unknown group: {a.group}")
            print("Use --list to see available groups.")
            return
        gid = a.group
    elif a.weak:
        accs = [(gid, group_accuracy(h, gid)) for gid in data.keys()]
        accs.sort(key=lambda x: x[1])
        gid = accs[0][0]
    else:
        gid = select_group(data, h)

    try:
        practice(data, gid, h, a.continuous, a.debug)
    except KeyboardInterrupt:
        pass
    finally:
        restore_term()
        save_history(h)
        print("\nProgress saved.")


if __name__ == "__main__":
    main()
