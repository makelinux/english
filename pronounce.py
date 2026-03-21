#!/usr/bin/env python3
"""English pronunciation trainer - practice words grouped by phoneme."""

import argparse
import ctypes
import json
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

import threading
import time

DATA = Path(__file__).parent / "words.yaml"
CONF_DIR = Path.home() / ".english-pronounce"
HIST = CONF_DIR / "history.json"
REF_DIR = CONF_DIR / "ref"
CAL_FILE = CONF_DIR / "calibration.json"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
N_MFCC = 13
CAL_WORDS = ["test", "one", "two", "three", "four", "five"]

# calibration state - loaded at startup
cal = {"bias": 0, "scale": 70, "top_db": 20, "delay": 0.3}


def load_calibration():
    if CAL_FILE.exists():
        with open(CAL_FILE) as f:
            cal.update(json.load(f))
        cal.pop("gain", None)


def save_calibration():
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)


def load_words():
    with open(DATA) as f:
        return Box(yaml.safe_load(f))


def load_history():
    if HIST.exists():
        with open(HIST) as f:
            return json.load(f)
    return {"words": {}, "groups": {}}


def save_history(h):
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST, "w") as f:
        json.dump(h, f, indent=2)


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
        return
    except Exception:
        pass
    subprocess.run(
        ["espeak-ng", "-s", "130", "-v", "en-us", text],
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
    return max(0, min(100, int(100 * (1 - d / cal["scale"]))))


def segment_similarity(mfcc_ref, mfcc_rec):
    """Compare start/middle/end thirds, return (start, mid, end) pcts."""
    n3 = len(mfcc_ref) // 3
    m3 = len(mfcc_rec) // 3
    parts = []
    for i in range(3):
        r = mfcc_ref[i * n3:(i + 1) * n3] if i < 2 else mfcc_ref[i * n3:]
        c = mfcc_rec[i * m3:(i + 1) * m3] if i < 2 else mfcc_rec[i * m3:]
        if len(r) < 2 or len(c) < 2:
            parts.append(0)
        else:
            parts.append(dist_to_pct(dtw_distance(r, c)))
    return tuple(parts)


def audio_similarity(ref_path, rec_raw):
    """Compare reference WAV with recorded raw bytes.
    Returns (total_pct, start_pct, mid_pct, end_pct)."""
    ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
    rec = np.frombuffer(rec_raw, dtype=np.int16).astype(np.float32) / 32768.0
    ref = normalize_volume(ref)
    rec = normalize_volume(rec)
    ref, _ = librosa.effects.trim(ref, top_db=cal["top_db"])
    rec, _ = librosa.effects.trim(rec, top_db=cal["top_db"])
    if len(ref) < 1600 or len(rec) < 1600:
        return 0, 0, 0, 0
    mfcc_ref = extract_mfcc(ref)
    mfcc_rec = extract_mfcc(rec)
    total = dist_to_pct(dtw_distance(mfcc_ref, mfcc_rec))
    s, m, e = segment_similarity(mfcc_ref, mfcc_rec)
    return total, s, m, e


STT_ALIASES = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10",
    "know": "no", "write": "right", "night": "knight",
    "road": "rode", "red": "read", "led": "lead",
    "sea": "see", "through": "threw",
}

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
    for words in ipa_to_words.values():
        if e in words and h in words:
            return 100
    if e in h.split():
        return 80
    r = SequenceMatcher(None, e, h).ratio()
    return int(r * 100)


def record_audio(duration=5, pause=0.8):
    """Record with VAD - stops after pause seconds of silence."""
    vad = webrtcvad.Vad(1)
    chunk_ms = 30
    chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * chunk_ms / 1000)
    max_chunks = int(duration * 1000 / chunk_ms)
    pa = pasimple.PaSimple(
        pasimple.PA_STREAM_RECORD,
        pasimple.PA_SAMPLE_S16LE,
        CHANNELS, SAMPLE_RATE,
        app_name="pronounce",
        stream_name="record",
    )
    chunks = []
    speech_started = False
    speech_run = 0
    silence = 0
    try:
        for _ in range(max_chunks):
            c = pa.read(chunk_bytes)
            chunks.append(c)
            if vad.is_speech(c, SAMPLE_RATE):
                speech_run += 1
                if speech_run >= 4:
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
    return b"".join(chunks), speech_started


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


def wait_key(timeout=2):
    """Wait up to timeout for keypress. Returns last key pressed or None."""
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        # read any buffered keys first
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


def record_word(word, rec):
    """Record, recognize, and score. Returns (heard, pct, sim, stt, segs, peak)."""
    print("  Speak now...", end=" ", flush=True)
    try:
        ref = get_ref_path(word)
        while True:
            raw, spoke = record_audio()
            peak_raw = int(np.max(np.abs(np.frombuffer(raw, dtype=np.int16))))
            if spoke and peak_raw > 1000:
                break
        clear_line()
        samples = np.frombuffer(raw, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))
        dur = len(samples) / SAMPLE_RATE
        play_raw(raw)
        print("  Processing...", end=" ", flush=True)
        # audio similarity
        if ref.exists():
            sim, seg_s, seg_m, seg_e = audio_similarity(ref, raw)
        else:
            sim, seg_s, seg_m, seg_e = 0, 0, 0, 0
        # STT
        try:
            heard = rec.recognize_google(raw_to_sr_audio(raw))
        except sr.UnknownValueError:
            heard = None
        except sr.RequestError as e:
            heard = None
        stt = stt_score(word, heard)
        pct = max(sim, stt)
        clear_line()
        return heard, pct, sim, stt, seg_s, seg_m, seg_e, peak, dur
    except Exception as e:
        clear_line()
        print(f"  Recording error: {e}")
        return None, 0, 0, 0, 0, 0, 0, 0, 0


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


def select_group(data, h):
    """Let user select a phoneme group or pick weakest."""
    groups = list(data.keys())
    print("\nPhoneme groups:\n")
    for i, gid in enumerate(groups):
        g = data[gid]
        acc = group_accuracy(h, gid)
        tag = f" {acc:.0%}" if acc >= 0 else " new"
        print(f"  {i + 1}. {g.name}{tag}")
    print(f"  {len(groups) + 1}. Weakest group (auto-select)")
    print()

    while True:
        try:
            c = input("Pick a group (number): ").strip()
            n = int(c)
        except (ValueError, EOFError):
            continue
        if n == len(groups) + 1:
            accs = [(gid, group_accuracy(h, gid)) for gid in groups]
            accs.sort(key=lambda x: x[1])
            return accs[0][0]
        if 1 <= n <= len(groups):
            return groups[n - 1]


def practice_word(w, rec, num="", cont=False, debug=False):
    """Practice one word with retries. Returns best pct, or -1 for quit."""
    print(f"\n{num}{w.word}  {w.ipa}")

    ensure_ref(w.word)

    best = 0
    while True:
        print("  Listen...", end=" ", flush=True)
        speak(w.word)
        clear_line()

        heard, pct, sim, stt, seg_s, seg_m, seg_e, peak, dur = record_word(w.word, rec)
        if pct > best:
            best = pct

        heard_s = f"  heard: {heard}" if heard else ""
        dbg = f"  {dur:.1f}s peak={peak * 100 // 32768}%" if debug else ""
        print(f"  {pct_bar(pct)} {pct}%{heard_s}"
              f"  {pct_block(sim)}{sim:2d}%:"
              f" {pct_block(seg_s)}{seg_s:2d}%"
              f" {pct_block(seg_m)}{seg_m:2d}%"
              f" {pct_block(seg_e)}{seg_e:2d}%{dbg}")

        if pct >= 80:
            break

        if cont:
            k = wait_key(2)
            if k == 'q':
                return -1
            if k == 's':
                break
            if k == 'p':
                input("  Paused. [Enter] resume: ")
                clear_line()
                print("\033[A\033[K", end="", flush=True)
        else:
            try:
                c = input("  [Enter] retry, [s] skip: ").strip()
                clear_line()
                print("\033[A\033[K", end="", flush=True)
            except EOFError:
                break
            if c == "s":
                break

    return best


def practice(data, gid, h, cont=False, debug=False):
    """Run a practice session for a phoneme group."""
    g = data[gid]
    print(f"\n--- {g.name} ---")
    print(f"  {g.description}")
    if cont:
        print(f"  [p] pause, [s] skip, [q] quit\n")
    else:
        print(f"  [s] skip, [q] quit.\n")

    rec = sr.Recognizer()

    words = pick_words(g.words, h)
    results = []

    for i, w in enumerate(words):
        best = practice_word(w, rec, f"{i + 1}/{len(words)}: ", cont, debug)
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
                    input("  Paused. [Enter] resume: ")
                    clear_line()
                    print("\033[A\033[K", end="", flush=True)
            else:
                try:
                    c = input("\n  [Enter] next word, [q] quit: ").strip()
                    clear_line()
                    print("\033[A\033[K", end="", flush=True)
                except EOFError:
                    break
                if c == "q":
                    break

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
    raw, _ = record_audio(duration=2)  # silence sample
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
    for w in CAL_WORDS:
        ref_path = get_ref_path(w)
        raw = [None]
        t = threading.Thread(target=lambda: raw.__setitem__(0, record_audio(3)[0]))
        t.start()
        time.sleep(cal["delay"])
        speak(w)
        t.join()

        ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
        rec = np.frombuffer(raw[0], dtype=np.int16).astype(np.float32) / 32768.0
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
        sim, s, m, e = audio_similarity(ref_path, raw[0])
        print(f"    {w:10s} {pct_block(sim)}{sim:3d}%:"
              f" {pct_block(s)}{s:2d}%"
              f" {pct_block(m)}{m:2d}%"
              f" {pct_block(e)}{e:2d}%")

    print(f"\n  Saved to {CAL_FILE}")


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
    a = p.parse_args()

    load_calibration()
    data = load_words()
    build_ipa_map(data)
    h = load_history()

    if a.list:
        list_groups(data)
        return
    if a.stats:
        show_stats(h)
        return
    if a.calibrate:
        calibrate()
        return

    print("English pronunciation trainer")
    print("The app will say each word, then you repeat it.\n")

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
        save_history(h)
        print("\n\nProgress saved.")


if __name__ == "__main__":
    main()
