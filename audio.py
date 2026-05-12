"""Audio recording, playback, processing and TTS utilities."""

import ctypes
import logging
import os
import select
import subprocess
import sys
import threading
import time
import wave
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

import librosa
import numpy as np
import pasimple
import speech_recognition as sr
import webrtcvad
import yaml
from gtts import gTTS
from pydub import AudioSegment
from scipy.spatial.distance import cdist

CONF_DIR = Path.home() / ".config" / "english-pronounce"
REF_DIR = CONF_DIR / "ref"
CAL_FILE = CONF_DIR / "calibration.yaml"

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
N_MFCC = 13
CAL_WORDS = ["one", "two", "three", "four", "five"]
VOICES_MALE = [
    "Achird", "Algenib", "Algieba", "Alnilam", "Charon",
    "Enceladus", "Fenrir", "Iapetus", "Orus", "Puck",
    "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar",
    "Umbriel", "Zubenelgenubi",
]
VOICES_FEMALE = [
    "Achernar", "Aoede", "Autonoe", "Callirrhoe", "Despina",
    "Erinome", "Gacrux", "Kore", "Laomedeia", "Leda",
    "Pulcherrima", "Sulafat", "Vindemiatrix", "Zephyr",
]
VOICES_ALL = VOICES_MALE + VOICES_FEMALE

# state
cal = {"bias": 0, "scale": 70, "top_db": 20, "delay": 0.3}
calibrated = False
_peak_max = 0.2
voice = "Puck"

DIM = "\033[2m"
RST = "\033[0m"
_BARS = ".▁▂▃▄▅▆▇█"


def status(msg=''):
    print(f"\r{msg}\033[K", end='', flush=True, file=sys.stderr)


def load_calibration():
    global calibrated, _peak_max
    if CAL_FILE.exists():
        with open(CAL_FILE) as f:
            cal.update(yaml.safe_load(f))
        cal.pop("gain", None)
        calibrated = True
        cal.pop("mic_peak", None)
        cal.pop("vu_peak", None)


def save_calibration():
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_FILE, "w") as f:
        yaml.dump(cal, f)


def get_ref_path(word, v=None):
    import hashlib
    name = word if len(word) < 80 else hashlib.md5(word.encode()).hexdigest()
    return REF_DIR / f"{name}-{v or voice}.wav"


def _gemini_tts_wav(text, v=None):
    """Returns AudioSegment or None."""
    try:
        from google import genai
        from google.genai import types
        status(f"  {DIM}gemini-3.1-flash-tts ...{RST}")
        c = genai.Client()
        r = c.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=v or voice
                        )
                    )
                )
            )
        )
        d = r.candidates[0].content.parts[0].inline_data.data
        seg = AudioSegment(data=d, sample_width=2,
                           frame_rate=24000, channels=1)
        status()
        return seg.set_channels(1).set_frame_rate(SAMPLE_RATE) \
                  .set_sample_width(SAMPLE_WIDTH)
    except Exception:
        status()
        return None


REF_MAX_MB = 200


def _trim_ref_cache():
    """Remove least recently accessed files if cache exceeds REF_MAX_MB."""
    files = list(REF_DIR.glob("*.wav"))
    total = sum(f.stat().st_size for f in files)
    if total <= REF_MAX_MB * 1024 * 1024:
        return
    files.sort(key=lambda f: f.stat().st_atime)
    while total > REF_MAX_MB * 1024 * 1024 * 0.8 and files:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink()


def ensure_ref(word, v=None):
    """Cache reference audio - Gemini TTS first, gTTS fallback."""
    v = v or voice
    p = get_ref_path(word, v)
    if p.exists():
        return p
    REF_DIR.mkdir(parents=True, exist_ok=True)
    seg = None if v == "gTTS" else _gemini_tts_wav(word, v)
    if not seg:
        buf = BytesIO()
        gTTS(word, lang="en").write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_mp3(buf)
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE) \
                  .set_sample_width(SAMPLE_WIDTH)
    seg.export(str(p), format="wav")
    _trim_ref_cache()
    return p


def speak(text):
    try:
        ref = ensure_ref(text)
        seg = AudioSegment.from_wav(str(ref))
        seg = seg + (-10)
        with pasimple.PaSimple(
            pasimple.PA_STREAM_PLAYBACK,
            pasimple.PA_SAMPLE_S16LE,
            1, SAMPLE_RATE,
            app_name="pronounce",
        ) as pa:
            pa.write(seg.raw_data)
            pa.drain()
    except Exception as e:
        print(f"\n  speak error: {e}", file=sys.stderr)
        subprocess.run(
            ["espeak-ng", "-s", "130", "-v", "en-us", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_mfcc(samples, sr_rate=SAMPLE_RATE):
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
    peak = np.max(np.abs(samples))
    if peak < 1e-6:
        return samples
    return samples / peak


def dist_to_pct(dist):
    d = max(0, dist - cal["bias"])
    return max(0, min(100, int(100 * (1 - d / (cal["scale"] * 3)))))


def audio_similarity(ref_path, rec_raw):
    """Returns 0-100."""
    ref, _ = librosa.load(str(ref_path), sr=SAMPLE_RATE)
    rec = np.frombuffer(rec_raw, dtype=np.int16).astype(np.float32) / 32768.0
    ref = normalize_volume(ref)
    rec = normalize_volume(rec)
    ref, _ = librosa.effects.trim(ref, top_db=cal["top_db"])
    rec, _ = librosa.effects.trim(rec, top_db=cal["top_db"])
    if len(ref) < 1600 or len(rec) < 1600:
        return 0
    return dist_to_pct(dtw_distance(extract_mfcc(ref), extract_mfcc(rec)))


def normalize_raw(raw):
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(a))
    if peak < 100:
        return raw
    a = a / peak * 16000
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


def raw_to_sr_audio(raw):
    pad = b'\x00' * (SAMPLE_RATE * SAMPLE_WIDTH)
    norm = normalize_raw(raw)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pad + norm + pad)
    return sr.AudioData(buf.getvalue(), SAMPLE_RATE, SAMPLE_WIDTH)


def _raw_to_wav(raw):
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(normalize_raw(raw))
    return buf.getvalue()


def record_audio(duration=5, pause=0.8, on_chunk=None, check_keys=False):
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
    pk = 0.0
    pk_n = 0
    try:
        for _ in range(max_chunks):
            c = pa.read(chunk_bytes)
            chunks.append(c)
            # VU meter - emit one bar per 300ms (10 chunks)
            if on_chunk:
                p = float(np.max(np.abs(
                    np.frombuffer(c, dtype=np.int16)))) / 32768.0
                if p > pk:
                    pk = p
                pk_n += 1
                if pk_n >= 10:
                    on_chunk(pk)
                    pk = 0.0
                    pk_n = 0
            if check_keys and select.select([sys.stdin], [], [], 0)[0]:
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
    return b"".join(chunks), speech_started, key


def play_raw(raw, volume=0.3):
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


def quick_calibrate():
    global calibrated, _peak_max
    w = CAL_WORDS[0]
    ref_path = ensure_ref(w)
    raw = [None]
    t = threading.Thread(
        target=lambda: raw.__setitem__(0, record_audio(3)[0]))
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
        return
    rec_mfcc = extract_mfcc(rec)
    d_match = dtw_distance(extract_mfcc(ref), rec_mfcc)
    # compare against different words to find mismatch distance
    diffs = []
    for other in CAL_WORDS:
        if other == w:
            continue
        op = ensure_ref(other)
        o, _ = librosa.load(str(op), sr=SAMPLE_RATE)
        o = normalize_volume(o)
        o, _ = librosa.effects.trim(o, top_db=cal["top_db"])
        if len(o) >= 1600:
            diffs.append(dtw_distance(extract_mfcc(o), rec_mfcc))
    cal["bias"] = d_match * 0.9
    if diffs:
        d_mis = np.median(diffs)
        cal["scale"] = (d_mis - cal["bias"]) / 3
    else:
        cal["scale"] = d_match * 0.6
    p = float(np.max(np.abs(rec)))
    if p > _peak_max:
        _peak_max = p
    calibrated = True


def _ref_raw(word):
    ref = ensure_ref(word)
    seg = AudioSegment.from_wav(str(ref))
    return seg.set_channels(1).set_frame_rate(
        SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH).raw_data


def test_rec():
    print("Recording test - speak into mic, Ctrl+C to stop\n")
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
                print(_BARS[min(8, int(peak * 40))], end="", flush=True)
                peak = 0.0
                count = 0
    except KeyboardInterrupt:
        pass
    finally:
        pa.close()
    print()
