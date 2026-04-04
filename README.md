# English pronunciation trainer

CLI tools for practicing English pronunciation and estimating
vocabulary size.

## pronounce.py

Practice words grouped by phoneme. The app plays each word via
gTTS, records your attempt, then scores it using audio
similarity and Google speech recognition. Press `f` anytime for
AI feedback and mentoring from Google Gemini.

### Usage

```sh
./pronounce.py              # interactive group selection
./pronounce.py -g th_voiced # practice specific group
./pronounce.py -w           # auto-select weakest group
./pronounce.py -c           # continuous mode (no Enter between words)
./pronounce.py -cw          # continuous + weakest group
./pronounce.py -t "ship"    # practice a specific word
./pronounce.py --list       # list all phoneme groups
./pronounce.py --stats      # show performance stats
./pronounce.py --calibrate  # calibrate mic/speaker
./pronounce.py --test-rec   # test recording histogram
./pronounce.py -d           # debug info (duration, peak, selfcheck)
./pronounce.py --sim-threshold 50  # set audio similarity pass threshold
./pronounce.py --test-feedback       # test Gemini accuracy (both)
./pronounce.py --test-feedback good  # test on etalon audio
./pronounce.py --test-feedback bad   # test on wrong words
```

### Keys

During recording (continuous mode):
- `s` - skip word
- `f` - get AI feedback (Gemini)
- `q` - quit

Between attempts:
- `Enter` - retry
- `s` - skip
- `f` - feedback
- `p` - pause (continuous mode)

### Scoring

Each attempt shows two scores:
- Speech recognition - did Google hear the right word
- Audio similarity - comparison with gTTS reference
  (requires `--calibrate`)

Pass conditions (with calibration):
- sim >= threshold (default 60%), or
- sim >= half threshold and perfect STT, or
- pct >= 80% when sim is disabled

Audio similarity is colored by `--sim-threshold` (default 60%):
green above, yellow above half, red below.
Without calibration, only STT is used (80%+ to pass).

### Phoneme groups

22 groups, 220 words covering:
- th voiced/voiceless, sh/ch, r/l, v/w, s/z, f/v
- vowel pairs, diphthongs, schwa, eɪ/aɪ
- silent letters, word stress, -ed endings, -tion/-sion
- consonant clusters, ng sound, j/y, ough variations
- vowel+r combinations, zh sound, h sound

Groups and words are defined in `words.yaml`.

### AI feedback and mentoring

Press `f` during practice to get pronunciation feedback from
Google Gemini. The AI coach listens to your recording,
evaluates it against standard American accent, and tells you
what to fix. If your pronunciation is good, it just says
"Good". Feedback is spoken aloud via gTTS so you can keep
practicing hands-free. Requires `GEMINI_API_KEY`.

Use `--test-feedback` to verify Gemini accuracy on gTTS
reference audio. Tests correct pronunciation (should say
"Good") and mismatched words (should detect errors).

### Calibration

Audio similarity requires calibration. Calibration requires
speakers - it plays reference words aloud and records them
through the mic to measure channel characteristics.

On first run, the app does a quick one-word auto-calibration.
Run `--calibrate` for more precise tuning with multiple words.
Settings are saved to `~/.english-pronounce/calibration.yaml`.
With headphones only (no speakers), audio similarity is
unavailable and scoring falls back to speech recognition.

### Debug mode

`-d` shows duration, peak amplitude, and loopback selfcheck.
Selfcheck plays the reference word through the speaker while
recording via mic, then computes audio_similarity on the
recording to verify the scoring pipeline.

## vocab.py

Estimate your English vocabulary size using binary search over
a frequency-sorted word list (~35k words from wordfreq,
lemmatized to base forms, filtered to words with WordNet
definitions).

### Usage

```sh
./vocab.py
```

The app shows words one at a time. Press right arrow or `y` if
you know the word, left arrow or `n` if you don't. Unknown
words show their WordNet definition and synonyms. After 14
rounds (or when the range narrows to <500 words), it prints
your estimated vocabulary size.

## Requirements

- Python 3
- PulseAudio (for recording/playback via pasimple)
- Internet connection (Google STT, gTTS, Gemini)

```sh
pip install -r requirements.txt
```

For vocab.py, also install:

```sh
pip install wordfreq nltk
python3 -c "import nltk; nltk.download('wordnet')"
```

## Files

- `pronounce.py` - pronunciation trainer
- `vocab.py` - vocabulary size estimator
- `words.yaml` - phoneme groups, words, STT equivalences
- `requirements.txt` - Python dependencies
- `~/.english-pronounce/` - user data directory
  - `history.yaml` - practice history
  - `calibration.yaml` - mic/speaker calibration
  - `ref/` - cached gTTS reference audio
  - `vocab_cache.yaml` - cached lemmatized word list
