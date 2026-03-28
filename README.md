# English pronunciation trainer

CLI tools for practicing English pronunciation and estimating
vocabulary size.

## pronounce.py

Practice words grouped by phoneme. The app plays each word via
gTTS, records your attempt, then scores it using MFCC/DTW audio
similarity and Google Speech-to-Text.

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
./pronounce.py --vu         # test mic levels
./pronounce.py -d           # show debug info (duration, peak)
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
- STT score - did Google Speech-to-Text hear the right word
- Audio similarity - MFCC/DTW comparison with gTTS reference,
  broken down into start/middle/end segments (requires
  `--calibrate`)

The bar shows the best of the two. 80%+ is a pass.
Audio similarity is hidden without calibration to avoid
false positives.

### Phoneme groups

22 groups, 220 words covering:
- th voiced/voiceless, sh/ch, r/l, v/w, s/z, f/v
- vowel pairs, diphthongs, schwa, eɪ/aɪ
- silent letters, word stress, -ed endings, -tion/-sion
- consonant clusters, ng sound, j/y, ough variations
- vowel+r combinations, zh sound, h sound

Groups and words are defined in `words.yaml`.

### AI feedback

Press `f` to get pronunciation feedback from Google Gemini
(gemini-flash-latest). Evaluates against standard American
accent. Requires `GEMINI_API_KEY` environment variable.

Use `--test-feedback` to verify Gemini accuracy on gTTS
reference audio. Tests correct pronunciation (should say
"Good") and mismatched words (should detect errors).

### Calibration

Run `--calibrate` to measure your mic/speaker channel. This
plays reference words through the speaker, records them, and
computes bias/scale for more accurate audio similarity scoring.
Settings are saved to `~/.english-pronounce/calibration.yaml`.

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
