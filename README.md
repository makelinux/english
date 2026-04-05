# English pronunciation trainer

CLI tools for practicing English pronunciation and estimating
vocabulary size.

## pronounce.py

Practice words grouped by phoneme:
- Speaks each word via gTTS
- Records and replays your attempt so you hear yourself
- Scores using audio similarity and speech recognition
- AI feedback

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
./pronounce.py --test-feedback       # test AI feedback accuracy
```

### Sample output

```
1/10: foot  /fʊt/  ######.............. 30%  heard: food  30%
1/10: foot  /fʊt/  #######............. 39%  heard: food  39%
  You said "food". Try to make the vowel sound shorter and more relaxed, like the 'oo' in "book" or "good".
Also, make sure the final sound is a crisp 't' sound, not a 'd'.
1/10: foot  /fʊt/  #################### 100%  heard: foot  44%

```

### Keys

During recording (continuous mode):
- `Enter` - retry
- `p` - pause (continuous mode)
- `s` - skip word
- `f` - get AI feedback
- `s` - skip
- `q` - quit

### Scoring

Each attempt shows two scores:
- Speech recognition - did Google hear the right word
- Audio similarity - comparison with gTTS reference

Pass conditions:
- sim >= threshold (default 60%), or
- sim >= half threshold and perfect speech recognition, or
- speech recognition >= 80% if audio similarity unavailable

Audio similarity is colored by `--sim-threshold` (default 60%):
green above, yellow above half, red below.

### Phoneme groups

22 groups, 220 words covering:
- th voiced/voiceless, sh/ch, r/l, v/w, s/z, f/v
- vowel pairs, diphthongs, schwa, eɪ/aɪ
- silent letters, word stress, -ed endings, -tion/-sion
- consonant clusters, ng sound, j/y, ough variations
- vowel+r combinations, zh sound, h sound

Groups and words are defined in `words.yaml`.

### AI feedback

Press `f` to get pronunciation feedback. The AI listens to
your recording, compares it to standard American accent, and
tells you what to fix. Good pronunciation gets just "Good".
Feedback is spoken aloud via gTTS for hands-free practice.

Backends:
- Gemini (default) - set `GEMINI_API_KEY` env variable
- OpenAI-compatible (llama-stack, etc) - configure in
  `~/.config/english-pronounce/config.yaml`:

```yaml
openai:
  base_url: http://localhost:8321/v1/
  api_key: none
  api_type: responses  # or completions
  model: gemini/gemini-2.5-flash
  audio_format: image_url  # or input_audio, input_file
```

Use `--test-feedback` to verify accuracy - plays mismatched
words and checks that the AI detects errors.

### Calibration

Audio similarity requires calibration. Calibration requires
speakers - it plays reference words aloud and records them
through the mic to measure channel characteristics.

On first run, the app does a quick one-word auto-calibration.
Run `--calibrate` for more precise tuning with multiple words.
Settings are saved to `~/.config/english-pronounce/calibration.yaml`.
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
- Internet connection (Google speech recognition, gTTS)
- AI feedback: Gemini API key or OpenAI-compatible server

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
- `~/.config/english-pronounce/` - user data directory
  - `history.yaml` - practice history
  - `calibration.yaml` - mic/speaker calibration
  - `ref/` - cached gTTS reference audio
  - `config.yaml` - OpenAI-compatible API settings
  - `vocab_cache.yaml` - cached lemmatized word list
