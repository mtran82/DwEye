# fuzzykb — Phase 0 prototype

Context-aware fuzzy autocomplete TUI for eye-typing input (noisy-channel model).

Run (from `fuzzykb/`, using the project venv):

    ../venv/Scripts/python.exe tui.py

Keys: letters/`'` type · **space** accept suggestion · **right** bypass (verbatim + space) ·
**up/down** select · **1-5** direct pick · **enter** accept + finish · **.** end sentence
(commits any typed fragment verbatim, appends '.', resets LM context) ·
**backspace** delete char; crossing a word boundary re-opens that word for editing ·
**esc / ctrl-c** quit.

Rebuild data (only needed once): `../venv/Scripts/python.exe scripts/setup_data.py && ../venv/Scripts/python.exe scripts/train_ngrams.py`

Architecture (each module independently swappable):

    tui.py         prompt_toolkit rendering + key bindings only
    controller.py  input state + accept/bypass logic (UI-agnostic, headless-testable;
                   future eye-tracker driver calls the same methods)
    pipeline.py    orchestration: cheap path per keystroke; background worker for
                   the expensive LM pass (generation counter, phase-2 hook)
    candidates.py  SymSpell fuzzy (edit dist<=2) + true prefix completion merged
    scorers.py     NGramScorer (trigram, stupid backoff, pure python); LMScorer stub
    combiner.py    noisy-channel: W_LM*log P(word|ctx) + W_ERR*log P(input|word)
                   + B_VALID bias; emits compound suggestions ("a apl" -> "an apple")
    config.py      all tunable weights
    stats.py       session JSONL log -> future gaze confusion-matrix training data

Known limitations (expected, phase-1 targets):
- Corpus is 3 Gutenberg novels (500k tokens): modern/rare words rank poorly
  ("appl" prefers "application" over "apple" in Austen-world).
- "a" -> "an" only fires when the an-reading beats the a-reading inside the
  trigram LM; the neural LM reranker (phase 2) is the real fix.
- Garbage input ("xyq") still matches something at edit distance 2; needs a
  channel-prior floor (phase 1).
- SHELVED (evaluation honesty): the n-gram is trained on all of corpus.txt,
  so the held-out slice used by scripts/eval.py shares training data with
  the model and its scores are optimistic. Retrain on a train-only split
  before trusting absolute Gutenberg numbers.

## Evaluation harness

    ../venv/Scripts/python.exe scripts/eval.py --words 3000 --seed 42

Simulated typing over two target sources (modern quotes via dwyl/quotes +
a held-out slice of the Gutenberg corpus), with optional synthetic gaze-noise
corruption (0/10/25%). Accept policy: take the first moment the true word
appears in the top-3; measures top-1/2/3 reach rates, keystroke savings,
latency, and blind-top-1 wrong-accept risk. Per-word results land in
data/eval_results.jsonl.

Baseline (n=3000/condition, seed 42, with frequency + prefix priors):
quotes 95% top-1 reach / 99% top-3, 1.9 keys/word saved (top-3 user),
blind-top-1 wrong-accept 4.5%; Gutenberg-heldout 98% top-1 / 99% top-3,
2.4 keys/word. Latency ~1.4ms mean. Pre-prior baseline: quotes top-1
92.0%, blind-wrong 7.3%; the priors moved ranking, not reach.
