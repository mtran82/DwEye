"""Async pipeline: cheap path per keystroke, neural LM rerank on pause.

Cheap path (every keystroke, synchronous): SymSpell + n-gram combiner (~2ms).
Expensive path: neural LM rerank of the top LM_TOP candidates. Two modes:
  - use_lm=True in suggest(): synchronous rerank (eval harness measures
    full-quality suggestions; user "waits" for the model).
  - TUI mode: suggest() returns the cheap list immediately; the background
    worker reranks after PAUSE_RERANK_S and stores (gen, suggestions) plus
    fires an on_result callback; stale generations are ignored.

Results carry a generation counter; the caller drops stale ones.
"""
import threading
import time

import combiner
import config
import lm as lmmod
from candidates import CandidateGen
from scorers import NGramScorer


class Pipeline:
    def __init__(self, enable_lm=True):
        self.cand = CandidateGen()
        self.ngram = NGramScorer()
        if not self.ngram.load_cache():
            raise FileNotFoundError("ngram cache missing - run scripts/setup_data.py")
        self.neural = lmmod.NeuralLMScorer(config.LM_MODEL)
        self.enable_lm = enable_lm and self.neural.available

        self.on_result = None            # callback(gen, suggestions) for TUI
        self._gen = 0
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pending = None             # (gen, ctx, frag, cheap_suggestions)
        self._expensive = None           # (gen, suggestions)
        self._expensive_result = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ---- cheap synchronous path ----
    def suggest(self, context_words, fragment, use_lm=False):
        t0 = time.perf_counter()
        cands = self.cand.lookup(fragment)
        t1 = time.perf_counter()
        scored = combiner.combine(context_words, fragment, cands,
                                  self.ngram, self.cand.freq)
        t2 = time.perf_counter()

        if use_lm and scored and self.neural.available:
            t2b = time.perf_counter()
            scored = self._rerank(context_words, fragment, scored)
            t2 = time.perf_counter()
        else:
            t2 = time.perf_counter()

        with self._lock:
            self._gen += 1
            gen = self._gen
            if not use_lm:
                self._pause_event.set()
                self._pending = (gen, list(context_words), fragment, scored)
            self._expensive_result = None
        return scored, ((t1 - t0) * 1000, (t2 - t1) * 1000), gen

    def _rerank(self, context_words, fragment, cheap):
        """Neural rerank of the top LM_TOP cheap suggestions."""
        top = cheap[: config.LM_TOP]
        if not top:
            return cheap
        ctx_text = " ".join(context_words[-20:])
        words = [s["word"] for s in top]
        lm_a = self.neural.score_batch(ctx_text, words)
        lm_an = None
        func_scores = None
        fixable = (context_words and context_words[-1] == "a"
                   and not fragment.strip().lower().startswith("an"))
        if fixable:
            vowels = [s["word"] for s in top
                      if s["word"][:1] in "aeiou" and s["word"] != "a"]
            if vowels:
                an_ctx = " ".join(context_words[:-1][-20:] + ["an"])
                lm_an = dict(zip(vowels,
                                 self.neural.score_batch(an_ctx, vowels)))
            base_ctx = " ".join(context_words[:-1][-20:])
            fa, fan = self.neural.score_batch(base_ctx, ["a", "an"])
            func_scores = {"a": fa, "an": fan}
        res = lmmod.rerank(context_words, fragment, top, lm_a, lm_an,
                           func_scores, self.cand.freq)
        return res + cheap[config.LM_TOP:]

    # ---- commit hook ----
    def on_commit(self, context_words):
        with self._lock:
            self._pause_event.set()

    # ---- background worker (async LM rerank for the TUI) ----
    def _worker_loop(self):
        while True:
            self._pause_event.wait()
            self._pause_event.clear()
            time.sleep(config.PAUSE_RERANK_S)  # debounce window
            if self._pause_event.is_set():
                continue  # newer event arrived; restart debounce
            with self._lock:
                gen, ctx, frag, cheap = getattr(self, "_pending",
                                                (None, [], "", []))
                if gen != self._gen or self._pause_event.is_set():
                    continue
            try:
                reranked = self._rerank(ctx, frag, cheap)
            except Exception as e:  # model load failure etc: keep cheap list
                print(f"[lm] rerank failed: {e}")
                reranked = cheap
            with self._lock:
                if gen == self._gen:   # nothing newer happened
                    self._expensive_result = (gen, reranked)
            if self.on_result and gen == self._gen:
                self.on_result(gen, reranked)

    def poll_expensive(self, gen):
        """Returns reranked suggestions if a fresh result exists for `gen`."""
        with self._lock:
            if self._expensive_result and self._expensive_result[0] == gen:
                res = self._expensive_result[1]
                self._expensive_result = None
                return res
        return None