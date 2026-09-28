"""Scorer interface + n-gram scorer (pure Python) + LM stub.

A scorer implements:
    score(context_words: list[str], candidate: str) -> float
returning a LOG probability-ish value (higher = better context fit).
"""
import math
import os
import pickle
import re
from collections import defaultdict

import config

_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class NGramScorer:
    """Trigram scorer with backoff, trained on a plain-text corpus.

    log P(w | w-2, w-1) via stupid-backoff (Brants & Franz 2009):
      trigram count>0: log c(w-2,w-1,w) - log c(w-2,w-1) - log(0.4^-0)
      backoff levels multiply by 0.4 each step.
    """

    DISCOUNT = 0.4

    def __init__(self):
        self.uni = defaultdict(int)
        self.bi = defaultdict(int)
        self.tri = defaultdict(int)
        self.total = 0

    # ---- training ----
    def train(self, corpus_path: str, save=True):
        with open(corpus_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = tokenize(line)
                self._add_sequence(words)
        if save:
            self._save_cache()

    def _add_sequence(self, words):
        s = ["<s>", "<s>"] + words + ["</s>"]
        for w in s:
            self.uni[w] += 1
        self.total += len(s)
        for a, b in zip(s, s[1:]):
            self.bi[(a, b)] += 1
        for a, b, c in zip(s, s[1:], s[2:]):
            self.tri[(a, b, c)] += 1

    def _save_cache(self):
        os.makedirs(os.path.dirname(config.NGRAM_CACHE), exist_ok=True)
        with open(config.NGRAM_CACHE, "wb") as f:
            pickle.dump(
                {"total": self.total,
                 "uni": dict(self.uni), "bi": dict(self.bi), "tri": dict(self.tri)},
                f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cache(self) -> bool:
        if not os.path.exists(config.NGRAM_CACHE):
            return False
        with open(config.NGRAM_CACHE, "rb") as f:
            d = pickle.load(f)
        self.total = d["total"]
        self.uni = defaultdict(int, d["uni"])
        self.bi = defaultdict(int, d["bi"])
        self.tri = defaultdict(int, d["tri"])
        return True

    # ---- scoring ----
    def score(self, context_words, candidate: str) -> float:
        """Log probability of `candidate` following `context_words`.

        Interpolated (Jelinek-Mercer) trigram/bigram/unigram instead of hard
        stupid-backoff: unseen n-grams now get proportional mass from lower
        orders rather than a flat floor.
        """
        ctx = list(context_words[-2:]) if context_words else []
        if not ctx:
            ctx = ["<s>", "<s>"]
        elif len(ctx) == 1:
            ctx = ["<s>"] + ctx
        w1, w2, w3 = ctx[0], ctx[1], candidate.lower()

        L3, L2, L1 = 0.6, 0.3, 0.1
        c12 = self.bi.get((w1, w2), 0)
        p3 = L3 * (self.tri.get((w1, w2, w3), 0) / c12) if c12 else 0.0
        c2 = self.uni.get(w2, 0)
        p2 = L2 * (self.bi.get((w2, w3), 0) / c2) if c2 else 0.0
        p1 = L1 * (self.uni.get(w3, 0) / self.total) if self.total else 0.0
        p = p3 + p2 + p1
        if p <= 0:
            p = 1.0 / (self.total + len(self.uni))  # unseen-word floor
        return math.log(p)

    def vocab_size(self):
        return len(self.uni)


class LMScorer:
    """Stub for the Phase-2 neural LM. Same interface as NGramScorer.

    Phase 2: load GPT-2-small / Qwen-0.5B, KV-cache the context prefix,
    batch-score the top ~50 candidates in one forward pass.
    """

    def __init__(self, fallback: NGramScorer):
        self.fallback = fallback

    def score(self, context_words, candidate: str) -> float:
        return self.fallback.score(context_words, candidate)