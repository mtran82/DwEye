"""Candidate generation: SymSpell fuzzy matching + true prefix completion.

SymSpell only returns words within MAX_EDIT_DISTANCE, so "ap" can never
produce "apple" (distance 3). We additionally scan the frequency dictionary
for words starting with the fragment (bisect prefix range) and merge both
sources before ranking.
"""
import bisect
import os

from symspellpy import SymSpell, Verbosity

import config


class CandidateGen:
    def __init__(self):
        self.ss = SymSpell(max_dictionary_edit_distance=config.MAX_EDIT_DISTANCE,
                           prefix_length=7)
        if not os.path.exists(config.FREQ_DICT):
            raise FileNotFoundError(
                f"{config.FREQ_DICT} missing - run scripts/setup_data.py first")
        # keep raw entries for prefix scans
        self.words = []      # sorted list of (word, count)
        with open(config.FREQ_DICT, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    self.words.append((parts[0], int(parts[1])))
        self.words.sort()
        self._sorted_terms = [w for w, _ in self.words]
        self.freq = {w: c for w, c in self.words}

        self.ss.load_dictionary(config.FREQ_DICT, term_index=0, count_index=1)

    def _prefix_candidates(self, frag):
        """Words starting with `frag`, top-N by frequency, dist 0."""
        lo = bisect.bisect_left(self._sorted_terms, frag)
        out = []
        for i in range(lo, len(self._sorted_terms)):
            w = self._sorted_terms[i]
            if not w.startswith(frag):
                break
            out.append((w, 0, self.words[i][1]))
        out.sort(key=lambda t: -t[2])
        return out[: config.PREFIX_CANDIDATES]

    def lookup(self, fragment: str):
        """fragment -> list[(word, edit_distance)] best-effort, deduped.

        Prefix matches carry dist 0 (they contain everything typed).
        """
        frag = fragment.strip().lower()
        if not frag:
            return []
        results = self.ss.lookup(frag, verbosity=Verbosity.ALL,
                                 max_edit_distance=config.MAX_EDIT_DISTANCE,
                                 transfer_casing=False)
        out = []
        for item in results:
            term = getattr(item, "term", None)
            dist = getattr(item, "distance", None)
            if term is None:  # tuple fallback for older versions
                term, dist = item[0], item[1]
            out.append((term, dist))
        out.extend((w, d) for w, d, _ in self._prefix_candidates(frag))

        # dedupe, keeping the minimum edit distance per word
        best = {}
        for term, dist in out:
            if term not in best or dist < best[term]:
                best[term] = dist
        merged = list(best.items())
        merged.sort(key=lambda t: (t[1], -len(t[0])))
        return merged[: config.MAX_CANDIDATES]