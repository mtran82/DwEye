"""Input controller: state + accept/bypass logic, UI-agnostic.

Kept separate from the prompt_toolkit layer so the accept paths can be
tested headlessly and later driven by the eye-tracker instead of keys.

State model:
  words       display words (may carry sentence punctuation, e.g. "plesnt.")
  lm_context  sentence-scoped word list for the pipeline (punctuation-free;
              reset when a sentence ends), index-aligned with words
  fragment    in-progress word
"""
import time

import stats


class Controller:
    def __init__(self, pipeline, on_change=None):
        self.pipe = pipeline
        self.on_change = on_change or (lambda: None)
        self.words = []          # display words
        self.lm_context = []     # pipeline context (reset at sentence end)
        self.fragment = ""       # in-progress word
        self.suggestions = []    # suggestion dicts from combiner
        self.selected = 0
        self.status = ""
        self.timings = (0.0, 0.0)

    # ---- typing ----
    def on_char(self, ch):
        self.fragment += ch
        t0 = time.perf_counter()
        self.refresh()
        self.status = f"typed '{ch}' in {(time.perf_counter()-t0)*1000:.1f}ms"

    def on_backspace(self):
        """Delete one character; crossing a word boundary re-opens the last
        committed word as the fragment. Crossing a sentence-end boundary
        ("word.") re-opens the word without its period; the LM context was
        already reset at the sentence end, so it is left alone."""
        if self.fragment:
            self.fragment = self.fragment[:-1]
        elif self.words:
            last = self.words.pop()
            if len(self.lm_context) == len(self.words) + 1:
                self.lm_context.pop()   # word was LM-context-attached
            self.fragment = last[:-1] if last.endswith(".") else last
        self.refresh()

    def on_period(self):
        """End the current sentence: commit whatever is typed, attach '.',
        reset the LM context so the next word starts fresh."""
        if self.fragment:
            self.words.append(self.fragment)
        elif not self.words:
            return
        self.words[-1] = self.words[-1] + "."
        self.fragment = ""
        self.suggestions = []
        self.lm_context = []
        self.pipe.on_commit(self.lm_context)
        self.status = "sentence ended"
        self.refresh()

    def move_selection(self, delta):
        if self.suggestions:
            self.selected = (self.selected + delta) % len(self.suggestions)

    # ---- suggestions ----
    def refresh(self):
        scored, timings, _gen = self.pipe.suggest(self.lm_context,
                                                  self.fragment)
        self.timings = timings
        self.suggestions = scored[:5]
        self.selected = 0
        self.on_change()

    # ---- commit actions ----
    def accept(self, finish=False):
        """Accept selected suggestion (or verbatim fragment if no list)."""
        if self.suggestions:
            sug = self.suggestions[self.selected]
            if sug["compound"]:
                # e.g. "an apple": replace the trailing "a" and add the word
                parts = sug["compound"].split()
                if self.words and self.words[-1] == "a":
                    self.words = self.words[:-1] + parts
                    self.lm_context = self.lm_context[:-1] + parts
                else:
                    self.words.extend(parts)
                    self.lm_context.extend(parts)
                stats.log_event(self.lm_context[:-len(parts)], self.fragment,
                                sug["compound"], self.suggestions)
            else:
                stats.log_event(self.lm_context, self.fragment, sug["word"],
                                self.suggestions)
                self.words.append(sug["word"])
                self.lm_context.append(sug["word"])
        else:
            if self.fragment:
                stats.log_event(self.lm_context, self.fragment,
                                self.fragment, [])
                self.words.append(self.fragment)
                self.lm_context.append(self.fragment)
        self.fragment = ""
        self.suggestions = []
        self.pipe.on_commit(self.lm_context)
        self.on_change()
        if finish:
            return " ".join(self.words)
        return None

    def bypass(self):
        """Right-arrow: insert typed fragment verbatim + space."""
        if self.fragment:
            stats.log_event(self.lm_context, self.fragment, self.fragment,
                            self.suggestions)
            self.words.append(self.fragment)
            self.lm_context.append(self.fragment)
        self.fragment = ""
        self.suggestions = []
        self.pipe.on_commit(self.lm_context)
        self.on_change()

    def get_text(self):
        shown = " ".join(self.words)
        return (shown + " " if shown else "") + self.fragment