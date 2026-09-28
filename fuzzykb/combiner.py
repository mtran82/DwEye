"""Noisy-channel combiner: log-linear merge of LM and error-model scores.

Also emits compound suggestions for the function-word case: when the context
ends in "a" and a candidate starts with a vowel, "an <word>" is scored with
the corrected context and offered as a single suggestion (e.g. "a apl" ->
"an apple"). Accepting it fixes both words.
"""
import math

import config

VOWELS = "aeiou"


def error_logprob(edit_distance: int) -> float:
    """log P(observed_input | candidate) under a simple edit-count model."""
    return edit_distance * math.log(config.ERR_DISCOUNT)


def combine(context_words, fragment, candidates, scorer, candidate_freq=None):
    """candidates: list[(word, edit_dist)] -> ranked list of dicts:
    {word, prob, lm, err, total, compound} where compound is None or the
    display string like "an apple" (accepted value replaces two words).

    candidate_freq: optional {word: corpus_count} for the frequency prior.
    """
    frag = fragment.strip().lower()
    candidate_freq = candidate_freq or {}
    fixable = (context_words and context_words[-1] == "a"
                   and not frag.startswith("an"))
    ctx_an = (list(context_words[:-1]) + ["an"]) if fixable else None

    scored = []
    for word, dist in candidates:
        lm = scorer.score(context_words, word)
        err = error_logprob(dist)
        total = config.W_LM * lm + config.W_ERR * err

        # frequency prior: common words are a priori more likely intended
        freq = candidate_freq.get(word)
        if freq:
            total += config.W_FREQ * math.log10(freq + 1)

        # prefix prior: candidates that contain everything typed so far are
        # progressively preferred as the fragment grows ("ap" -> "apple"
        # beats "at" once enough chars match)
        if dist == 0 and word.startswith(frag) and len(word) > 0:
            total += config.W_PREFIX * (len(frag) / len(word))

        compound = None
        if fixable and word[0] in VOWELS and word != "a":
            lm_an = scorer.score(ctx_an, word)
            # choose whichever context fits better; if the "an" reading wins,
            # the suggestion becomes the two-word compound
            if config.W_LM * lm_an + config.W_ERR * err > total:
                total = config.W_LM * lm_an + config.W_ERR * err
                compound = "an " + word
        if word == frag:
            total += config.B_VALID
        scored.append((word, dist, lm, total, compound))

    if not scored:
        return []

    mx = max(s[3] for s in scored)
    exps = [math.exp(s[3] - mx) for s in scored]
    z = sum(exps)
    out = []
    for (word, dist, lm, total, compound), e in zip(scored, exps):
        out.append({"word": word, "prob": e / z, "lm": lm,
                    "err": error_logprob(dist), "total": total,
                    "compound": compound})

    out.sort(key=lambda t: -t["total"])
    return out