"""Session logging: (context, fragment, chosen word, top alternatives) -> JSONL.

This is the raw data for the future confusion-matrix / gaze-error model fit.
"""
import json
import os
import time

import config


def log_event(context_words, fragment, chosen, alternatives):
    """chosen: str or None (bypass/verbatim). alternatives: suggestion dicts."""
    os.makedirs(os.path.dirname(config.SESSION_LOG), exist_ok=True)
    alts = [{"word": a["word"], "prob": round(a["prob"], 4),
             "compound": a.get("compound")} for a in alternatives[:5]]
    rec = {
        "ts": time.time(),
        "context": context_words[-10:],
        "fragment": fragment,
        "chosen": chosen,
        "alternatives": alts,
    }
    with open(config.SESSION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")