"""Evaluation harness: simulated typing over target texts.

Measures autocomplete quality without a human: for each word in a target
text, its prefixes are fed to the pipeline one char at a time (as a typer
would), with optional synthetic corruption standing in for gaze noise.

Accept policy: the simulated user accepts the first time the TRUE word
appears in the top-3 suggestion list; otherwise types the word out fully
and bypasses. Also tracks the "blind top-1 user" risk: the fraction of
acceptance moments where #1 is NOT the true word.

Usage: python scripts/eval.py [--words 3000] [--seed 42]

Output: console report + data/eval_results.jsonl (per-word rows).
NOTE (shelved, per user): the n-gram model is NOT retrained on a train-only
split, so the held-out Gutenberg slice shares training data with the model
and its scores are optimistic. Revisit before trusting absolute numbers.
"""
import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import Pipeline  # noqa: E402
import config  # noqa: E402

TOP_N = 3
RATES = [0.0, 0.10, 0.25]
WORD_RE = re.compile(r"[A-Za-z']+")


def load_quotes(min_words=3):
    quotes = json.load(open("data/quotes_raw.json", encoding="utf-8"))
    return [[w.lower() for w in WORD_RE.findall(q.get("text", ""))]
            for q in quotes
            if len(WORD_RE.findall(q.get("text", ""))) >= min_words]


def load_gutenberg_heldout(frac=0.1):
    """Last `frac` of the corpus as held-out sentences (NOT retrained away)."""
    with open(config.CORPUS_FILE, encoding="utf-8") as f:
        text = f.read()
    text = text[-int(len(text) * frac):]
    texts = []
    for sent in re.split(r"[.!?]+", text):
        words = WORD_RE.findall(sent)
        if len(words) >= 3:
            texts.append([w.lower() for w in words])
    return texts


def corrupt(word, rng):
    """One synthetic gaze-noise edit: transpose / delete / insert / substitute."""
    if len(word) < 3:
        return word
    op = rng.choice(["transpose", "delete", "insert", "substitute"])
    i = rng.randrange(len(word))
    if op == "transpose" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i:]
    return word[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]


class Eval:
    def __init__(self, pipe, rate, rng, use_lm=False):
        self.pipe = pipe
        self.rate = rate
        self.rng = rng
        self.use_lm = use_lm
        self.rows = []

    def run_text(self, words):
        context = []
        for true in words:
            typed = corrupt(true, self.rng) if self.rng.random() < self.rate else true
            self.rows.append({"true": true, "typed": typed,
                              **self._simulate(context, true, typed)})
            context.append(true)

    def _simulate(self, context, true, typed):
        """Type `typed` char by char, recording the rank of `true` in the
        top-3 after each prefix (rank 0 = not in list).

        Policy columns derived from the trajectory:
          top-N user: accepts at the first prefix where true is within top-N;
          if that never happens, types the full word (bypass).
        """
        traj = []            # (prefix_len, rank_in_top3)
        latencies = []
        sugg = []
        for i in range(1, len(typed) + 1):
            t0 = time.perf_counter()
            sugg, _, _ = self.pipe.suggest(context, typed[:i],
                                           use_lm=self.use_lm)
            latencies.append((time.perf_counter() - t0) * 1000)
            top_words = [(s["compound"].split()[-1] if s["compound"] else s["word"])
                         for s in sugg[:TOP_N]]
            rank = top_words.index(true) + 1 if true in top_words else 0
            traj.append((i, rank))

        def policy_chars(n):
            for i, rank in traj:
                if rank and rank <= n:
                    return i, rank
            return len(typed), None

        t1_chars, t1_rank = policy_chars(1)
        t2_chars, t2_rank = policy_chars(2)
        t3_chars, t3_rank = policy_chars(3)

        # blind top-1 risk: at the top-3 acceptance point, was #1 the true word?
        blind_wrong = None
        if t3_rank:
            top1 = ((sugg[0]["compound"].split()[-1] if sugg[0]["compound"]
                     else sugg[0]["word"]) if sugg else None)
            blind_wrong = 0 if top1 == true else 1

        return {
            "traj": traj,
            "accepted_t1": t1_rank is not None,
            "accepted_t2": t2_rank is not None,
            "accepted_t3": t3_rank is not None,
            "chars_t1": t1_chars, "chars_t2": t2_chars, "chars_t3": t3_chars,
            "word_len": len(true),
            "blind_top1_wrong": blind_wrong,
            "latency": sum(latencies) / len(latencies) if latencies else 0.0,
        }


def summarize(rows, label):
    n = len(rows)
    r1 = sum(1 for r in rows if r["accepted_t1"]) / n
    r2 = sum(1 for r in rows if r["accepted_t2"]) / n
    r3 = sum(1 for r in rows if r["accepted_t3"]) / n

    def keys(col):
        vals = [r["word_len"] - r[f"chars_{col}"] for r in rows]
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        med = vals_sorted[len(vals) // 2]
        return f"{mean:5.2f} (med {med})"

    lat = [r["latency"] for r in rows]
    blind = [r["blind_top1_wrong"] for r in rows if r["blind_top1_wrong"] is not None]
    lines = [
        f"--- {label} (n={n}) ---",
        f"reach:   top-1 {r1*100:5.1f}%   top-2 {r2*100:5.1f}%   top-3 {r3*100:5.1f}%",
        f"keys saved/word (avg, median):  "
        f"t1 {keys('t1')}   t2 {keys('t2')}   t3 {keys('t3')}",
        f"mean latency: {sum(lat)/len(lat):.2f}ms  max: {max(lat):.1f}ms",
        f"blind-top1-user wrong accepts: {sum(blind)/len(blind)*100:.1f}%",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lm", action="store_true",
                    help="neural LM rerank at every keystroke (slow)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pipe = Pipeline(enable_lm=True)

    sources = {"quotes": load_quotes(),
               "gutenberg_heldout": load_gutenberg_heldout()}

    out_path = os.path.join(config.DATA_DIR, "eval_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    for src_name, texts in sources.items():
        rng.shuffle(texts)
        stream = []
        for t in texts:
            stream.extend(t)
            if len(stream) >= args.words:
                break
        stream = stream[: args.words]
        sents = [stream[i:i + 12] for i in range(0, len(stream), 12)]
        for rate in RATES:
            ev = Eval(pipe, rate, rng, use_lm=args.lm)
            for s in sents:
                ev.run_text(s)
            rep = summarize(ev.rows,
                            f"{src_name}  corruption={rate*100:.0f}%")
            with open(out_path, "a", encoding="utf-8") as f:
                for r in ev.rows:
                    f.write(json.dumps({"source": src_name,
                                        "rate": rate, **r}) + "\n")
            print(rep)
            print()

    print("=" * 60)
    print("full per-word results ->", out_path)


if __name__ == "__main__":
    main()