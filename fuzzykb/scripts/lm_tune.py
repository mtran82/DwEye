"""Tune W_NLM / B_VALID for the neural LM reranker.

Sweeps a small grid over a 300-word quote stream with 25% corruption
(the hard condition) and reports strict first-appearance top-1 plus keys
saved, so the winner can be applied to config.py.

Usage: python scripts/lm_tune.py [--words 300]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from eval import load_quotes, corrupt, TOP_N  # noqa: E402


def evaluate(pipe, sents, rate):
    rng = random.Random(42)
    top1 = 0
    saved = 0
    n = 0
    for words in sents:
        ctx = []
        for true in words:
            typed = corrupt(true, rng) if rng.random() < rate else true
            saved_i = 0
            for i in range(1, len(typed) + 1):
                s, _, _ = pipe.suggest(ctx, typed[:i], use_lm=True)
                tw = [(x["compound"].split()[-1] if x["compound"] else x["word"])
                      for x in s[:TOP_N]]
                if true in tw:
                    top1 += 1 if tw.index(true) == 0 else 0
                    saved_i = len(true) - i
                    break
            saved += saved_i
            n += 1
            ctx.append(true)
    return top1 / n, saved / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=300)
    args = ap.parse_args()

    pipe = Pipeline(enable_lm=True)
    texts = load_quotes()
    random.Random(7).shuffle(texts)
    stream = []
    for t in texts:
        stream.extend(t)
        if len(stream) >= args.words:
            break
    sents = [stream[i:i + 12] for i in range(0, len(stream), 12)]

    results = []
    base = (config.W_NLM, config.B_VALID)
    for wnlm in [0.5, 1.0, 2.0]:
        for bval in [2.0, 3.0, 4.0]:
            config.W_NLM, config.B_VALID = wnlm, bval
            t1, saved = evaluate(pipe, sents, rate=0.25)
            results.append((t1, saved, wnlm, bval))
            print(f"W_NLM={wnlm} B_VALID={bval}: top1={t1:.3f} keys={saved:.2f}",
                  flush=True)
    config.W_NLM, config.B_VALID = base
    results.sort(key=lambda r: (-r[0], -r[1]))
    print("\nTOP 3:")
    for r in results[:3]:
        print(f"  top1={r[0]:.3f} keys={r[1]:.2f}  W_NLM={r[2]} B_VALID={r[3]}")


if __name__ == "__main__":
    main()