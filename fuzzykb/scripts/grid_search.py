"""Weight grid-search over the noisy-channel combiner.

Loads the pipeline once, then sweeps config weight combinations in-process
(config attributes are read at call time by combiner.combine), evaluating
each on a quote sample (the harder source) with a fixed seed.

Objective: maximize top-1 reach; ties broken by keys saved (t1) then lower
blind wrong-accept risk. Writes the winner to stdout; does NOT modify
config.py (that's a manual apply after review).

Usage: python scripts/grid_search.py [--words 1000] [--trials 40]
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from eval import load_quotes, corrupt, TOP_N  # noqa: E402

GRID = {
    "W_LM": [0.75, 1.0, 1.25],
    "W_ERR": [0.5, 1.0, 2.0],
    "W_FREQ": [0.0, 0.25, 0.5],
    "W_PREFIX": [0.8, 1.2, 1.6],
    "B_VALID": [1.5, 2.0, 2.5],
}


def evaluate(pipe, sentences, rates, rng_seed=42):
    rng = random.Random(rng_seed)
    rows = []
    for words in sentences:
        context = []
        for true in words:
            typed = corrupt(true, rng) if rng.random() < rates else true
            best = None
            for i in range(1, len(typed) + 1):
                sugg, _, _ = pipe.suggest(context, typed[:i])
                top = [(s["compound"].split()[-1] if s["compound"] else s["word"])
                       for s in sugg[:TOP_N]]
                if true in top:
                    best = i
                    break
            chars = best if best is not None else len(typed)
            rows.append({"len": len(true), "chars": chars,
                         "top1": best == 1 if best else False})
            context.append(true)
    n = len(rows)
    top1 = sum(1 for r in rows if r["top1"]) / n
    saved = sum(r["len"] - r["chars"] for r in rows) / n
    return top1, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    pipe = Pipeline()
    texts = load_quotes()
    rng = random.Random(7)
    rng.shuffle(texts)
    stream = []
    for t in texts:
        stream.extend(t)
        if len(stream) >= args.words:
            break
    sents = [stream[i:i + 12] for i in range(0, len(stream), 12)]

    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    random.Random(0).shuffle(combos)
    combos = combos[: args.trials]

    results = []
    base = {k: getattr(config, k) for k in keys}
    for trial, combo in enumerate(combos):
        for k, v in zip(keys, combo):
            setattr(config, k, v)
        t0 = time.time()
        top1, saved = evaluate(pipe, sents, rates=0.0)
        top1c, savedc = evaluate(pipe, sents, rates=0.25)
        score = top1 + top1c  # clean + corrupted top-1 reach
        results.append((score, top1, saved, top1c, savedc, dict(zip(keys, combo))))
        print(f"[{trial+1}/{len(combos)}] score={score:.3f} "
              f"top1={top1:.3f}/{top1c:.3f} keys={saved:.2f}/{savedc:.2f} "
              f"{dict(zip(keys, combo))}", flush=True)

    for k, v in base.items():
        setattr(config, k, v)

    results.sort(key=lambda r: -r[0])
    print("\nTOP 5:")
    for r in results[:5]:
        print(f"score={r[0]:.3f} top1={r[1]:.3f}/{r[3]:.3f} "
              f"keys={r[2]:.2f}/{r[4]:.2f}")
        print("   ", r[5])
    print("\nCURRENT (default) config was:", base)


if __name__ == "__main__":
    main()