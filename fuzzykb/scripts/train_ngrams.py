"""Train the n-gram cache: Gutenberg corpus + a slice of text8 (Wikipedia).

text8 adds modern encyclopedic text to fix the style mismatch ("appl" ->
"application" over "apple"). Token slice is limited to keep the pure-Python
n-gram dicts within RAM (~6M tokens).

Run after setup_data.py, which fetches text8 as data/text8.zip.
"""
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorers import NGramScorer  # noqa: E402
import config  # noqa: E402

TEXT8 = os.path.join(config.DATA_DIR, "text8.zip")
TEXT8_TOKENS = 6_000_000  # slice limit


def main():
    s = NGramScorer()
    t0 = time.time()

    if os.path.exists(config.CORPUS_FILE):
        s.train(config.CORPUS_FILE, save=False)
        print(f"gutenberg: {s.total} tokens so far")

    if os.path.exists(TEXT8):
        with zipfile.ZipFile(TEXT8) as z:
            raw = z.read("text8").decode("ascii")
        words = raw.split()[:TEXT8_TOKENS]
        CH = 20000
        for i in range(0, len(words), CH):
            s._add_sequence(words[i:i + CH])
        print(f"text8: +{min(TEXT8_TOKENS, len(words))} tokens")
    else:
        print("text8.zip missing - Gutenberg-only model")

    s._save_cache()
    print(f"trained in {time.time()-t0:.1f}s: {s.total} tokens, "
          f"{s.vocab_size()} types -> {config.NGRAM_CACHE}")


if __name__ == "__main__":
    main()