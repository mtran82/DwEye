"""Fetch/prepare data: symspell frequency dictionary + n-gram training corpus.

Frequency dictionary: official symspell repo (raw GitHub).
Corpus: public-domain Gutenberg texts.
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

FREQ_URL = ("https://raw.githubusercontent.com/mammothb/symspellpy/master/"
            "symspellpy/frequency_dictionary_en_82_765.txt")

# bundled with the symspellpy wheel - preferred source
WHEEL_FREQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "venv", "Lib", "site-packages", "symspellpy",
                          "frequency_dictionary_en_82_765.txt")

CORPUS_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",   # Pride and Prejudice
    "https://www.gutenberg.org/files/1400/1400-0.txt",   # Great Expectations
    "https://www.gutenberg.org/files/2852/2852-0.txt",   # Hound of the Baskervilles
]


def fetch(url, dest):
    print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def extract_body(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start, end = 0, len(lines)
    for i, ln in enumerate(lines[:500]):
        if "*** START OF" in ln.upper():
            start = i + 1
            break
    for i in range(max(0, len(lines) - 500), len(lines)):
        if "*** END OF" in lines[i].upper():
            end = i
            break
    body = "".join(lines[start:end])
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.startswith("[") and "Transcriber" not in ln)


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not os.path.exists(config.FREQ_DICT):
        if os.path.exists(WHEEL_FREQ):
            import shutil
            shutil.copy(WHEEL_FREQ, config.FREQ_DICT)
            print("copied frequency dictionary from symspellpy wheel")
        else:
            fetch(FREQ_URL, config.FREQ_DICT)
    else:
        print("frequency dictionary already present")

    if os.path.exists(config.CORPUS_FILE):
        os.remove(config.CORPUS_FILE)
    for url in CORPUS_URLS:
        part = os.path.join(config.DATA_DIR, os.path.basename(url))
        if not os.path.exists(part):
            fetch(url, part)
        body = extract_body(part)
        print(f"  {part}: {len(body)} chars")
        with open(config.CORPUS_FILE, "a", encoding="utf-8") as out:
            out.write(body + "\n")
    print("done")


if __name__ == "__main__":
    main()