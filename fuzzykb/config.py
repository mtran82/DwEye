"""Central tunables for the fuzzy-keyboard pipeline.

All scoring happens in log space:
    log_score(C) = w_lm * log P(C | context) + w_err * log P(input | C) + b_valid
"""

# ---- noisy-channel weights -------------------------------------------------
W_LM = 0.75           # weight of context LM probability
W_ERR = 2.0           # weight of error model (edit distance) term
W_FREQ = 0.25         # weight of log10(word frequency) prior
W_PREFIX = 1.2        # weight of prefix-match fraction (len(frag)/len(cand))
B_VALID = 2.0         # bias added when candidate == the typed fragment verbatim
                      # (prevents "correcting" an already-valid word away)
# weights above tuned by scripts/grid_search.py (40 random configs, seed 42)

# ---- error model -----------------------------------------------------------
ERR_DISCOUNT = 0.1    # multiplicative penalty per edit operation (ln: -2.3)

# ---- candidate generation --------------------------------------------------
MAX_EDIT_DISTANCE = 2
MAX_CANDIDATES = 100  # cap on symSpell candidates before scoring
PREFIX_CANDIDATES = 25  # top-frequency words starting with the typed prefix

# ---- n-gram scorer ---------------------------------------------------------
NGRAM_ORDER = 3       # trigram with bigram/unigram backoff
LM_LAMBDA = 0.75      # mix weight: ngram logprob vs uniform-over-vocab fallback

# ---- neural LM reranker (phase 2) ------------------------------------------
LM_MODEL = "Qwen/Qwen2.5-1.5B"
W_NLM = 1.0            # weight of neural LM logprob in the rerank blend
LM_TOP = 20            # how many cheap-path candidates the LM rescores

# ---- async expensive path --------------------------------------------------
PAUSE_RERANK_S = 0.35 # idle time on a fragment before the "expensive" pass runs

# ---- paths -----------------------------------------------------------------
DATA_DIR = "data"
FREQ_DICT = "data/frequency_dictionary_en_82_765.txt"
CORPUS_FILE = "data/corpus.txt"
NGRAM_CACHE = "data/ngrams.pkl"
SESSION_LOG = "data/session_log.jsonl"