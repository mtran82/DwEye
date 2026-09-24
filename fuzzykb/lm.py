"""Neural LM reranker (phase 2): Qwen2.5-0.5B on GPU.

score_candidates re-ranks a cheap-path suggestion list using neural LM
logprob + the same err/freq/prefix/bias terms the n-gram combiner uses.
The model loads lazily on first use (keeps TUI boot instant); the pipeline
worker thread is the only caller, so a lock guards the model.
"""
import math
import threading

import config

try:
    import os
    # model is already cached locally; skip hub checks + hide load progress
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NeuralLMScorer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._tok = None
        self._model = None
        self.available = HAS_TORCH

    def load(self):
        with self._lock:
            if self._model is None and self.available:
                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, dtype=torch.float16)
                if torch.cuda.is_available():
                    model = model.cuda()
                self._model = model.eval()

    def score_batch(self, context_text, cand_words):
        """LM logprob of each candidate word following context_text."""
        self.load()
        with self._lock:
            tok, model = self._tok, self._model
            if not context_text.strip():
                context_text = " "  # sentence start: empty seq breaks forward
            ctx_ids = tok(context_text, return_tensors="pt").input_ids
            seqs = [tok(" " + w, add_special_tokens=False).input_ids
                    for w in cand_words]
            maxlen = max(len(s) for s in seqs)
            pad = tok.pad_token_id or tok.eos_token_id
            batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                batch[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            dev = next(model.parameters()).device
            with torch.no_grad():
                n = len(seqs)
                # context pass at batch size n so candidate KV shapes match
                ctx_batch = ctx_ids.expand(n, -1)
                ctx_out = model(ctx_batch.to(dev), use_cache=True)
                past = ctx_out.past_key_values
                ctx_logprobs = torch.log_softmax(
                    ctx_out.logits[:, -1].float(), dim=-1)
                logits = model(batch.to(dev), attention_mask=mask.to(dev),
                               past_key_values=past).logits
                logprobs = torch.log_softmax(logits.float(), dim=-1)
            out = []
            for i, s in enumerate(seqs):
                lp = ctx_logprobs[i, s[0]].item()
                for j in range(1, len(s)):
                    lp += logprobs[i, j - 1, s[j]].item()
                out.append(lp)
            return out


def rerank(context_words, fragment, suggestions, lm_scores, lm_an, func_scores,
           freq):
    """Combine LM scores with err/freq/prefix/bias; returns rescored list.

    suggestions: cheap-path dicts; lm_scores parallel to suggestions.
    lm_an: optional dict word -> LM score under the 'an'-corrected context.
    func_scores: optional {'a': lp, 'an': lp} - LM logprob of the function
    word itself following the context minus its last word. The compound
    decision is made on the JOINT probability
      P('a'|ctx) + P(word|...a)  vs  P('an'|ctx) + P(word|...an)
    so replacing the function word must pay for itself.
    """
    frag = fragment.strip().lower()
    fixable = (context_words and context_words[-1] == "a"
               and not frag.startswith("an"))

    scored = []
    for sug, lm in zip(suggestions, lm_scores):
        word = sug["word"]
        best_lm = lm
        compound = None
        if fixable and word[0] in "aeiou" and word != "a" \
                and lm_an and word in lm_an and func_scores:
            joint_a = func_scores["a"] + lm
            joint_an = func_scores["an"] + lm_an[word]
            if joint_an > joint_a:
                best_lm = joint_an
                compound = "an " + word
            else:
                best_lm = joint_a
        total = config.W_NLM * best_lm
        total += config.W_ERR * sug["err"]
        f = freq.get(word)
        if f:
            total += config.W_FREQ * math.log10(f + 1)
        if sug["err"] == 0 and frag and word.startswith(frag):
            total += config.W_PREFIX * (len(frag) / len(word))
        if word == frag:
            total += config.B_VALID
        scored.append({**sug, "lm": best_lm, "total": total,
                       "compound": compound})

    if not scored:
        return []
    mx = max(s["total"] for s in scored)
    exps = [math.exp(s["total"] - mx) for s in scored]
    z = sum(exps)
    out = []
    for s, e in zip(scored, exps):
        out.append({**s, "prob": e / z})
    out.sort(key=lambda t: -t["total"])
    return out