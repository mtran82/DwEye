"""Spike: load Qwen2.5-0.5B, KV-cache context, batch-score candidates.

Measures whether the neural reranker fits the latency budget:
  - load once
  - forward context once (past_key_values)
  - one batched forward over N candidate continuations
Reports wall time per suggestion call and sanity of the ranking.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16).cuda().eval()
print(f"loaded in {time.time()-t0:.1f}s")


def score(context_text, cand_words):
    """LM logprob of each candidate word following context_text."""
    ctx_ids = tok(context_text, return_tensors="pt").input_ids.cuda()
    seqs = [tok(" " + w, add_special_tokens=False).input_ids for w in cand_words]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id or tok.eos_token_id
    batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    with torch.no_grad():
        n = len(seqs)
        # build the past with batch size n so candidate KV shapes match
        ctx_batch = ctx_ids.expand(n, -1)
        ctx_out = model(ctx_batch, use_cache=True)
        past = ctx_out.past_key_values
        # last context position predicts each candidate's first token
        ctx_logprobs = torch.log_softmax(ctx_out.logits[:, -1].float(), dim=-1)
        logits = model(batch.cuda(), attention_mask=mask.cuda(),
                       past_key_values=past).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, s in enumerate(seqs):
        lp = ctx_logprobs[i, s[0]].item()
        for j, tid in enumerate(s):
            if j == 0:
                continue
            lp += logprobs[i, j - 1, tid].item()
        out.append(lp)
    return out


context = "johnny gave a"
candidates = ["apple", "pal", "answer", "carrot", "smile", "prayer",
              "applecart", "application", "elephant", "apology", "ape",
              "amber", "aim", "anvil", "ankle", "ally", "album", "anchovy",
              "arrogant", "ostrich"]

lp = score(context, candidates)
ranked = sorted(zip(candidates, lp), key=lambda t: -t[1])
print("top 5 by LM:")
for w, l in ranked[:5]:
    print(f"  {w:12} {l:7.2f}")
print("bottom 3:")
for w, l in ranked[-3:]:
    print(f"  {w:12} {l:7.2f}")

t0 = time.time()
for _ in range(10):
    score(context, candidates)
dt = (time.time() - t0) / 10
print(f"batch of {len(candidates)} candidates: {dt*1000:.1f}ms")