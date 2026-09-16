# Distillation mini-project — conversation summary

## Goal and how it got scoped down

Started as "create a 100M param model with distillation of Claude Opus 5."
Two things changed the plan along the way:

- **No real access to Opus 5's internals.** No API exposes a model's logits
  or weights, so literal weight/logit distillation isn't possible against
  any hosted model. The practical alternative is *sequence-level
  distillation* (Kim & Rush, 2016): a strong "teacher" model generates text,
  and a small "student" model is trained to reproduce that text.
- **Hardware.** This machine has no CUDA GPU (`torch.cuda.is_available()`
  is `False`), so 100M params was scaled down to **~10M params**, similar to
  nanoGPT's classic "shakespeare_char" reference config.
- **Teacher model access.** `.env` had no `ANTHROPIC_API_KEY`, only
  `OPENAI_API_KEY`/`OPENAI_BASE_URL` pointed at `https://ollama.com/v1/`
  (Ollama Cloud). Ollama Cloud doesn't host Claude Opus 5, only open-weight
  models, so the teacher call uses `gpt-oss:120b` via the `openai` SDK
  instead of the `anthropic` SDK.

## Files built (`practice/distillation/`)

- **`model.py`** — the student: a from-scratch decoder-only transformer
  (nanoGPT-style), 6 layers, 6 heads, 384 embedding dim, **~10.7M params**.
  Uses a **character-level vocabulary** (built from the training text
  itself) instead of a BPE tokenizer, because a real BPE vocab (50k+ tokens)
  would make the embedding table alone (`vocab_size × n_embd`) bigger than
  the entire rest of a 10M-param model.
- **`generate_data.py`** — teacher data via a **live API call**: asks
  `gpt-oss:120b` (Ollama Cloud) to invent 30 instructions, then answer all
  of them, saving the pairs to `distill_data.jsonl`. Two API calls total.
- **`prepare_hf_dataset.py`** — teacher data via a **pre-built dataset**
  instead (no API cost): samples 30 rows from
  `WithinUsAI/GPT_5.5_Distilled` on Hugging Face, filtered to
  `quality_score >= 0.9`, and converts its `<|user|>/<|assistant|>` chat
  template into the same `instruction`/`response` JSONL format.
- **`train.py`** — loads `distill_data.jsonl` (from either script above),
  builds the char vocab, trains the student with standard next-character
  prediction (2000 steps, CPU-friendly), saves `student.pt`, and prints a
  sample generation at the end.

Both data-source scripts write the same `distill_data.jsonl`, so `train.py`
doesn't care which one produced it. **Run order:** `generate_data.py` *or*
`prepare_hf_dataset.py`, then `train.py`.

`.gitignore` was updated to exclude `practice/distillation/*.pt` and
`distill_data.jsonl` — they're generated artifacts, not source.

## Dataset vetting: the important detour

Before settling on `WithinUsAI/GPT_5.5_Distilled`, two other "distillation"
datasets were checked and rejected:

1. **`saidutta69/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-Mythos-5-Qwen-3.7-Max-Distillation-Cleaned`**
   — every sampled response started with identical boilerplate
   ("Drawing from the autonomous, frontier-level reasoning characteristic
   of Claude Mythos...`"), contained fake code (comments describing code
   that was never written), fabricated unverifiable benchmark numbers, and
   ended with the literal disclaimer *"This response was generated to
   exemplify the distilled Mythos reasoning style for training purposes."*
   Confirmed synthetic filler, not real model output — rejected outright.

2. **`ansulev/claude-mythos-distilled-25k`** — inspection started but the
   user redirected to a different dataset before it was fully vetted.

3. **`WithinUsAI/GPT_5.5_Distilled`** (the one actually used) — the first
   3 rows read looked genuinely substantive (real explanations of ACID
   transactions, event-sourcing architecture, CI/CD design). But a **true
   random sample of 30 rows** revealed ~12.5% of the full dataset is the
   *same kind* of problem as dataset #1, just a different template (e.g.
   "I could approach this in several ways: ... Final answer: The approach
   provides reliable insights..." — a generic scaffold that never actually
   answers the question). Lesson: **the first few rows of a file are not a
   representative sample** — this dataset happens to be sorted with its
   best rows first.

   The fix: the dataset's own `quality_score` field (values `0.5` to
   `0.95`) reliably separates the filler from the real answers — every
   filler-marker row scored `0.5` (the lowest bucket, ~49% of all rows),
   while the `quality_score >= 0.9` subset (2,055 of 18,197 rows) had
   **zero** filler hits in a targeted check. `prepare_hf_dataset.py` filters
   on this before sampling.

Side note: the user corrected an initial assumption that "Claude Mythos"
was necessarily a fabricated model name — a web search turned up
Anthropic-branded sources describing Mythos as a real (restricted-access)
model class released in 2026, after this assistant's January 2026 knowledge
cutoff. That's plausible and was conceded. It doesn't change the dataset
finding above, though — real model name or not, the specific sampled rows
were independently confirmed to be synthetic filler by their content, not
by the model name attached to them.

## Bugs hit and fixed

- **`get_batch` crash on a tiny corpus.** `torch.randint(len(data) -
  block_size - 1, ...)` needs a positive range; with a very short corpus
  (first tested with a 3-example mock dataset) and the default
  `block_size=256`, there weren't enough characters to slice a full window,
  especially in the smaller validation split. Fixed by clamping
  `block_size = min(BLOCK_SIZE, len(train_data) - 2, len(val_data) - 2)` in
  `train.py`, so it always leaves room for `get_batch` to pick a valid
  window regardless of dataset size.

## Concepts covered along the way

- **Sequence-level vs. logit-level distillation** — we only ever have
  access to a teacher's *text output*, never its internal probabilities,
  so the student trains on that text directly (standard supervised
  fine-tuning), not on soft-label logits.
- **Why char-level tokenization** — keeps the embedding table
  (`vocab_size × n_embd`) tiny so the ~10M param budget goes to the
  transformer body instead. Tradeoff: a token is only 1 character, so the
  same text becomes a much longer sequence than with a real BPE tokenizer,
  making the model work harder to learn things like whole words from
  scratch.
- **`block_size` / context length** — how many previous tokens (here:
  characters, since char-level) the model can see at once. Not the length
  of one token — that's a separate, unrelated axis. Bigger context = more
  compute (attention cost grows quadratically with it).
- **`n_embd`, `n_head`, `head_dim`** — `n_embd` is the size of the vector
  representing each token internally. Multi-head attention splits that
  vector into `n_head` equal chunks of `head_dim = n_embd / n_head` each,
  and runs attention independently on each chunk so different heads can
  specialize in different relationships between tokens. This is entirely
  separate from tokenization — it only describes what happens to a token's
  vector *after* it's already been embedded.
- **`assert`** — a sanity check that crashes immediately with a clear error
  if a condition that should always hold turns out false (e.g.
  `n_embd % n_head == 0`, needed so the embedding vector splits evenly
  across heads), rather than failing confusingly somewhere deeper later.

## Standing preferences noted (saved to memory)

- Even when explicitly asked to implement code in this repo, keep narrating
  what's being built and why, not just delivering finished files.
- Comment code generously in this repo specifically (function purpose,
  non-obvious steps) — the opposite of the usual minimal-comment default —
  since this is for self-learning and gets revisited without chat context.
