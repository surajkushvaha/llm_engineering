"""Train the tiny GPT student on teacher data (from either generate_data.py
or prepare_hf_dataset.py -- both write the same distill_data.jsonl format).

This is sequence-level distillation: the student just does standard
next-token training on the teacher's (instruction, response) text -- we
never see the teacher's internal logits/weights, only its final text output,
and that text becomes what the student is trained to predict.
Char-level tokenizer, CPU-friendly, meant as a pipeline demo rather than
a production model.

Run order: generate_data.py OR prepare_hf_dataset.py first (either one
produces distill_data.jsonl), THEN this file.
"""

import json
from pathlib import Path

import torch

from model import GPT, GPTConfig

DATA_PATH = Path(__file__).parent / "distill_data.jsonl"
CKPT_PATH = Path(__file__).parent / "student.pt"

BLOCK_SIZE = 256  # max context length in characters the model trains/generates with
BATCH_SIZE = 16  # how many training examples are processed together per step
MAX_STEPS = 2000  # total training steps (not epochs -- each step is one random batch)
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 200  # how often (in steps) to check validation loss and print progress


def load_examples() -> list[dict]:
    """Read the JSONL file produced by generate_data.py / prepare_hf_dataset.py
    back into a list of {"instruction": ..., "response": ...} dicts."""
    examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_corpus(examples: list[dict]) -> str:
    """Turn the list of (instruction, response) pairs into one long string
    the model trains on, with a consistent template around each example so
    the model can learn "after '### Response:', produce an answer, then stop
    at <|end|>". This template is also reused at generation time in main().
    """
    parts = [f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['response']}\n<|end|>\n" for ex in examples]
    return "".join(parts)


def build_vocab(text: str):
    """Char-level tokenizer: the vocabulary is just every unique character
    that appears in the corpus. stoi/itos = "string to index" / "index to
    string", the two lookup tables needed to go from text <-> token ids."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict) -> torch.Tensor:
    """Turn a string into a 1D tensor of integer token ids, one per char."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample `batch_size` random windows of length `block_size` from data.
    x = the window itself, y = the same window shifted one character to the
    right (i.e. "the correct next character at every position of x") --
    that's the actual training signal for next-token prediction.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    examples = load_examples()
    print(f"loaded {len(examples)} teacher examples")
    corpus = build_corpus(examples)
    stoi, itos = build_vocab(corpus)
    print(f"vocab size: {len(stoi)} chars, corpus length: {len(corpus)} chars")

    data = encode(corpus, stoi)
    # Standard train/validation split: hold out the last 10% of the corpus
    # to measure whether the model is generalizing or just memorizing.
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    # BLOCK_SIZE is a target context length, but a small teacher dataset may
    # not have enough characters to fill it (especially the smaller val
    # split) -- clamp so get_batch always has room to slice a full window.
    block_size = min(BLOCK_SIZE, len(train_data) - 2, len(val_data) - 2)
    if block_size < BLOCK_SIZE:
        print(f"corpus too small for block_size={BLOCK_SIZE}, using {block_size} instead")

    config = GPTConfig(vocab_size=len(stoi), block_size=block_size)
    model = GPT(config).to(device)
    print(f"model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(train_data, block_size, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad()  # clear gradients from the previous step
        loss.backward()  # compute gradients of loss w.r.t. every weight
        optimizer.step()  # nudge every weight slightly to reduce the loss

        if step % EVAL_INTERVAL == 0 or step == 1:
            # model.eval() turns off dropout for a clean loss reading; we
            # switch back to model.train() right after so training resumes
            # with dropout active.
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, block_size, BATCH_SIZE, device)
                _, val_loss = model(vx, vy)
            model.train()
            print(f"step {step}: train_loss {loss.item():.4f}, val_loss {val_loss.item():.4f}")
            # What to watch for: train_loss should steadily drop. If
            # val_loss starts climbing while train_loss keeps dropping,
            # that's overfitting -- the model is memorizing the training
            # examples rather than learning a general pattern (very likely
            # with a small teacher dataset, see prepare_hf_dataset.py notes).

    # Save the trained weights *and* everything needed to use them again
    # later (the config to rebuild the same architecture, stoi/itos to
    # convert text <-> token ids the same way).
    torch.save({"model": model.state_dict(), "config": config, "stoi": stoi, "itos": itos}, CKPT_PATH)
    print(f"saved checkpoint to {CKPT_PATH}")

    print("\n--- sample generation ---")
    model.eval()
    # Same "### Instruction: ... ### Response:" template used in
    # build_corpus, so the model recognizes where it's supposed to start
    # generating an answer.
    prompt = "### Instruction:\nWhat is the capital of France?\n### Response:\n"
    idx = encode(prompt, stoi).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=20)
    text = "".join(itos[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
