"""Convert a sample of WithinUsAI/GPT_5.5_Distilled into our distill_data.jsonl
format (instruction/response pairs), as an alternative teacher source to
calling a live API in generate_data.py.

Vetted by hand before use: unlike some similarly-named "distilled" datasets
on the hub, this one's rows are genuine substantive answers (not templated
filler) -- *once filtered by quality_score*. A random sample of the raw
dataset showed ~12.5% of rows are generic templated filler (e.g. "I could
approach this in several ways: ... Final answer: ..." without ever actually
answering the question) -- and those rows are reliably the ones scored 0.5,
the lowest bucket. Rows scored >= 0.9 had zero filler hits in that check,
which is why MIN_QUALITY_SCORE below is set there. Don't lower it without
re-checking a sample by hand.

This is one of two ways to get training data (see also generate_data.py,
which calls a live model instead of reading a pre-built dataset). Both
scripts write to the same distill_data.jsonl, which train.py reads either
way -- run whichever one of the two you want, then run train.py.
"""

import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "WithinUsAI/GPT_5.5_Distilled"
FILENAME = "GPT_5.5_Distilled.jsonl"
SAMPLE_SIZE = 30
MIN_QUALITY_SCORE = 0.9  # ~half the dataset is templated filler scored 0.5; this field reliably separates it out
SEED = 0  # fixed seed so re-running picks the same sample instead of a different random 30 each time
OUT_PATH = Path(__file__).parent / "distill_data.jsonl"

# Each raw row's "text" field is one whole chat transcript like:
#   <|user|>\n{question}\n<|assistant|>\n<think>{reasoning}</think>\n{answer}
# These two regexes pull out the pieces we actually want.
TEMPLATE_RE = re.compile(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>\s*(.*)", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_row(text: str) -> dict | None:
    """Pull {"instruction": ..., "response": ...} out of one raw transcript
    string, dropping the <think>...</think> reasoning trace (if present) --
    we only want the final answer as the training target, to keep things
    simple for the tiny char-level student model.
    Returns None if the row doesn't match the expected template at all.
    """
    match = TEMPLATE_RE.search(text)
    if not match:
        return None
    instruction, response = match.groups()
    response = THINK_RE.sub("", response).strip()
    return {"instruction": instruction.strip(), "response": response}


def main():
    # Downloads (and caches locally) just this one file from the HF dataset
    # repo -- no need to clone the whole repo or run any of its code.
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME)

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    rows = [r for r in rows if r.get("quality_score", 0) >= MIN_QUALITY_SCORE]
    print(f"{len(rows)} rows with quality_score >= {MIN_QUALITY_SCORE}")

    random.seed(SEED)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    examples = []
    for row in sample:
        parsed = parse_row(row["text"])
        if parsed and parsed["instruction"] and parsed["response"]:
            examples.append(parsed)

    # Same JSONL format generate_data.py writes, so train.py doesn't care
    # which of the two scripts produced this file.
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples (from {len(sample)} sampled rows) to {OUT_PATH}")


if __name__ == "__main__":
    main()
