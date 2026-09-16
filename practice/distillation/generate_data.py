"""Generate a small teacher dataset for distillation, via the openai SDK
pointed at Ollama Cloud's OpenAI-compatible endpoint (OPENAI_BASE_URL).

We don't have access to a teacher model's logits/weights, so this is
*sequence-level* distillation (Kim & Rush, 2016): the teacher's generated
text becomes the training targets for the small student model in train.py.

Two API calls total, to keep this a cheap demo batch:
1. Ask the teacher to invent N diverse instructions.
2. Ask the teacher to answer all of them in one pass, as JSON.

TEACHER_MODEL must be a model tag Ollama Cloud actually hosts (check
https://ollama.com/search?c=cloud or `ollama list` for available cloud
models) -- swap it if the default below isn't available on your account.

This is one of two ways to get training data (see also prepare_hf_dataset.py,
which reads a pre-built dataset instead of calling an API). Both scripts
write to the same distill_data.jsonl, which train.py reads either way --
run whichever one of the two you want, then run train.py.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY / OPENAI_BASE_URL etc. from the .env file into the environment

TEACHER_MODEL = "gpt-oss:120b"
NUM_EXAMPLES = 30
OUT_PATH = Path(__file__).parent / "distill_data.jsonl"


def require_client() -> OpenAI:
    """Build the API client, failing fast with a clear message if the
    required .env values are missing, rather than a confusing error later."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "OPENAI_BASE_URL and OPENAI_API_KEY must be set in .env to reach "
            "the teacher endpoint."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def chat(client: OpenAI, prompt: str, max_tokens: int) -> str:
    """Send one message, get back the model's reply text. Small wrapper so
    the two functions below don't repeat this boilerplate."""
    resp = client.chat.completions.create(
        model=TEACHER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def generate_instructions(client: OpenAI, n: int) -> list[str]:
    """Ask the teacher model to invent n instructions/questions on its own,
    e.g. "What is the capital of France?" or "Write a haiku about rain".
    We ask for a JSON array back so this is easy to parse programmatically.
    """
    prompt = (
        f"Generate exactly {n} short, diverse instructions/questions a user "
        "might ask an AI assistant (mix of factual questions, short "
        "explanations, simple how-tos, and short creative prompts). "
        "Return ONLY a JSON array of strings, no other text."
    )
    return json.loads(chat(client, prompt, max_tokens=1024))


def generate_responses(client: OpenAI, instructions: list[str]) -> list[dict]:
    """Ask the teacher to answer every generated instruction, all in one
    request (cheaper than one API call per instruction). Returns a list of
    {"instruction": ..., "response": ...} dicts -- this is the actual
    teacher data the student model will train on.
    """
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(instructions))
    prompt = (
        "Answer each of the following instructions concisely (2-4 sentences "
        "each). Return ONLY a JSON array of objects with keys "
        '"instruction" and "response", in the same order, no other text.\n\n'
        f"{numbered}"
    )
    return json.loads(chat(client, prompt, max_tokens=4096))


def main():
    client = require_client()

    print(f"Asking {TEACHER_MODEL} to invent {NUM_EXAMPLES} instructions...")
    instructions = generate_instructions(client, NUM_EXAMPLES)
    print(f"Got {len(instructions)} instructions. Asking {TEACHER_MODEL} to answer them...")

    examples = generate_responses(client, instructions)

    # JSON Lines format: one JSON object per line. Chosen because it's easy
    # to append to, stream, and read one example at a time (see train.py's
    # load_examples), unlike a single big JSON array.
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} teacher examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
