import json
import random
from pathlib import Path

_DEFAULT = Path(__file__).parent / "MMLU.jsonl"


def to_metadata(sample: dict) -> dict:
    """Project an MMLU sample down to the fields carried through the RAG pipeline.

    MMLU is multiple-choice, but we run it open-ended (choices are not shown
    to the model). ``reference_answer`` is the text of the correct choice —
    downstream evaluators should compare the model's free-form answer to it
    semantically rather than via exact string match.
    """
    g = sample.get("gold", {})
    return {
        "source_dataset": "mmlu",
        "id": sample.get("id"),
        "reference_answer": g.get("reference_answer"),
    }


def load_mmlu(path=_DEFAULT, limit=None, shuffle=True, seed=42):
    """Stream MMLU-nutrition (test split) samples from the preprocessed JSONL."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out
