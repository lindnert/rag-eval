import json
import random
from pathlib import Path

_DEFAULT = Path(__file__).parent / "LLMDRS.jsonl"


def to_metadata(sample: dict) -> dict:
    """Project an LLMDRS sample down to the fields carried through the RAG pipeline.

    LLMDRS has no structural gold — only the GPT-4 ``reference_answer``. Used
    here to probe whether the eval framework detects when recommendations
    deviate from established guidelines.
    """
    g = sample.get("gold", {})
    return {
        "source_dataset": "llmdrs",
        "id": sample.get("id"),
        "reference_answer": g.get("reference_answer"),
    }


def load_llmdrs(path=_DEFAULT, limit=None, shuffle=True, seed=42):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out
