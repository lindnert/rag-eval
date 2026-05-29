import json
import random
from pathlib import Path

_DEFAULT = Path(__file__).parent / "NGQA.jsonl"


def to_metadata(sample: dict) -> dict:
    """Project an NGQA sample down to the fields carried through the RAG pipeline.

    The pipeline expects ``source_dataset``, ``id``, and ``reference_answer`` at
    the top level; anything dataset-specific goes under ``dataset_metadata`` so
    the cross-dataset top-level schema stays stable.
    """
    g = sample.get("gold", {})
    return {
        "source_dataset": "ngqa",
        "id": sample.get("id"),
        "reference_answer": g.get("reference_answer"),
        "dataset_metadata": {
            "difficulty": sample.get("difficulty"),
            "has_conflict": bool(g.get("conflicts")),
            "csv_short_answer": g.get("csv_short_answer"),
            "is_healthy_agrees_with_csv_answer": g.get(
                "is_healthy_agrees_with_csv_answer"
            ),
        },
    }


def load_ngqa(
    path=_DEFAULT,
    difficulty=None,
    has_conflict=None,
    is_healthy_agrees_with_csv_answer=None,
    limit=None,
    shuffle=True,
    seed=42,
):
    """
    Stream NGQA samples from the preprocessed JSONL, with optional filtering.

    Filters:
      - difficulty: 'easy' | 'medium' | 'hard'.
      - has_conflict: structural filter — True/False on whether the graph
            encodes any user-condition vs. nutrient-tag contradict edge.
      - is_healthy_agrees_with_csv_answer: labeling filter — True/False on
            whether NGQA's CSV short-answer polarity (Yes/No) matches the
            user-specific is_healthy derived from contradict edges.
            Disagreement only occurs on the hard split (~614 samples), always
            with is_healthy=False but csv_short_answer="Yes".
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if difficulty is not None and s.get("difficulty") != difficulty:
                continue
            if has_conflict is not None and bool(s["gold"]["conflicts"]) != has_conflict:
                continue
            if is_healthy_agrees_with_csv_answer is not None and \
                    s["gold"].get("is_healthy_agrees_with_csv_answer") != is_healthy_agrees_with_csv_answer:
                continue
            out.append(s)
    # Shuffle the full filtered set *before* truncating so `limit=N` returns a
    # diverse sample rather than the first N rows of a food-sorted CSV.
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out
