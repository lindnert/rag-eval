import json
import random
from pathlib import Path

from common.constants import REJECTION_ANSWER

_DEFAULT = Path(__file__).parent / "medqa_test.jsonl"


def to_metadata(sample: dict) -> dict:
    """MEDQA is used as an out-of-domain rejection probe — questions are USMLE-
    style medical (not nutrition), so a well-behaved RAG should retrieve nothing
    relevant and either abstain or flag low confidence.
    """
    return {
        "source_dataset": "medqa",
        "id": sample["id"],
        "reference_answer": sample["gold"]["reference_answer"],
        # Keep the original MEDQA answer around for audit/traceability — the
        # gold we score against was deliberately overwritten (see load_medqa).
        "dataset_metadata": {
            "original_medqa_answer": sample.get("original_medqa_answer"),
        },
    }


def load_medqa(path=_DEFAULT, limit=None, shuffle=True, seed=42):
    """Load MEDQA test split, projecting each raw row to the common schema.

    The source file ships without ids and with `question`/`answer` keys; we
    synthesize `medqa_<idx>` from the row position and normalize to the
    `query` / `gold.reference_answer` shape used by the other loaders.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            out.append({
                "id": f"medqa_{idx}",
                "source_dataset": "medqa",
                "query": row["question"],
                # Out-of-domain rejection probe: the nutrition corpus cannot
                # answer these USMLE questions, so the *correct* behaviour is to
                # abstain. Overwrite the gold with the canonical REJECTION_ANSWER
                # so answer-correctness scores a well-behaved abstention as right
                # and a confident hallucination as wrong. The real USMLE answer
                # is preserved under original_usmle_answer for traceability.
                "gold": {"reference_answer": REJECTION_ANSWER},
                "original_medqa_answer": row["answer"],
            })
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out
