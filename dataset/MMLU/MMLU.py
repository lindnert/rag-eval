"""Preprocess the MMLU "nutrition" subject into a single JSONL dataset.

Reads the ``test`` parquet split (306 questions — ``dev`` and ``validation``
are tiny and skipped) and writes ``MMLU.jsonl`` with one record per question
in the same shape as the other datasets in this repo:

    {
        "id": "mmlu_test_<row_idx>",
        "source_dataset": "mmlu",
        "query": "<question>",
        "gold": {
            "reference_answer": "<text of the correct choice>",
            "choices":          ["<A>", "<B>", "<C>", "<D>"],
            "answer_index":     <int 0..3>,
        },
    }

The MCQ choices are *not* baked into ``query`` — we want the model to retrieve
and answer open-endedly, not pick from four options. The full choice list and
the gold index are kept in ``gold`` so a stricter MCQ-style probe remains
possible later without re-processing.
"""

import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
_SRC = _HERE / "test-00000-of-00001.parquet"
_OUT = _HERE / "MMLU.jsonl"


def _row_to_record(question: str, choices: list, ans_idx: int, idx: int) -> dict:
    return {
        "id": f"mmlu_{idx}",
        "source_dataset": "mmlu",
        "query": question.rstrip(),
        "gold": {
            "reference_answer": choices[ans_idx],
            "choices": choices,
            "answer_index": ans_idx,
        },
    }


def main():
    df = pd.read_parquet(_SRC).reset_index(drop=True)
    records = [
        _row_to_record(
            row["question"],
            list(row["choices"]),
            int(row["answer"]),
            i,
        )
        for i, row in enumerate(df.to_dict("records"))
    ]

    with open(_OUT, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records -> {_OUT}")


if __name__ == "__main__":
    main()
