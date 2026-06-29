"""
Canonical schema for pipeline output rows.

One row = one (sample, variant) pair. Top-level fields are stable across
datasets and variants; dataset-specific fields live under `dataset_metadata`;
RAG-only diagnostics are top-level but null for `no_rag`.

`finalize(row)` reorders keys into CANONICAL_ORDER for readable, diff-friendly
JSONL. Unknown extras are appended at the end so nothing is silently dropped.
"""

CANONICAL_ORDER = (
    "pipeline_id",
    "source_dataset",
    "id",
    "variant",
    "query",
    "reference_answer",
    "answer",
    "rejected",
    "contexts",
    "retrieval_scores",
    "gen_logprob_stats",
    "ragas_scores",
    "deepeval_scores",
    "sc_metadata",
    "dataset_metadata",
)


def finalize(row: dict) -> dict:
    ordered = {k: row[k] for k in CANONICAL_ORDER if k in row}
    extras = {k: v for k, v in row.items() if k not in ordered}
    return {**ordered, **extras}
