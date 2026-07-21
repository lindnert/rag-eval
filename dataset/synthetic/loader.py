import json
import os
import random
from pathlib import Path

_SYNTH_DIR = Path(__file__).resolve().parent


def _default_dataset() -> Path:
    """Resolve the dataset to load when no explicit path is given.

    The finished dataset is the curated, git-tracked top-level
    dataset/synthetic/synthetic_dataset.json (also the file present on the SLURM
    node — the per-run generated_* dirs are gitignored). Override with
    SYNTH_DATASET_FILE (e.g. to evaluate a raw generated_*/ pilot run) or pass an
    explicit `path` to load.py.
    """
    override = os.getenv("SYNTH_DATASET_FILE")
    if override:
        return Path(override)
    curated = _SYNTH_DIR / "synthetic_dataset.json"
    if not curated.exists():
        raise FileNotFoundError(
            f"no {curated} — put your finished dataset there, or pass an explicit "
            f"path / SYNTH_DATASET_FILE."
        )
    return curated


def to_metadata(sample: dict) -> dict:
    """Project a synthetic golden down to the fields the RAG pipeline carries.

    The records already store the canonical top-level keys (``source_dataset``,
    ``id``, ``reference_answer``, ``query``) plus a rich ``dataset_metadata``
    block, so this is mostly a pass-through. ``reference_contexts`` preserves the
    guideline chunks the golden was generated from — the pipeline retrieves its
    own contexts, but keeping the gold ones lets later analyses compute
    context recall.
    """
    return {
        "source_dataset": sample.get("source_dataset", "synthetic_guidelines"),
        "id": sample.get("id"),
        "reference_answer": sample.get("reference_answer"),
        "dataset_metadata": {
            **(sample.get("dataset_metadata") or {}),
            "reference_contexts": sample.get("contexts"),
        },
    }


def load_synthetic(path=None, lang=None, limit=None, shuffle=True, seed=42):
    """Load the validated synthetic guideline goldens (a flat JSON list).

    `path` defaults to the most recent run's dataset (see `_default_dataset`).
    `lang` ('en'|'de') filters to goldens whose question language matches — pass
    RAG_LANG so a per-language pipeline run answers each query with a
    same-language system prompt (the German goldens ride the RAG_LANG=de run,
    the English ones the RAG_LANG=en run), with no per-query prompt switching.
    """
    path = Path(path) if path is not None else _default_dataset()
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    if lang is not None:
        out = [
            r for r in out
            if (r.get("dataset_metadata") or {}).get("question_lang") == lang
        ]
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out
