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
    dataset_metadata = sample.get("dataset_metadata") or {}
    return {
        "source_dataset": sample.get("source_dataset", "synthetic_guidelines"),
        "id": sample.get("id"),
        "reference_answer": sample.get("reference_answer"),
        # Per-query language ('en'|'de') so the pipeline answers each golden in
        # its own language (see rag.utils.process_single_query); the other
        # datasets are English and tagged 'en' at load time.
        "lang": dataset_metadata.get("question_lang", "en"),
        "dataset_metadata": {
            **dataset_metadata,
            "reference_contexts": sample.get("contexts"),
        },
    }


def load_synthetic(path=None, lang=None, limit=None, shuffle=True, seed=42):
    """Load the validated synthetic guideline goldens (a flat JSON list).

    `path` defaults to the most recent run's dataset (see `_default_dataset`).
    `lang` ('en'|'de') optionally filters to goldens whose question language
    matches. Leave it None (the default) so a single run loads both languages;
    the pipeline then answers each golden with a same-language prompt selected
    per query from its `question_lang` (see rag.utils / dataset.synthetic loader
    `to_metadata`).
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
