import json
import os
import random
from pathlib import Path

_SYNTH_DIR = Path(__file__).resolve().parent


def _default_dataset() -> Path:
    """Resolve the dataset to load when no explicit path is given.

    Runs write to per-run dirs (dataset/synthetic/generated_<model>_<stamp>/),
    so there is no single fixed filename. Honour SYNTH_DATASET_FILE if set;
    otherwise pick the most recently modified synthetic_dataset.json under any
    generated_* dir. Callers can always pass an explicit `path` to load.py.
    """
    override = os.getenv("SYNTH_DATASET_FILE")
    if override:
        return Path(override)
    candidates = sorted(
        _SYNTH_DIR.glob("generated*/synthetic_dataset.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no generated*/synthetic_dataset.json under {_SYNTH_DIR} — run the "
            f"generation pipeline first, or pass an explicit path / "
            f"SYNTH_DATASET_FILE."
        )
    return candidates[0]


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
