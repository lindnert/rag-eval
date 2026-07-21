"""
Post-generation validation — the filtering deepeval's FiltrationConfig
implies but does not perform (its threshold only triggers rewrite-retries;
nothing is ever discarded).

Stages, each recorded in the attrition report:
  1. dedupe        near-identical inputs (normalized text)
  2. quality       hard cutoff on synthetic_input_quality (critic score)
  3. completeness  drop goldens with empty query/reference_answer/contexts
  4. faithfulness  FaithfulnessMetric of reference_answer vs the golden's OWN
                   context — drops goldens whose reference answer is not
                   supported by the guideline chunks it was generated from

Outputs:
  dataset/synthetic/generated/synthetic_dataset.json  final dataset
  dataset/synthetic/generated/validation_report.json  attrition + histograms

Needs the llama-server gen endpoint (same job as generation). Rerun with
different cutoffs is cheap for stages 1-3; stage 4 scores are cached in the
report-annotated goldens file so a cutoff change does not re-run the LLM.
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from dataset.synthetic.synth_config import (
    OUTPUT_DIR,
    SYNTH_FAITHFULNESS_CUTOFF,
    SYNTH_HARD_CUTOFF,
    SYNTH_MAX_CONCURRENT,
)
from dataset.synthetic.synth_llm import build_critic

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT = _PROJECT_ROOT / OUTPUT_DIR
_SCORED_FILE = _OUT / "goldens_all_scored.json"
_FINAL_FILE = _OUT / "synthetic_dataset.json"
_REPORT_FILE = _OUT / "validation_report.json"


def _norm(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


# A faithful reference answer can still be UNANSWERABLE — the generator, told to
# stay grounded, honestly reports that the context lacks the requested info. The
# faithfulness metric passes those (the hedge IS supported by the context), so we
# drop them explicitly: no unanswerable goldens is a hard requirement (other
# datasets already cover abstention).
_UNANSWERABLE_MARKERS = (
    # English
    "does not offer", "does not provide", "does not contain", "does not specify",
    "does not include", "does not mention", "does not address", "not provided in the context",
    "not available in the context", "the context does not", "no specific",
    "no information", "cannot be determined", "is not specified",
    # German
    "enthält keine", "bietet keine", "keine spezifischen", "keine angaben",
    "nicht im kontext", "der kontext enthält", "lässt sich nicht", "geht nicht auf",
    "keine informationen", "nicht angegeben", "wird nicht", "liegen keine",
)


def _is_answerable(answer: str) -> bool:
    low = (answer or "").lower()
    return not any(m in low for m in _UNANSWERABLE_MARKERS)


def _histogram(scores, edges=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    hist = Counter()
    for s in scores:
        if s is None:
            hist["missing"] += 1
            continue
        for lo, hi in zip(edges, edges[1:]):
            if lo <= s < hi:
                hist[f"[{lo:.1f},{hi if hi <= 1 else 1.0:.1f})"] += 1
                break
    return dict(sorted(hist.items()))


def _load_all_goldens() -> list[dict]:
    records = []
    for f in sorted(_OUT.glob("goldens_*.json")):
        if f.name == _SCORED_FILE.name:
            continue
        with open(f, encoding="utf-8") as fh:
            records.extend(json.load(fh))
    return records


async def _score_faithfulness(records: list[dict]) -> None:
    """Attach faithfulness_of_reference to each record (in place)."""
    semaphore = asyncio.Semaphore(max(1, SYNTH_MAX_CONCURRENT))

    async def _one(rec):
        async with semaphore:
            metric = FaithfulnessMetric(model=build_critic(), async_mode=True)
            test_case = LLMTestCase(
                input=rec["query"],
                actual_output=rec["reference_answer"],
                retrieval_context=rec["contexts"],
            )
            try:
                await metric.a_measure(test_case)
                rec["dataset_metadata"]["faithfulness_of_reference"] = metric.score
                rec["dataset_metadata"]["faithfulness_reason"] = getattr(metric, "reason", None)
            except Exception as e:
                print(f"[validate] faithfulness FAILED for {rec['id']}: {e}", flush=True)
                rec["dataset_metadata"]["faithfulness_of_reference"] = None
                rec["dataset_metadata"]["faithfulness_reason"] = f"ERROR: {e}"

    todo = [r for r in records if "faithfulness_of_reference" not in r["dataset_metadata"]]
    print(f"[validate] scoring faithfulness for {len(todo)} goldens", flush=True)
    await asyncio.gather(*(_one(r) for r in todo))


def main():
    records = _load_all_goldens()
    report = {"total_generated": len(records), "stages": {}}
    print(f"[validate] loaded {len(records)} goldens", flush=True)
    if not records:
        raise SystemExit("no goldens_*.json found — run generate_synthetic first")

    # 1. dedupe
    seen: set[str] = set()
    deduped = []
    for r in records:
        key = _norm(r["query"] or "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    report["stages"]["dedupe"] = {"kept": len(deduped), "dropped": len(records) - len(deduped)}

    # 2. input-quality hard cutoff
    q_scores = [r["dataset_metadata"].get("synthetic_input_quality") for r in deduped]
    report["input_quality_histogram"] = _histogram(q_scores)
    quality_ok = [
        r for r in deduped
        if (r["dataset_metadata"].get("synthetic_input_quality") or 0.0) >= SYNTH_HARD_CUTOFF
    ]
    report["stages"]["quality_cutoff"] = {
        "cutoff": SYNTH_HARD_CUTOFF,
        "kept": len(quality_ok),
        "dropped": len(deduped) - len(quality_ok),
    }

    # 3. completeness
    complete = [
        r for r in quality_ok
        if (r["query"] or "").strip() and (r["reference_answer"] or "").strip() and r["contexts"]
    ]
    report["stages"]["completeness"] = {
        "kept": len(complete),
        "dropped": len(quality_ok) - len(complete),
    }

    # 3b. answerability — drop goldens whose reference answer hedges that the
    # context lacks the requested info (faithfulness would keep these).
    answerable = [r for r in complete if _is_answerable(r["reference_answer"])]
    report["stages"]["answerable"] = {
        "kept": len(answerable),
        "dropped": len(complete) - len(answerable),
    }
    complete = answerable

    # 4. faithfulness of reference_answer vs own context (scores cached)
    if _SCORED_FILE.exists():
        with open(_SCORED_FILE, encoding="utf-8") as f:
            cached = {r["id"]: r["dataset_metadata"] for r in json.load(f)}
        for r in complete:
            meta = cached.get(r["id"])
            if meta and "faithfulness_of_reference" in meta:
                r["dataset_metadata"]["faithfulness_of_reference"] = meta["faithfulness_of_reference"]
                r["dataset_metadata"]["faithfulness_reason"] = meta.get("faithfulness_reason")
    asyncio.run(_score_faithfulness(complete))
    with open(_SCORED_FILE, "w", encoding="utf-8") as f:
        json.dump(complete, f, ensure_ascii=False, indent=2)

    f_scores = [r["dataset_metadata"].get("faithfulness_of_reference") for r in complete]
    report["faithfulness_histogram"] = _histogram(f_scores)
    final = [
        r for r in complete
        if (r["dataset_metadata"].get("faithfulness_of_reference") or 0.0)
        >= SYNTH_FAITHFULNESS_CUTOFF
    ]
    report["stages"]["faithfulness_cutoff"] = {
        "cutoff": SYNTH_FAITHFULNESS_CUTOFF,
        "kept": len(final),
        "dropped": len(complete) - len(final),
    }

    report["final_count"] = len(final)
    report["final_by_condition"] = dict(
        Counter(r["dataset_metadata"]["condition"] for r in final)
    )
    report["final_by_profile"] = dict(
        Counter(r["dataset_metadata"]["styling_profile"] for r in final)
    )
    report["final_by_evolution"] = dict(
        Counter(
            str((r["dataset_metadata"].get("evolutions") or ["none"])[0])
            for r in final
        )
    )

    with open(_FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    with open(_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"[validate] final dataset: {len(final)} goldens -> {_FINAL_FILE}", flush=True)


if __name__ == "__main__":
    main()
