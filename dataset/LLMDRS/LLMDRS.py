"""Preprocess issai/LLM_for_Dietary_Recommendation_System into a JSONL dataset.

Reads the 50 English case files in ``cases_results/`` (each ``<id>_0_gpt.txt``
is a single blob: profile prompt + GPT-4 dietary recommendations + a follow-up
diet plan that we discard), and writes ``LLMDRS.jsonl`` with one record per
case in the same shape as NGQA's JSONL:

    {"id": ..., "query": ..., "gold": {"reference_answer": ...}, "source_dataset": "llmdrs"}

Split rules (per user):
  - ``query``  = everything up to the first ``:`` followed by blank line(s)
                 and a ``1.`` numbered-list item.
  - ``gold.reference_answer`` = the numbered list, truncated at
                 "Give a specific diet plan for the day" (the follow-up turn).
"""

import json
import re
from pathlib import Path

_HERE = Path(__file__).parent
_SRC_DIR = _HERE / "cases_results"
_OUT = _HERE / "LLMDRS.jsonl"

_LIST_RE = re.compile(r"(?m)^\s*1\.\s")
_BOUNDARY_RE = re.compile(r"dietary recommendation", re.IGNORECASE)
_CUTOFF = "Give a specific diet plan for the day"


def _parse(text: str, case_id: str) -> dict:
    # 1. Locate the start of the numbered list "1. ..." — unambiguous start of response.
    list_m = _LIST_RE.search(text)
    if not list_m:
        raise ValueError(f"{case_id}: no numbered list found")
    list_start = list_m.start()

    # 2. Find the LAST "dietary recommendation" before the list — that's the
    # boundary heading/transition sentence (the first one is in the opening
    # "Provide dietary recommendation for this patient profile.").
    boundaries = list(_BOUNDARY_RE.finditer(text, 0, list_start))
    if len(boundaries) < 2:
        raise ValueError(f"{case_id}: no boundary 'dietary recommendation' before list")
    last = boundaries[-1]

    # 3. Walk back from the boundary phrase to the start of its sentence/line
    # (previous '.' or '\n'). Everything before that is the patient profile.
    # The boundary heading is always preceded by a run of 2+ whitespace chars
    # (multiple spaces or blank lines) — that's the most reliable separator
    # since some files have the entire profile as a single line of colon-
    # separated key:value pairs with no periods at all.
    gap = None
    for g in re.finditer(r"\s{2,}", text[: last.start()]):
        gap = g
    sent_start = gap.end() if gap else 0

    # 4. Walk forward from the boundary phrase to the next ':' or '\n' to skip
    # the rest of the heading/transition sentence.
    nxt = [i for i in (text.find(":", last.end()), text.find("\n", last.end())) if i != -1]
    sent_end = min(nxt) + 1

    query = text[:sent_start].rstrip()
    answer = text[sent_end:].lstrip()

    cut = answer.find(_CUTOFF)
    if cut != -1:
        answer = answer[:cut]
    answer = answer.rstrip()

    return {
        "id": case_id,
        "source_dataset": "llmdrs",
        "query": query,
        "gold": {"reference_answer": answer},
    }


def main():
    files = sorted(_SRC_DIR.glob("*_gpt.txt"))
    print(f"Found {len(files)} case files in {_SRC_DIR}")

    records = []
    for f in files:
        # filename: "<id>_0_gpt.txt" → case_id "<id>_0"
        # Filenames are "<id>_0_gpt.txt" — the "_0" suffix marks the English
        # variant (vs "_1" Russian, "_2" Kazakh, which we don't load).
        case_id = f.stem.removesuffix("_gpt").removesuffix("_0")
        text = f.read_text(encoding="utf-8")
        records.append(_parse(text, case_id))

    records.sort(key=lambda r: r["id"])
    with open(_OUT, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records → {_OUT}")


if __name__ == "__main__":
    main()
