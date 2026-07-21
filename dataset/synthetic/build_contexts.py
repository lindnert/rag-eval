"""
Build the Path-C context sets + persona pool for synthetic generation.

Runs WITHOUT a GPU model: it queries the existing bge-m3 FAISS index by
vector (reconstructed from the index itself), so no embedding calls are made.
Run it on the cluster login node where richtlinien/faiss_index_bge_m3_cosine
exists (or locally with FAISS_INDEX_DIR pointing at a copy):

    python -m dataset.synthetic.build_contexts

Two complementary context sets are written (a hybrid design — personalization
matters more than cross-lingual, but both are covered):

  reference  (contexts_reference.json): 1 German DGE reference-value table +
             its English IOM counterpart tables for the SAME life-stage
             (age/sex). Cross-lingual, personalized by demographic. DGE and IOM
             use different age cutoffs, so bands are aligned by INDEX from the
             oldest end (see build_reference_contexts) — the adult bands line up
             exactly, which is what every (adult) persona needs.

  condition  (contexts_condition.json): English guideline chunks selected for a
             curated persona's clinical conditions (e.g. a CKD + diabetes
             persona gets kidney + diabetes chunks). Monolingual English,
             personalized by condition; the question bridges the chunks.

Every context holds exactly RAG_K chunks — the same k the retriever fetches at
inference — so a synthetic context mirrors what the RAG system will actually
see, rather than an arbitrary size.

Personas (personas.json) merge two sources: NGQA/NHANES profiles sampled
coverage-greedily so every distinct condition tag appears, plus the curated,
corpus-grounded personas in personas_curated.json.
"""

import json
import os
import random
import re
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from rag.llm_config import RAG_K  # context size == retriever's k (not arbitrary)
from dataset.synthetic.synth_config import (
    CHUNKS_FILE,
    CONTEXTS_CONDITION_FILE,
    CONTEXTS_REFERENCE_FILE,
    CURATED_PERSONAS_FILE,
    NGQA_FILE,
    PERSONAS_FILE,
    SYNTH_SEED,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Mirrors retrieval.embeddings.PASSAGE_PREFIX (imported by value to avoid
# pulling in the retrieval package's BM25/embedding dependencies here).
PASSAGE_PREFIX = ""
FAISS_INDEX_DIR = os.getenv(
    "FAISS_INDEX_DIR",
    str(_PROJECT_ROOT / "richtlinien" / "faiss_index_bge_m3_cosine"),
)

# Maps a persona condition tag to substrings that identify the guideline files
# covering it (matched case-insensitively against chunk `source`). Keys are the
# tags used in personas_curated.json's "conditions".
CONDITION_SOURCE_KEYWORDS = {
    "kidney": ["kidney", "renal"],
    # "nutrition_in_liver", not bare "liver": the latter also matches the
    # obesity-in-GI/liver-disease joint guideline, which belongs to obesity.
    "liver": ["nutrition_in_liver"],
    "cancer": ["cancer", "eating-hints", "nci"],
    "ibd": ["inflammatory_bowel", "inflammatory-bowel"],
    "pancreatitis": ["pancreat"],
    "obesity": ["obesity"],
    "micronutrient": ["micronutrient", "vitamind", "vitamin_d", "vitamin-d", "calcium"],
    "celiac": ["celiac", "gluten"],
    "osteoporosis": ["osteoporosis"],
    "diabetes": ["diab"],
    "heart": ["cholesterol", "heart"],
    "intestinal_failure": ["intestinal_failure", "intestinal-failure", "home_parenteral", "parenteral"],
    "polymorbid": ["polymorbid"],
    "hiv": ["hiv"],
}


class _StubEmbeddings(Embeddings):
    """Index is queried by reconstructed vectors only — never embeds."""

    def embed_documents(self, texts):
        raise RuntimeError("_StubEmbeddings should never be called")

    def embed_query(self, text):
        raise RuntimeError(
            "_StubEmbeddings should never be called — query by vector instead"
        )


def _load_chunks() -> list[dict]:
    with open(_PROJECT_ROOT / CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _load_index(n_chunks: int):
    vs = FAISS.load_local(
        FAISS_INDEX_DIR,
        _StubEmbeddings(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    if vs.index.ntotal != n_chunks:
        raise SystemExit(
            f"ERROR: index at {FAISS_INDEX_DIR} has {vs.index.ntotal} vectors but "
            f"{CHUNKS_FILE} has {n_chunks} chunks — the index was built from a "
            f"different chunk set. Rebuild it (slurm/build_faiss_index.sh) or "
            f"point FAISS_INDEX_DIR at the matching one."
        )
    return vs


def _verify_alignment(vs, chunks) -> None:
    """Vector i must correspond to chunks[i] (from_texts preserves order)."""
    for i in (0, len(chunks) // 2, len(chunks) - 1):
        doc = vs.docstore.search(vs.index_to_docstore_id[i])
        expected = PASSAGE_PREFIX + chunks[i]["text"]
        if doc.page_content != expected:
            raise SystemExit(
                f"ERROR: docstore position {i} does not match chunk {i} — "
                f"cannot map vectors to chunk ids safely."
            )


def _chunk_record(chunks, idx, **extra) -> dict:
    m = chunks[idx]["metadata"]
    rec = {
        "chunk_id": idx,
        "text": chunks[idx]["text"],
        "source": m.get("source"),
        "folder": m.get("folder"),
        "lang": m.get("lang"),
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Life-stage parsing (reference contexts)
# ---------------------------------------------------------------------------
_OLD = 120  # open-ended upper bound in years for ">70" / "und älter"


def _parse_dge_age(text: str):
    """(low, high) in years for a DGE 'Altersgruppe' label, else None.

    Skips month-based (infant) bands and the non-age groups (pregnancy /
    lactation / bare numbers) so only comparable year-based bands are aligned.
    """
    m = re.search(r"Altersgruppe:\s*([^\n.]+)", text)
    if not m:
        return None
    label = m.group(1).strip()
    if "Monate" in label:  # infants, months — skip
        return None
    r = re.search(r"(\d+)\s*bis\s*unter\s*(\d+)\s*Jahre", label)
    if r:
        return (int(r.group(1)), int(r.group(2)))
    r = re.search(r"(\d+)\s*Jahre\s*und\s*[äa]lter", label)
    if r:
        return (int(r.group(1)), _OLD)
    return None


def _parse_iom_group(text: str):
    """(sex, low, high) for an IOM 'Life-stage group' label, else None.

    sex in {'M','F','C'} (C = sexless Children). Skips month-based infants and
    the pregnancy/lactation rows (no German counterpart to compare against).
    """
    m = re.search(r"Life-stage group:\s*([^\n.]+)", text)
    if not m:
        return None
    label = m.group(1).strip()
    if "mo" in label.lower():  # infants in months — skip
        return None
    low = label.lower()
    if low.startswith("male"):
        sex = "M"
    elif low.startswith("female"):
        sex = "F"
    elif low.startswith("child"):
        sex = "C"
    else:  # pregnancy / lactation
        return None
    if ">" in label:  # "> 70 y"
        n = re.search(r">\s*(\d+)", label)
        return (sex, int(n.group(1)), _OLD) if n else None
    nums = re.findall(r"\d+", label)
    if len(nums) >= 2:
        return (sex, int(nums[0]), int(nums[1]))
    return None


# German->English nutrient names, so DGE and IOM tables can be matched on shared
# nutrients (values differ; the names are what makes a comparison answerable).
_DE2EN_NUTRIENT = {
    "calcium": "calcium", "eisen": "iron", "zink": "zinc", "magnesium": "magnesium",
    "kupfer": "copper", "jod": "iodine", "selen": "selenium", "phosphor": "phosphorus",
    "mangan": "manganese", "molybd": "molybdenum", "fluorid": "fluoride",
    "chrom": "chromium", "kalium": "potassium", "natrium": "sodium", "chlorid": "chloride",
    "vitamin a": "vitamin a", "vitamin c": "vitamin c", "vitamin d": "vitamin d",
    "vitamin e": "vitamin e", "vitamin k": "vitamin k", "vitamin b6": "vitamin b6",
    "vitamin b12": "vitamin b12", "folat": "folate", "niacin": "niacin",
    "thiamin": "thiamin", "riboflavin": "riboflavin", "biotin": "biotin",
}


def _dge_nutrients(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        m = re.match(r"-\s*([^(:\n]+?)\s*[\(:]", line)
        if not m:
            continue
        name = m.group(1).strip().lower()
        for de, en in _DE2EN_NUTRIENT.items():
            if name.startswith(de):
                out.add(en)
    return out


def _iom_nutrients(text: str) -> set[str]:
    out = set()
    valid = set(_DE2EN_NUTRIENT.values())
    for line in text.splitlines():
        m = re.match(r"-\s*([^:\n]+?):", line)
        if not m:
            continue
        name = re.sub(r"\s*\(.*", "", m.group(1).strip().lower())
        if name in valid:
            out.add(name)
    return out


def _iom_table_type(text: str) -> str:
    """The DRI table kind (RDA/AI Elements, EAR, UL Vitamins, ...), so male and
    female chunks of the SAME kind are paired (identical nutrient columns)."""
    m = re.search(r"\(DRIs\)\s*[—–-]\s*(.*?)\.\s*Life-stage", text)
    return m.group(1).strip() if m else "?"


# ---------------------------------------------------------------------------
# Builder 1: cross-lingual reference contexts (life-stage aligned by index)
# ---------------------------------------------------------------------------
def build_reference_contexts(chunks, all_vecs) -> list[dict]:
    def src(i):
        return chunks[i]["metadata"].get("source") or ""

    # DGE year-bands -> the chunk ids that carry them.
    dge_bands: dict[tuple, list[int]] = {}
    for i in range(len(chunks)):
        if "DGE-Referenzwerte" in src(i):
            band = _parse_dge_age(chunks[i]["text"])
            if band:
                dge_bands.setdefault(band, []).append(i)

    # IOM year-ranges -> {table_type -> {sex: chunk id}} (male & female chunks of
    # the same table type share nutrient columns, so they compare cleanly).
    iom_ranges: dict[tuple, dict[str, dict[str, int]]] = {}
    for i in range(len(chunks)):
        if "intstitute" in src(i).lower():
            p = _parse_iom_group(chunks[i]["text"])
            if p:
                sex, lo, hi = p
                tt = _iom_table_type(chunks[i]["text"])
                iom_ranges.setdefault((lo, hi), {}).setdefault(tt, {})[sex] = i

    # Align by INDEX from the OLDEST end: DGE has more (child) bands than IOM, so
    # anchoring at the top keeps the adult bands — the ones every persona falls
    # in — exactly matched. The 2 youngest DGE bands are left unpaired.
    dge_desc = sorted(dge_bands, reverse=True)
    iom_desc = sorted(iom_ranges, reverse=True)
    n_pairs = min(len(dge_desc), len(iom_desc))
    n_en = RAG_K - 1  # English partners per context

    contexts: list[dict] = []
    for k in range(n_pairs):
        band = dge_desc[k]
        irange = iom_desc[k]
        # Sanity gate: index alignment is anchored at the adult end, so the older
        # bands line up but the youngest DGE bands (DGE splits children finer) end
        # up index-paired with a non-overlapping IOM band. Keep a context only if
        # the two aligned age ranges actually share years.
        if min(band[1], irange[1]) - max(band[0], irange[0]) <= 0:
            continue

        # Choose the (DGE slice, IOM table type) pair that shares the MOST
        # nutrients — that shared set is what a cross-lingual comparison question
        # can actually be asked about. Prefer a table type that has both a male
        # and a female chunk (so the context also supports a within-IOM
        # male/female contrast); RAG_K - 1 English partners.
        best_key = None
        best = None  # (dge_idx, table_type, shared_list)
        for dge_idx in dge_bands[band]:
            dnut = _dge_nutrients(chunks[dge_idx]["text"])
            for tt, sexmap in iom_ranges[irange].items():
                sample = next(iter(sexmap.values()))
                shared = dnut & _iom_nutrients(chunks[sample]["text"])
                # Rank by shared-nutrient count, then prefer a table with both
                # sexes, then table name for determinism.
                key = (len(shared), len(sexmap), tt)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (dge_idx, tt, sorted(shared))
        if best is None or not best[2]:
            continue
        dge_idx, tt, shared = best

        sexmap = iom_ranges[irange][tt]
        partners = [sexmap[s] for s in ("M", "F", "C") if s in sexmap][:n_en]
        members = [dge_idx] + partners
        lo, hi = band
        ilo, ihi = irange
        contexts.append(
            {
                "context_id": f"ctx_ref_{len(contexts):03d}",
                "context_type": "reference",
                "context_lang": "mixed",
                "dge_age_band": f"{lo}-{hi if hi < _OLD else '120+'}",
                "iom_age_band": f"{ilo}-{ihi if ihi < _OLD else '120+'}",
                "iom_table_type": tt,
                "shared_nutrients": shared,
                "question_langs": ["en", "de"],
                "chunks": [
                    _chunk_record(
                        chunks,
                        idx,
                        pair_sim=(
                            1.0 if idx == dge_idx
                            else round(float(all_vecs[idx] @ all_vecs[dge_idx]), 4)
                        ),
                    )
                    for idx in members
                ],
            }
        )
    return contexts


# ---------------------------------------------------------------------------
# Builder 2: condition-bridged contexts (persona-driven, English)
# ---------------------------------------------------------------------------
def _condition_pool(chunks, tag) -> list[int]:
    kws = CONDITION_SOURCE_KEYWORDS.get(tag, [])
    out = []
    for i in range(len(chunks)):
        m = chunks[i]["metadata"]
        if m.get("lang") != "en":
            continue
        s = (m.get("source") or "").lower()
        if any(kw in s for kw in kws):
            out.append(i)
    return out


def _central(pool, all_vecs) -> int:
    """Most representative chunk of a pool: highest summed cosine to the pool."""
    mat = all_vecs[np.array(pool)]
    centrality = mat @ mat.sum(axis=0)
    return pool[int(np.argmax(centrality))]


def build_condition_contexts(chunks, all_vecs, curated_personas) -> list[dict]:
    contexts: list[dict] = []
    for persona in curated_personas:
        conds = persona.get("conditions", [])
        pools = {c: _condition_pool(chunks, c) for c in conds}
        pools = {c: p for c, p in pools.items() if p}  # drop empty
        if not pools:
            print(f"  [condition] persona {persona['id']}: no chunks for {conds}, skipped")
            continue

        # One representative chunk per condition (spans the persona's conditions).
        members: list[int] = []
        for pool in pools.values():
            rep = _central(pool, all_vecs)
            if rep not in members:
                members.append(rep)

        union = sorted({i for pool in pools.values() for i in pool})
        # Pad up to RAG_K with the nearest neighbours of the first representative
        # within the persona's own chunk pool (keeps it topical to the persona).
        if len(members) < RAG_K:
            seed = members[0]
            sims = all_vecs[np.array(union)] @ all_vecs[seed]
            for j in np.argsort(-sims):
                cand = union[int(j)]
                if cand not in members:
                    members.append(cand)
                if len(members) == RAG_K:
                    break
        members = members[:RAG_K]

        contexts.append(
            {
                "context_id": f"ctx_cond_{len(contexts):03d}",
                "context_type": "condition",
                "context_lang": "en",
                "persona_id": persona["id"],
                "conditions": list(pools.keys()),
                "question_langs": ["en"],
                "chunks": [_chunk_record(chunks, idx) for idx in members],
            }
        )
    return contexts


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
def _profile_conditions(profile: str) -> list[str]:
    head = profile.split(". Their dietary habits")[0]
    head = head.replace("The user has the following health conditions:", "").strip()
    return [t.strip() for t in head.split(",") if t.strip()]


def build_personas_ngqa() -> list[dict]:
    """NGQA/NHANES profiles chosen coverage-greedily: repeatedly take the profile
    that covers the most still-uncovered condition tags, until every tag is hit.
    Count is data-driven (however many it takes), not a fixed cap."""
    seen: dict[str, str] = {}
    with open(_PROJECT_ROOT / NGQA_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            profile = (json.loads(line).get("context_variants") or {}).get("user_profile")
            if not profile:
                continue
            key = profile.split(". Their dietary habits")[0]
            seen.setdefault(key, profile)

    profiles = sorted(seen.values())
    cond_sets = {p: set(_profile_conditions(p)) for p in profiles}
    all_tags = set().union(*cond_sets.values()) if cond_sets else set()

    # Coverage-greedy with a redundancy target: each NHANES condition should
    # appear in at least TARGET profiles (not just once — a single-cover set is
    # only ~3 profiles since each carries ~5 conditions, too thin for a control
    # pool). TARGET mirrors the curated set's ~2-3 personas-per-condition density.
    # Count stays data-driven — however many profiles it takes to hit the target.
    target = 3
    counts = {t: 0 for t in all_tags}
    chosen: list[str] = []
    remaining = dict(cond_sets)
    while remaining:
        need = {t for t in all_tags if counts[t] < target}
        if not need:
            break
        best = max(remaining, key=lambda p: (len(remaining[p] & need), p))
        if not (remaining[best] & need):
            break  # nothing left contributes to an under-target tag
        chosen.append(best)
        for t in remaining[best]:
            counts[t] += 1
        del remaining[best]

    random.Random(SYNTH_SEED).shuffle(chosen)
    personas = [
        {"id": f"ngqa_{i:02d}", "origin": "ngqa", "text": text}
        for i, text in enumerate(chosen)
    ]
    print(
        f"{len(seen)} distinct NGQA profiles; {len(all_tags)} condition tags "
        f"each covered >={target}x by {len(personas)} sampled personas"
    )
    return personas


def load_curated_personas() -> list[dict]:
    with open(_PROJECT_ROOT / CURATED_PERSONAS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return [{"origin": "curated", **p} for p in raw]


# ---------------------------------------------------------------------------
def main():
    chunks = _load_chunks()
    vs = _load_index(len(chunks))
    _verify_alignment(vs, chunks)
    all_vecs = vs.index.reconstruct_n(0, vs.index.ntotal)

    curated = load_curated_personas()
    ngqa_personas = build_personas_ngqa()
    personas = ngqa_personas + curated

    ref_ctx = build_reference_contexts(chunks, all_vecs)
    cond_ctx = build_condition_contexts(chunks, all_vecs, curated)

    out_ref = _PROJECT_ROOT / CONTEXTS_REFERENCE_FILE
    out_cond = _PROJECT_ROOT / CONTEXTS_CONDITION_FILE
    out_per = _PROJECT_ROOT / PERSONAS_FILE
    for path, data in ((out_ref, ref_ctx), (out_cond, cond_ctx), (out_per, personas)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"wrote {out_ref} ({len(ref_ctx)} reference contexts)")
    print(f"wrote {out_cond} ({len(cond_ctx)} condition contexts)")
    print(f"wrote {out_per} ({len(personas)} personas: "
          f"{len(ngqa_personas)} NGQA + {len(curated)} curated)")


if __name__ == "__main__":
    main()
