"""
Build mixed-language context groups + persona pool for synthetic generation.

Runs WITHOUT a GPU model: it queries the existing bge-m3 FAISS index by
vector (reconstructed from the index itself), so no embedding calls are made.
Run it on the cluster login node where richtlinien/faiss_index_bge_m3_cosine
exists (or locally with FAISS_INDEX_DIR pointing at a copy):

    python -m dataset.synthetic.build_contexts

Outputs (committed inputs for the SLURM generation job):
    dataset/synthetic/contexts_mixed.json
    dataset/synthetic/personas.json

Each context group = 1 German chunk (DGE-Referenzwerte) + its
SYNTH_EN_PARTNERS nearest English chunks by bge-m3 cosine similarity.
bge-m3 is multilingual, so de->en nearest neighbours are topically aligned
(DGE reference values pair with EFSA/DRV/IOM table chunks).
"""

import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from dataset.synthetic.synth_config import (
    CHUNKS_FILE,
    CONTEXTS_FILE,
    NGQA_FILE,
    PERSONAS_FILE,
    SYNTH_EN_PARTNERS,
    SYNTH_MAX_CONTEXTS,
    SYNTH_MAX_EN_REUSE,
    SYNTH_MIN_PAIR_SIM,
    SYNTH_NUM_PERSONAS,
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


def build_contexts(chunks, vs) -> list[dict]:
    rng = random.Random(SYNTH_SEED)
    de_ids = [i for i, c in enumerate(chunks) if c["metadata"].get("lang") == "de"]
    if not de_ids:
        raise SystemExit("ERROR: no German chunks found in the chunk file")
    print(f"{len(de_ids)} German chunks, targeting {SYNTH_MAX_CONTEXTS} contexts")
    rng.shuffle(de_ids)

    # Exact de->en scoring over the full vector matrix. An ANN top-k search is
    # useless here: the German DGE table chunks are near-duplicates of each
    # other, so a German chunk's top-100 neighbours are almost all German.
    # Vectors are unit-normalized at index build, so dot product == cosine
    # regardless of the index's metric type.
    all_vecs = vs.index.reconstruct_n(0, vs.index.ntotal)
    en_ids = np.array(
        [i for i, c in enumerate(chunks) if c["metadata"].get("lang") == "en"]
    )
    en_mat = all_vecs[en_ids]

    en_use_count: dict[int, int] = {}
    contexts: list[dict] = []
    skipped_low_sim = 0

    for de_id in de_ids:
        if len(contexts) >= SYNTH_MAX_CONTEXTS:
            break
        sims = en_mat @ all_vecs[de_id]
        order = np.argsort(-sims)

        partners: list[tuple[int, float]] = []
        for j in order:
            pos = int(en_ids[j])
            sim = float(sims[j])
            if sim < SYNTH_MIN_PAIR_SIM:
                break  # sorted descending; nothing better follows
            if en_use_count.get(pos, 0) >= SYNTH_MAX_EN_REUSE:
                continue
            partners.append((pos, sim))
            if len(partners) == SYNTH_EN_PARTNERS:
                break

        if len(partners) < SYNTH_EN_PARTNERS:
            skipped_low_sim += 1
            continue

        for pos, _ in partners:
            en_use_count[pos] = en_use_count.get(pos, 0) + 1

        members = [(de_id, 1.0)] + partners
        contexts.append(
            {
                "context_id": f"ctx_{len(contexts):03d}",
                "context_lang": "mixed",
                "chunks": [
                    {
                        "chunk_id": pos,
                        "text": chunks[pos]["text"],
                        "source": chunks[pos]["metadata"]["source"],
                        "folder": chunks[pos]["metadata"]["folder"],
                        "lang": chunks[pos]["metadata"]["lang"],
                        "pair_sim": round(sim, 4),
                    }
                    for pos, sim in members
                ],
            }
        )

    print(
        f"built {len(contexts)} contexts "
        f"({skipped_low_sim} German chunks skipped: partners below "
        f"sim={SYNTH_MIN_PAIR_SIM} or reuse-capped)"
    )
    if contexts:
        all_sims = [c["chunks"][1]["pair_sim"] for c in contexts]
        print(
            f"best-partner similarity: min={min(all_sims):.3f} "
            f"median={sorted(all_sims)[len(all_sims)//2]:.3f} max={max(all_sims):.3f}"
        )
    return contexts


def build_personas() -> list[dict]:
    """NHANES-derived user profiles sampled from NGQA, deduplicated by their
    health-condition set so the pool spans distinct clinical situations."""
    rng = random.Random(SYNTH_SEED)
    seen: dict[str, str] = {}
    with open(_PROJECT_ROOT / NGQA_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            profile = (row.get("context_variants") or {}).get("user_profile")
            if not profile:
                continue
            conditions = profile.split(". Their dietary habits")[0]
            if conditions not in seen:
                seen[conditions] = profile
    pool = sorted(seen.values())
    rng.shuffle(pool)
    personas = [
        {"id": f"persona_{i:02d}", "text": text}
        for i, text in enumerate(pool[:SYNTH_NUM_PERSONAS])
    ]
    print(f"{len(seen)} distinct condition profiles in NGQA, sampled {len(personas)}")
    return personas


def main():
    chunks = _load_chunks()
    vs = _load_index(len(chunks))
    _verify_alignment(vs, chunks)

    contexts = build_contexts(chunks, vs)
    personas = build_personas()

    out_ctx = _PROJECT_ROOT / CONTEXTS_FILE
    out_per = _PROJECT_ROOT / PERSONAS_FILE
    with open(out_ctx, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)
    with open(out_per, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_ctx} ({len(contexts)} contexts)")
    print(f"wrote {out_per} ({len(personas)} personas)")


if __name__ == "__main__":
    main()
