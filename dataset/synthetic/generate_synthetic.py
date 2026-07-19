"""
Generate synthetic goldens with the deepeval Synthesizer on a single SLURM
node (see slurm/run_synth.sh). Requires the llama-server gen endpoint to be
up (LLAMACPP_GEN_BASE_URL).

Passes (paired design — the SAME contexts are used for every question
language, so per-language results are directly comparable):

    for qlang in SYNTH_QUESTION_LANGS:            # en, de
        technical_<qlang>          over contexts[0::2]
        lay_<qlang>_p<NN>          over contexts[1::2], round-robin per persona

One Synthesizer instance per pass because StylingConfig is static per
instance (persona conditioning requires it). Each pass writes
results/synthetic/goldens_<pass>.json and is skipped if that file already
exists, so a requeued job resumes instead of regenerating.
"""

import json
import os
import time
from pathlib import Path

from deepeval.synthesizer import Synthesizer

from dataset.synthetic.synth_config import (
    CONTEXTS_FILE,
    OUTPUT_DIR,
    PERSONAS_FILE,
    SYNTH_GENERATOR_TAG,
    SYNTH_MAX_CONCURRENT,
    SYNTH_MAX_GOLDENS_PER_CONTEXT,
    SYNTH_QUESTION_LANGS,
    build_evolution_config,
    build_filtration_config,
    build_styling,
)
from dataset.synthetic.synth_llm import build_critic, build_generator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(rel_path: str):
    with open(_PROJECT_ROOT / rel_path, encoding="utf-8") as f:
        return json.load(f)


def _golden_records(goldens, ctx_by_id, *, pass_name, profile, qlang, persona_id):
    records = []
    for k, g in enumerate(goldens):
        ctx = ctx_by_id.get(g.source_file or "", None)
        meta = g.additional_metadata or {}
        records.append(
            {
                "id": f"synth_{pass_name}_{k:03d}",
                "source_dataset": "synthetic_guidelines",
                "query": g.input,
                "reference_answer": g.expected_output,
                "contexts": list(g.context or []),
                "dataset_metadata": {
                    "condition": f"{qlang}Q_mixedC",
                    "question_lang": qlang,
                    "context_lang": "mixed",
                    "styling_profile": profile,
                    "persona_id": persona_id,
                    "context_id": g.source_file,
                    "context_chunks": (
                        [
                            {
                                "chunk_id": c["chunk_id"],
                                "source": c["source"],
                                "lang": c["lang"],
                                "pair_sim": c["pair_sim"],
                            }
                            for c in ctx["chunks"]
                        ]
                        if ctx
                        else None
                    ),
                    "evolutions": meta.get("evolutions"),
                    "synthetic_input_quality": meta.get("synthetic_input_quality"),
                    "generator_model": SYNTH_GENERATOR_TAG,
                    "generation_pass": pass_name,
                },
            }
        )
    return records


def run_pass(pass_name, ctxs, styling, ctx_by_id, *, profile, qlang, persona_id=None):
    out_file = _PROJECT_ROOT / OUTPUT_DIR / f"goldens_{pass_name}.json"
    if out_file.exists():
        print(f"[{pass_name}] exists, skipping ({out_file})", flush=True)
        return
    if not ctxs:
        print(f"[{pass_name}] no contexts assigned, skipping", flush=True)
        return

    t0 = time.perf_counter()
    synthesizer = Synthesizer(
        model=build_generator(),
        async_mode=True,
        max_concurrent=SYNTH_MAX_CONCURRENT,
        filtration_config=build_filtration_config(build_critic()),
        evolution_config=build_evolution_config(),
        styling_config=styling,
    )
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=[[c["text"] for c in ctx["chunks"]] for ctx in ctxs],
        include_expected_output=True,
        max_goldens_per_context=SYNTH_MAX_GOLDENS_PER_CONTEXT,
        source_files=[ctx["context_id"] for ctx in ctxs],
        _send_data=False,
    )
    records = _golden_records(
        goldens, ctx_by_id,
        pass_name=pass_name, profile=profile, qlang=qlang, persona_id=persona_id,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(
        f"[{pass_name}] {len(records)} goldens from {len(ctxs)} contexts "
        f"in {time.perf_counter() - t0:.0f}s -> {out_file}",
        flush=True,
    )


def main():
    contexts = _load_json(CONTEXTS_FILE)
    personas = _load_json(PERSONAS_FILE)
    ctx_by_id = {c["context_id"]: c for c in contexts}
    (_PROJECT_ROOT / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tech_ctxs = contexts[0::2]
    lay_ctxs = contexts[1::2]
    n_p = len(personas)
    print(
        f"{len(contexts)} contexts ({len(tech_ctxs)} technical / {len(lay_ctxs)} lay), "
        f"{n_p} personas, question langs: {SYNTH_QUESTION_LANGS}",
        flush=True,
    )

    for qlang in SYNTH_QUESTION_LANGS:
        run_pass(
            f"technical_{qlang}",
            tech_ctxs,
            build_styling("technical", qlang),
            ctx_by_id,
            profile="technical",
            qlang=qlang,
        )
        for j, persona in enumerate(personas):
            run_pass(
                f"lay_{qlang}_p{j:02d}",
                lay_ctxs[j::n_p],
                build_styling("lay", qlang, persona["text"]),
                ctx_by_id,
                profile="lay",
                qlang=qlang,
                persona_id=persona["id"],
            )

    print("all passes done", flush=True)


if __name__ == "__main__":
    main()
