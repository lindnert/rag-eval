"""
Generate synthetic goldens with the deepeval Synthesizer from the two Path-C
context sets (see dataset/synthetic/build_contexts.py). Requires the generation
endpoint to be up (LLAMACPP_GEN_BASE_URL) — SLURM llama-server or the external
Ollama node.

Two context types, each with its own pass structure:

  reference  (cross-lingual, personalized by demographic): each context is
             processed in BOTH question languages. Contexts alternate between a
             technical (clinician) framing and a lay framing conditioned on an
             NGQA persona. max_goldens per context = its number of shared
             nutrients (the comparison surface), so a rich table yields more
             questions than a thin one.

  condition  (English, personalized by clinical condition): each context is
             bound to a curated persona; one English lay pass per context.

One Synthesizer instance per pass (StylingConfig is static per instance). Each
pass writes goldens_<pass>.json and is skipped if it already exists, so a
requeued/re-run job resumes instead of regenerating.
"""

import time
import json
from pathlib import Path

from deepeval.synthesizer import Synthesizer

from dataset.synthetic.synth_config import (
    CONTEXTS_CONDITION_FILE,
    CONTEXTS_REFERENCE_FILE,
    OUTPUT_DIR,
    PERSONAS_FILE,
    SYNTH_GENERATOR_TAG,
    SYNTH_MAX_CONCURRENT,
    SYNTH_MAX_CONTEXTS,
    SYNTH_MAX_GOLDENS_PER_CONTEXT,
    build_evolution_config,
    build_filtration_config,
    build_styling,
)
from dataset.synthetic.synth_llm import build_critic, build_generator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_band(band: str) -> tuple[int, int]:
    """'51-65' / '65-120+' -> (51, 65) / (65, 120)."""
    lo, hi = band.split("-", 1)
    return int(lo), int(hi.rstrip("+"))


def _demographic_persona(ctx, sex: str, qlang: str) -> str:
    """Build a plain-language age+sex persona from a reference context's own
    DGE/IOM bands, so the question can only ask what the tables can answer.

    The age is taken from the intersection of the two bands so it is valid for
    both the DGE and the IOM reference values in the context.
    """
    dge_lo, dge_hi = _parse_band(ctx["dge_age_band"])
    iom_lo, iom_hi = _parse_band(ctx["iom_age_band"])
    lo, hi = max(dge_lo, iom_lo), min(dge_hi, iom_hi)
    age = lo + min(5, (hi - lo) // 2)
    if qlang == "de":
        return f"eine {age}-jährige Frau" if sex == "female" else f"ein {age}-jähriger Mann"
    return f"a {age}-year-old woman" if sex == "female" else f"a {age}-year-old man"


def _load_json(rel_path: str):
    with open(_PROJECT_ROOT / rel_path, encoding="utf-8") as f:
        return json.load(f)


def _context_chunks_meta(ctx):
    out = []
    for c in ctx["chunks"]:
        rec = {"chunk_id": c["chunk_id"], "source": c["source"], "lang": c["lang"]}
        if "pair_sim" in c:
            rec["pair_sim"] = c["pair_sim"]
        out.append(rec)
    return out


def _golden_records(goldens, ctx_by_id, *, pass_name, profile, qlang, persona_id, context_type):
    cell = "refC" if context_type == "reference" else "condC"
    records = []
    for k, g in enumerate(goldens):
        ctx = ctx_by_id.get(g.source_file or "")
        meta = g.additional_metadata or {}
        dm = {
            "condition": f"{qlang}Q_{cell}",
            "question_lang": qlang,
            "context_lang": ctx.get("context_lang") if ctx else None,
            "context_type": context_type,
            "styling_profile": profile,
            "persona_id": persona_id,
            "context_id": g.source_file,
            "context_chunks": _context_chunks_meta(ctx) if ctx else None,
            "evolutions": meta.get("evolutions"),
            "synthetic_input_quality": meta.get("synthetic_input_quality"),
            "generator_model": SYNTH_GENERATOR_TAG,
            "generation_pass": pass_name,
        }
        if ctx and context_type == "reference":
            dm["shared_nutrients"] = ctx.get("shared_nutrients")
            dm["dge_age_band"] = ctx.get("dge_age_band")
            dm["iom_age_band"] = ctx.get("iom_age_band")
        elif ctx and context_type == "condition":
            dm["conditions"] = ctx.get("conditions")
        records.append(
            {
                "id": f"synth_{pass_name}_{k:03d}",
                "source_dataset": "synthetic_guidelines",
                "query": g.input,
                "reference_answer": g.expected_output,
                "contexts": list(g.context or []),
                "dataset_metadata": dm,
            }
        )
    return records


def run_pass(pass_name, ctx, styling, ctx_by_id, *, profile, qlang, max_goldens,
             context_type, persona_id=None):
    out_file = _PROJECT_ROOT / OUTPUT_DIR / f"goldens_{pass_name}.json"
    if out_file.exists():
        print(f"[{pass_name}] exists, skipping ({out_file})", flush=True)
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
        contexts=[[c["text"] for c in ctx["chunks"]]],
        include_expected_output=True,
        max_goldens_per_context=max_goldens,
        source_files=[ctx["context_id"]],
        _send_data=False,
    )
    records = _golden_records(
        goldens, ctx_by_id,
        pass_name=pass_name, profile=profile, qlang=qlang,
        persona_id=persona_id, context_type=context_type,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(
        f"[{pass_name}] {len(records)} goldens (max {max_goldens}) "
        f"in {time.perf_counter() - t0:.0f}s -> {out_file}",
        flush=True,
    )


def main():
    # SYNTH_MAX_CONTEXTS truncates each set for a quick pilot.
    ref_ctxs = _load_json(CONTEXTS_REFERENCE_FILE)[:SYNTH_MAX_CONTEXTS]
    cond_ctxs = _load_json(CONTEXTS_CONDITION_FILE)[:SYNTH_MAX_CONTEXTS]
    personas = _load_json(PERSONAS_FILE)
    persona_by_id = {p["id"]: p for p in personas}
    ctx_by_id = {c["context_id"]: c for c in ref_ctxs + cond_ctxs}
    (_PROJECT_ROOT / OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(
        f"{len(ref_ctxs)} reference contexts (cross-lingual), "
        f"{len(cond_ctxs)} condition contexts (en)",
        flush=True,
    )

    # --- Reference: alternate technical / lay(demographic), both languages ---
    # Reference tables personalize on age + sex only, so the lay framing uses a
    # demographic persona derived from the context's OWN band (no NGQA condition
    # personas — their diseases/goals are unanswerable from a DRI table and were
    # the source of the mismatched/unanswerable reference goldens).
    for i, ctx in enumerate(ref_ctxs):
        cid = ctx["context_id"]
        max_goldens = len(ctx.get("shared_nutrients") or []) or SYNTH_MAX_GOLDENS_PER_CONTEXT
        sex = "female" if (i // 2) % 2 == 0 else "male"
        for qlang in ctx.get("question_langs", ["en", "de"]):
            if i % 2 == 0:
                run_pass(
                    f"ref_technical_{cid}_{qlang}", ctx,
                    build_styling("technical", qlang), ctx_by_id,
                    profile="technical", qlang=qlang, max_goldens=max_goldens,
                    context_type="reference",
                )
            else:
                persona_text = _demographic_persona(ctx, sex, qlang)
                run_pass(
                    f"ref_lay_{cid}_{qlang}", ctx,
                    build_styling("reference_lay", qlang, persona_text), ctx_by_id,
                    profile="reference_lay", qlang=qlang,
                    persona_id=f"demo_{cid}_{sex[0]}",
                    max_goldens=max_goldens, context_type="reference",
                )

    # --- Condition: one English lay pass per context, bound persona ---
    for ctx in cond_ctxs:
        cid = ctx["context_id"]
        persona = persona_by_id.get(ctx["persona_id"])
        if persona is None:
            print(f"[{cid}] persona {ctx['persona_id']} not found, skipping", flush=True)
            continue
        for qlang in ctx.get("question_langs", ["en"]):
            run_pass(
                f"cond_{cid}_{qlang}", ctx,
                build_styling("lay", qlang, persona["text"]), ctx_by_id,
                profile="lay", qlang=qlang, persona_id=persona["id"],
                max_goldens=SYNTH_MAX_GOLDENS_PER_CONTEXT, context_type="condition",
            )

    print("all passes done", flush=True)


if __name__ == "__main__":
    main()
