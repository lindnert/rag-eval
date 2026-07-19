# Synthetic guideline dataset (deepeval Synthesizer)

Generates ~150 synthetic QA goldens from the `richtlinien/` nutrition
guidelines (CSV excluded — it is not part of `all_chunks.json`) for the
cross-lingual RAG evaluation. Everything runs against a llama.cpp/Ollama
OpenAI-compatible endpoint; no cloud APIs.

## Pipeline

| Step | Where | Command |
|---|---|---|
| 1. Build contexts + personas | login node (no GPU) | `python -m dataset.synthetic.build_contexts` |
| 2. Generate + validate | one 8 GB GPU SLURM node | `sbatch slurm/run_synth.sh` |

Pilot first: `SYNTH_MAX_CONTEXTS=6 sbatch slurm/run_synth.sh`, inspect
`results/synthetic/`, then full run.

Outputs land in `results/synthetic/`:
`synthetic_dataset.json` (final), `validation_report.json` (attrition +
score histograms), `goldens_<pass>.json` (raw, resumable per pass).

## Design decisions

**Contexts, not docs.** We use `generate_goldens_from_contexts` over groups
built from `richtlinien/all_chunks.json` — the exact corpus the retriever
serves — instead of `generate_goldens_from_docs`, which would re-chunk the
raw files and decouple the goldens from the RAG corpus.

**Mixed-language contexts (cross-lingual design).** Each context = 1 German
DGE-Referenzwerte chunk + its 2 nearest English chunks by bge-m3 cosine
similarity (the index is multilingual, and the German chunks are
reference-value content that overlaps EFSA/DRV/IOM tables). Pairings below
`SYNTH_MIN_PAIR_SIM=0.45` are skipped — forced pairings produce incoherent
MULTICONTEXT questions. Questions are generated in **both** languages over
the **same** contexts (paired design), so the two `condition` cells
(`enQ_mixedC`, `deQ_mixedC`) are directly comparable.

**Downstream RAG runs:** run each synthetic query ONCE, with the system
prompt matched to the question language (en question → en prompt, de → de
prompt). Question language is the controlled factor here; the dual
en/de-prompt runs remain for the existing all-English datasets, where the
prompt is the only language manipulation. A small prompt-language ablation
(de questions × en prompt) can be added later if needed.

**Personas from NGQA.** Lay-user styling is conditioned on NHANES-derived
`user_profile` strings sampled from `dataset/NGQA/NGQA.jsonl` (deduplicated
by health-condition set) and injected verbatim into the `scenario`. Personas
steer style and topic only — answers are grounded in the guideline chunks,
so there is no leakage from NGQA. Reuse is deliberate: profiles are
empirically grounded (real NHANES respondents, unlike invented personas),
and sharing the user population with NGQA means dataset-level performance
differences cannot be attributed to different user populations.
`personas.json` is a generated file — hand-curating the sample (e.g.
preferring conditions the guideline corpus covers: kidney, heart, diabetes,
hypertension) before the SLURM run is fine. One Synthesizer pass per
(persona, language), because `StylingConfig` is static per instance. The
`technical` profile (dietitian/clinician phrasing) runs without personas.
Split: contexts `[0::2]` technical, `[1::2]` lay.

**EvolutionConfig.** `num_evolutions=1` — with local models every extra
rewrite compounds drift from the source context. Distribution: MULTICONTEXT
0.30 (spanning the de+en chunks is exactly the behaviour under test),
REASONING 0.20, CONCRETIZING 0.15, COMPARATIVE 0.15, CONSTRAINED 0.10,
HYPOTHETICAL 0.10. **IN_BREADTH is excluded**: it broadens questions beyond
the given context and breaks the grounding guarantee.

**FiltrationConfig — what it actually does.** In deepeval 3.9.4 the
`synthetic_input_quality_threshold` only decides whether the critic's
feedback triggers a rewrite (up to `max_quality_retries`); the golden is
**kept either way**, with the score stored in
`additional_metadata["synthetic_input_quality"]`. Settings: threshold 0.6,
retries 2, critic = same model at temperature 0.

**Validation (`validate_synthetic.py`) does the real filtering:**
1. dedupe near-identical inputs,
2. hard cutoff `synthetic_input_quality >= 0.6` (revisit after checking the
   histogram in `validation_report.json`),
3. completeness (non-empty query/answer/context),
4. `FaithfulnessMetric(reference_answer vs own context) >= 0.8` — the critic
   scores input clarity only, never factual correctness, so this closes that
   gap. Faithfulness scores are cached in `goldens_all_scored.json`;
   changing cutoffs and re-running is free.

Manual spot-check of the final ~150 samples is still recommended (and easy
at this size) — cite it as the human-verification step in the thesis.

**Model.** Largest model that fits the 8 GB SLURM GPUs (default
`gemma-4-E4B-it` Q4; override `LLAMACPP_GEN_REPO`/`LLAMACPP_GEN_FILE`).
Generation is template-scaffolded, so 8B-class quality suffices when
combined with the validation pass. Deliberately a *different* model from
the 4B system under test — avoids self-preference bias. Generator samples
at `temperature 0.7` for question variety; critic and validation run at 0.
Escalation path if pilot quality disappoints: the Python only needs
`LLAMACPP_GEN_BASE_URL`, so it runs unchanged against an Ollama/llama-server
`/v1` endpoint on the external 12 GB node (branch
`test/new-node-larger-models`) — there, cap parallelism
(`OLLAMA_NUM_PARALLEL=2`, `SYNTH_MAX_CONCURRENT=2`): a 12B in 12 GB has
little KV headroom.

## Language / provenance tracking

Every record carries in `dataset_metadata`:
`condition` (`enQ_mixedC` / `deQ_mixedC`), `question_lang`, `context_lang`,
`styling_profile`, `persona_id`, `context_id`, `context_chunks` (chunk_id,
source file, per-chunk lang, pair similarity), `evolutions`,
`synthetic_input_quality`, `faithfulness_of_reference`, `generator_model`,
`generation_pass`. Any later analysis slices by one groupby on `condition`
(or per-chunk `lang` for retrieval hit-rate across languages).

Pure-language cells (enQ/enC etc.) can be added later by building en-only or
de-only context files and extending `SYNTH_QUESTION_LANGS` /
`generate_synthetic.py` — the metadata schema already covers them.

## Knobs (env)

All in `synth_config.py`; the important ones:
`SYNTH_MAX_CONTEXTS=40`, `SYNTH_EN_PARTNERS=2`, `SYNTH_MIN_PAIR_SIM=0.45`,
`SYNTH_NUM_PERSONAS=10`, `SYNTH_MAX_GOLDENS_PER_CONTEXT=2`,
`SYNTH_QUESTION_LANGS=en,de`, `SYNTH_TEMPERATURE=0.7`,
`SYNTH_QUALITY_THRESHOLD=0.6`, `SYNTH_HARD_CUTOFF=0.6`,
`SYNTH_FAITHFULNESS_CUTOFF=0.8`, `SYNTH_SEED=42`.

Budget: 40 contexts × 2 goldens × (1 technical + 1 lay pass) × 2 languages
= 160 raw goldens → ~150 after validation. Roughly 1,500 LLM calls total
(~2–4 h on one 8 GB node at `--parallel 3`).
