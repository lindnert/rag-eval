# Synthetic guideline dataset (deepeval Synthesizer)

Generates ~150 synthetic QA goldens from the `richtlinien/` nutrition
guidelines (CSV excluded — it is not part of `all_chunks.json`) for the
cross-lingual RAG evaluation. Everything runs against a llama.cpp/Ollama
OpenAI-compatible endpoint; no cloud APIs.

## Pipeline

| Step | Where | Command |
|---|---|---|
| 1. Build context sets + personas | login node (no GPU) | `python -m dataset.synthetic.build_contexts` |
| 2a. Generate + validate (SLURM) | one 8 GB GPU SLURM node | `sbatch slurm/run_synth.sh` |
| 2b. Generate + validate (Ollama node) | local machine / login node | `bash slurm/run_synth_ollama.sh` |

Step 1 writes two context sets — `contexts_reference.json` (cross-lingual) and
`contexts_condition.json` (condition-personalized) — plus `personas.json`
(NGQA + curated). See the design section for what each is.

2a and 2b are interchangeable backends for the same Python. 2b targets the
external 12 GB Ollama node (HTTP-only: URL + Bearer key from `.env`, no
shell access), so the script runs wherever the repo lives and only the API
calls leave the machine. Model default `gemma4:e4b`; escalate with
`SYNTH_OLLAMA_MODEL=gemma4:12b`.

Pilot first: `SYNTH_MAX_CONTEXTS=6 sbatch slurm/run_synth.sh` (or the same
prefix with `bash slurm/run_synth_ollama.sh`), inspect the output dir, then
full run.

Outputs land in `dataset/synthetic/generated/` (SLURM) or
`dataset/synthetic/generated_<model>/` (Ollama node — per-model dirs so runs
never collide) so the dataset is version-controlled alongside the code (unlike
`results/`, which is gitignored): `synthetic_dataset.json` (final),
`validation_report.json` (attrition + score histograms), `goldens_<pass>.json`
(raw, resumable per pass).

## Design decisions

**Contexts, not docs.** We use `generate_goldens_from_contexts` over groups
built from `richtlinien/all_chunks.json` — the exact corpus the retriever
serves — instead of `generate_goldens_from_docs`, which would re-chunk the
raw files and decouple the goldens from the RAG corpus. Every context holds
`RAG_K` chunks (the retriever's k), so a synthetic context mirrors what the
RAG system actually sees at inference.

**Hybrid design: two context sets (personalization first, cross-lingual
second).** The corpus is English-rich but German-only in its reference tables,
so the two goals are split:

- *Reference contexts* (`contexts_reference.json`, `condition` cell
  `*Q_refC`): 1 German DGE reference-value table + the English IOM tables for
  the **same life-stage** (male + female of one DRI table type). DGE and IOM
  use different age cutoffs, so bands are aligned by **index from the oldest
  end** — the adult bands line up exactly (`build_contexts._parse_dge_age` /
  `_parse_iom_group`). The DGE slice + IOM table type are chosen to **maximize
  shared nutrients**, stored per context as `shared_nutrients`; that count is
  the per-context `max_goldens`. Personalized by demographic (age/sex);
  generated in **both** languages, alternating a technical (clinician) framing
  and a lay framing conditioned on a **demographic persona derived from the
  context's own band** ("a 58-year-old woman"). Reference tables carry no
  disease/goal content, so the lay framing asks only about a nutrient reference
  amount and always names age+sex — otherwise the answer invents a band or the
  question drifts to something the table cannot answer.
- *Condition contexts* (`contexts_condition.json`, cell `*Q_condC`): `RAG_K`
  English guideline chunks selected for a curated persona's clinical
  conditions (`CONDITION_SOURCE_KEYWORDS`); the question bridges them.
  Personalized by condition; one **English** lay pass, bound to that persona.

**Downstream RAG runs:** run each synthetic query ONCE, with the system
prompt matched to the question language (en question → en prompt, de → de
prompt). `rag_pipeline.py` loads the synthetic set via
`load_synthetic(lang=RAG_LANG)`, so the German goldens ride the `RAG_LANG=de`
run and the English ones the `en` run. Question language is the controlled
factor here; the dual en/de-prompt runs remain for the existing all-English
datasets, where the
prompt is the only language manipulation. A small prompt-language ablation
(de questions × en prompt) can be added later if needed.

**Personas — two sources.** `personas.json` merges (a) NHANES-derived
`user_profile` strings from `dataset/NGQA/NGQA.jsonl`, sampled
**coverage-greedily** so every distinct condition tag appears in ≥3 profiles
(`origin: ngqa`), and (b) hand-authored, corpus-grounded personas in
`personas_curated.json` (`origin: curated`, each tagged with the guideline
topics it maps to). NGQA reuse is deliberate — sharing the user population
means dataset-level performance gaps can't be blamed on different populations;
the curated personas cover the deep clinical content NGQA never touches (liver,
cancer, IBD, pancreatitis, …). Curated personas drive the condition contexts
(bound 1:1). Reference contexts do **not** use NGQA/condition personas — their
lay framing uses an age+sex demographic derived from the band, because a
condition persona's diseases/goals are unanswerable from a DRI table. Personas
steer style/topic only — answers are grounded in the guideline chunks, so there
is no leakage. One Synthesizer pass per (context, styling, language), because
`StylingConfig` is static per instance; the `technical` profile
(dietitian/clinician phrasing) runs without a persona on alternate reference
contexts.

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
3b. answerability — drop goldens whose reference answer hedges that the context
   lacks the requested info (en+de markers). A faithful answer can still be
   unanswerable ("the context does not offer…"), and faithfulness keeps those;
   no unanswerable goldens is a hard requirement, so this stage removes them,
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
Escalation path if pilot quality disappoints (or SLURM is down):
`slurm/run_synth_ollama.sh` runs the same Python against the external 12 GB
Ollama node (gemma4:e4b default, `SYNTH_OLLAMA_MODEL=gemma4:12b` to
escalate). It caps `SYNTH_MAX_CONCURRENT=2` — Ollama queues the excess, but
a 12B in 12 GB has little KV headroom and the box isn't ours. Note the
generator model then differs between backends; the per-record
`generator_model` metadata and per-model output dirs keep runs separable.

## Language / provenance tracking

Every record carries in `dataset_metadata`:
`condition` (`enQ_refC` / `deQ_refC` / `enQ_condC`), `question_lang`,
`context_lang`, `context_type` (`reference` / `condition`), `styling_profile`,
`persona_id`, `context_id`, `context_chunks` (chunk_id, source, per-chunk
lang, pair_sim where applicable), plus type-specific fields
(`shared_nutrients` / `dge_age_band` / `iom_age_band` for reference,
`conditions` for condition), `evolutions`, `synthetic_input_quality`,
`faithfulness_of_reference`, `generator_model`, `generation_pass`. Any later
analysis slices by one groupby on `condition` or `context_type`.

## Knobs (env)

Counts are data-driven (reference contexts = adult DGE age bands; condition
contexts = curated personas; reference `max_goldens` = shared-nutrient count),
so the tunable knobs in `synth_config.py` are few:
`SYNTH_MAX_CONTEXTS` (truncates each set for a pilot),
`SYNTH_MAX_GOLDENS_PER_CONTEXT=2` (condition contexts),
`SYNTH_TEMPERATURE=0.7`, `SYNTH_QUALITY_THRESHOLD=0.6`,
`SYNTH_HARD_CUTOFF=0.6`, `SYNTH_FAITHFULNESS_CUTOFF=0.8`, `SYNTH_SEED=42`.

Budget: 5 reference contexts × ~5 goldens × 2 languages (≈50) + 26 condition
contexts × 2 goldens × 1 language (≈52) ≈ 100 raw goldens → ~70 after
validation. Pilot with `SYNTH_MAX_CONTEXTS=2` (→ 2 reference + 2 condition).
