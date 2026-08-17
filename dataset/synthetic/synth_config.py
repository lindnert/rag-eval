"""
Configuration for the deepeval synthetic-dataset generation.

Everything is env-overridable so the SLURM script can steer a run without code
changes, mirroring evaluation/eval_config_llamacpp.py.

Design decisions (full rationale in dataset/synthetic/README.md):

- Two context families are built from richtlinien/all_chunks.json — the exact
  corpus the retriever serves — via deepeval's generate_goldens_from_contexts
  (NOT generate_goldens_from_docs, which would re-chunk the raw documents and
  decouple goldens from the RAG corpus). Each context holds RAG_K chunks:
    * reference (contexts_reference.json): cross-lingual DGE<->IOM reference
      tables aligned on the SAME life-stage, personalized by age/sex; questions
      generated in BOTH en and de (condition = <question_lang>Q_refC).
    * condition (contexts_condition.json): English guideline chunks spanning a
      curated persona's clinical conditions, personalized by condition;
      questions in en (condition = enQ_condC).
- Styling is persona-conditioned: lay / reference_lay passes template a persona
  (NGQA-derived or curated) into the scenario; the technical profile runs
  without one. One Synthesizer pass per (context, styling, language), since
  StylingConfig is static per instance.
- FiltrationConfig's threshold only triggers rewrite-retries in deepeval
  3.9.4; low scorers are KEPT after max_quality_retries. Hard filtering happens
  in validate_synthetic.py: dedupe -> input-quality cutoff (>= SYNTH_HARD_CUTOFF)
  -> completeness -> answerability -> faithfulness of expected_output against
  its own context (>= SYNTH_FAITHFULNESS_CUTOFF).
"""

import os

from deepeval.synthesizer.config import (
    EvolutionConfig,
    FiltrationConfig,
    StylingConfig,
)
from deepeval.synthesizer.types import Evolution

# ---------------------------------------------------------------------------
# Paths (relative to repo root; the SLURM script cd's there)
# ---------------------------------------------------------------------------
CHUNKS_FILE = os.getenv("SYNTH_CHUNKS_FILE", "richtlinien/all_chunks.json")
PERSONAS_FILE = os.getenv("SYNTH_PERSONAS_FILE", "dataset/synthetic/personas.json")
NGQA_FILE = os.getenv("SYNTH_NGQA_FILE", "dataset/NGQA/NGQA.jsonl")
OUTPUT_DIR = os.getenv("SYNTH_OUTPUT_DIR", "dataset/synthetic/generated")

# Path C (hybrid) inputs. Two context builders write two files:
#   reference  = cross-lingual DGE<->IOM/EFSA tables aligned on the SAME
#                life-stage (personalized by age/sex); questions in en+de.
#   condition  = persona-condition-bridged English guideline chunks
#                (personalized by clinical condition); questions in en.
# Curated personas are hand-authored (corpus-grounded) and merged with the
# NGQA-sampled ones into PERSONAS_FILE by build_contexts.
CURATED_PERSONAS_FILE = os.getenv(
    "SYNTH_CURATED_PERSONAS_FILE", "dataset/synthetic/personas_curated.json"
)
CONTEXTS_REFERENCE_FILE = os.getenv(
    "SYNTH_CONTEXTS_REFERENCE_FILE", "dataset/synthetic/contexts_reference.json"
)
CONTEXTS_CONDITION_FILE = os.getenv(
    "SYNTH_CONTEXTS_CONDITION_FILE", "dataset/synthetic/contexts_condition.json"
)

# ---------------------------------------------------------------------------
# Context construction (build_contexts.py — login node / locally)
# ---------------------------------------------------------------------------
# Pilot cap: truncates each context set (reference + condition) in
# generate_synthetic. Context counts are otherwise data-driven (adult DGE age
# bands / curated personas), so this is only for a quick smoke test.
SYNTH_MAX_CONTEXTS = int(os.getenv("SYNTH_MAX_CONTEXTS", "40"))
SYNTH_SEED = int(os.getenv("SYNTH_SEED", "42"))

# ---------------------------------------------------------------------------
# Generation (generate_synthetic.py — single SLURM node)
# ---------------------------------------------------------------------------
# Condition contexts generate this many goldens each; reference contexts use
# their per-context shared-nutrient count instead (see generate_synthetic).
SYNTH_MAX_GOLDENS_PER_CONTEXT = int(os.getenv("SYNTH_MAX_GOLDENS_PER_CONTEXT", "2"))
# Match llama-server --parallel (GEN_PARALLEL in slurm/run_synth.sh).
SYNTH_MAX_CONCURRENT = int(os.getenv("SYNTH_MAX_CONCURRENT", "3"))
# Human-readable generator tag stored in every record's metadata
# (slurm/run_synth.sh exports the actual GGUF file stem).
SYNTH_GENERATOR_TAG = os.getenv("SYNTH_GENERATOR_TAG", "llamacpp-local")

# Question generation samples at a mildly creative temperature — at 0.0 the
# generator produces near-identical phrasings across contexts. The critic and
# the validation pass run at 0.0 (see synth_llm.py).
SYNTH_TEMPERATURE = float(os.getenv("SYNTH_TEMPERATURE", "0.7"))

# ---------------------------------------------------------------------------
# Filtration (in-run) + validation (post-hoc)
# ---------------------------------------------------------------------------
# In-run threshold: below it deepeval asks the critic for feedback and
# rewrites the input (up to max_quality_retries), then keeps the result
# EITHER WAY. 0.6 with a 12B critic triggers rewrites for genuinely clumsy
# inputs without spending most of the budget on retries.
SYNTH_QUALITY_THRESHOLD = float(os.getenv("SYNTH_QUALITY_THRESHOLD", "0.8"))
SYNTH_MAX_QUALITY_RETRIES = int(os.getenv("SYNTH_MAX_QUALITY_RETRIES", "2"))
# Hard post-hoc cutoffs applied in validate_synthetic.py. Revisit after
# inspecting the score histogram the validation step prints.
SYNTH_HARD_CUTOFF = float(os.getenv("SYNTH_HARD_CUTOFF", "0.8"))
SYNTH_FAITHFULNESS_CUTOFF = float(os.getenv("SYNTH_FAITHFULNESS_CUTOFF", "1.0"))

# ---------------------------------------------------------------------------
# Evolutions
# ---------------------------------------------------------------------------
# One evolution step per input: with local models every extra rewrite step
# compounds drift away from the source context, which shows up later as
# unanswerable questions and unsupported expected_outputs.
SYNTH_NUM_EVOLUTIONS = int(os.getenv("SYNTH_NUM_EVOLUTIONS", "1"))

# Tuned for personalized, answerable questions (no unanswerable goldens is a
# hard requirement — other datasets already cover those):
# - IN_BREADTH and HYPOTHETICAL are excluded: both push the question beyond what
#   the context supports (IN_BREADTH broadens the topic; HYPOTHETICAL invents
#   "what if" scenarios the guidelines can't answer) — the main sources of
#   unanswerable goldens.
# - CONCRETIZING + CONSTRAINED are weighted up: they add specifics and
#   qualifiers (age, sex, condition), which is exactly the personalization the
#   thesis targets, and both stay grounded in the context.
# - MULTICONTEXT stays high: spanning the chunks is the point of both context
#   sets (compare DE/EN reference tables; bridge a persona's conditions).
# - COMPARATIVE suits the reference contexts (DE vs US, male vs female values).
EVOLUTION_DISTRIBUTION = {
    Evolution.MULTICONTEXT: 0.25,
    Evolution.CONCRETIZING: 0.25,
    Evolution.CONSTRAINED: 0.20,
    Evolution.COMPARATIVE: 0.15,
    Evolution.REASONING: 0.15,
}


def build_evolution_config() -> EvolutionConfig:
    return EvolutionConfig(
        num_evolutions=SYNTH_NUM_EVOLUTIONS,
        evolutions=dict(EVOLUTION_DISTRIBUTION),
    )


def build_filtration_config(critic_model) -> FiltrationConfig:
    return FiltrationConfig(
        synthetic_input_quality_threshold=SYNTH_QUALITY_THRESHOLD,
        max_quality_retries=SYNTH_MAX_QUALITY_RETRIES,
        critic_model=critic_model,
    )


# ---------------------------------------------------------------------------
# Styling — StylingConfig fields are freeform strings injected into deepeval's
# generation prompts: `scenario` = who is asking and why, `task` = what the
# target system does, `input_format` = shape of the question,
# `expected_output_format` = shape of the reference answer.
# ---------------------------------------------------------------------------
_TASK = {
    "en": (
        "Answer nutrition and diet questions grounded strictly in official "
        "nutrition guidelines and dietary reference values (WHO, NIH, ESPEN, "
        "EFSA, DGE, national dietary guidelines)."
    ),
    "de": (
        "Ernährungsfragen beantworten, strikt gestützt auf offizielle "
        "Ernährungsrichtlinien und Referenzwerte (z.B. DGE, EFSA, WHO)."
    ),
}

_EXPECTED_OUTPUT = {
    "en": (
        "A concise, factual answer of 1-4 sentences in English that is fully "
        "supported by the provided guideline context. Include concrete "
        "quantities, limits or food recommendations from the context where "
        "available. Do not add information that is not in the context."
    ),
    "de": (
        "Eine knappe, faktische Antwort in 1-4 Sätzen auf Deutsch, die "
        "vollständig durch den bereitgestellten Richtlinien-Kontext gedeckt "
        "ist, mit konkreten Mengenangaben oder Empfehlungen aus dem Kontext, "
        "wo vorhanden. Keine Informationen ergänzen, die nicht im Kontext "
        "stehen."
    ),
}

_LAY_SCENARIO = {
    "en": (
        "A user of a nutrition assistant chatbot with the following profile: "
        "{persona} "
        "They describe symptoms, diagnoses or everyday eating situations in "
        "plain, sometimes vague language and want practical dietary guidance "
        "that fits their health conditions and habits."
    ),
    "de": (
        "Ein:e deutschsprachige:r Nutzer:in eines Ernährungs-Chatbots mit "
        "folgendem Profil: {persona} "
        "Die Person beschreibt Beschwerden, Diagnosen oder Alltagssituationen "
        "in einfacher, teils vager Sprache und möchte praktische "
        "Ernährungsempfehlungen, die zu ihren Erkrankungen und Gewohnheiten "
        "passen."
    ),
}

_LAY_INPUT_FORMAT = {
    "en": (
        "A single first-person question in everyday English, 1-2 sentences, "
        "no medical jargon, no references to 'the context' or 'the document'. "
        "May be slightly ambiguous or underspecified, the way real patients "
        "ask. Must specify at least one of: the person's age or life-stage, "
        "sex, or a health condition/dietary context from the profile. A bare "
        "question with no personal context is not acceptable."
    ),
    "de": (
        "Eine einzelne Frage in der Ich-Form auf Deutsch in Alltagssprache, "
        "1-2 Sätze, ohne Fachjargon und ohne Verweis auf 'den Kontext' oder "
        "'das Dokument'. Darf leicht mehrdeutig oder unterspezifiziert sein, "
        "wie echte Patient:innen fragen. Muss mindestens eine der folgenden "
        "Angaben enthalten: Alter bzw. Lebensphase, Geschlecht oder eine "
        "Erkrankung/Ernährungssituation aus dem Profil. Eine allgemeine Frage "
        "ohne persönlichen Bezug ist nicht zulässig."
    ),
}

_REF_LAY_SCENARIO = {
    "en": (
        "A user of a nutrition assistant chatbot: {persona}. They want to know "
        "their own recommended daily intake of a vitamin or mineral and ask in "
        "plain, everyday language."
    ),
    "de": (
        "Ein:e deutschsprachige:r Nutzer:in eines Ernährungs-Chatbots: "
        "{persona}. Die Person möchte ihren eigenen empfohlenen Tagesbedarf an "
        "einem Vitamin oder Mineralstoff wissen und fragt in einfacher "
        "Alltagssprache."
    ),
}

# Reference tables personalize ONLY on age + sex (they contain no disease or
# goal content). The question must state both so the answer's age band is not a
# surprise, and must stay on nutrient reference amounts so it stays answerable.
_REF_LAY_INPUT_FORMAT = {
    "en": (
        "A single first-person question in everyday English, 1-2 sentences, no "
        "medical jargon, no references to 'the context' or 'the document'. The "
        "question MUST state the person's age (or life-stage) AND sex, and ask "
        "about the recommended daily amount of ONE specific vitamin or mineral. "
        "Do NOT mention any disease, symptom, weight-loss/weight-gain, "
        "muscle-building or other goal — ask only about a nutrient reference "
        "amount."
    ),
    "de": (
        "Eine einzelne Frage in der Ich-Form auf Deutsch in Alltagssprache, "
        "1-2 Sätze, ohne Fachjargon und ohne Verweis auf 'den Kontext' oder "
        "'das Dokument'. Die Frage MUSS Alter (bzw. Lebensphase) UND Geschlecht "
        "der Person nennen und nach der empfohlenen Tagesmenge EINES konkreten "
        "Vitamins oder Mineralstoffs fragen. KEINE Erkrankung, kein Symptom, "
        "kein Abnehm-/Zunehm- oder Muskelaufbau-Ziel erwähnen — nur nach einem "
        "Nährstoff-Referenzwert fragen."
    ),
}

_TECH_SCENARIO = {
    "en": (
        "Dietitians, clinicians and nutrition-science students querying a "
        "guideline knowledge base for exact recommendations, dietary "
        "reference values and clinical nutrition protocols."
    ),
    "de": (
        "Diätassistent:innen, Ärzt:innen und Studierende der "
        "Ernährungswissenschaft, die eine Richtlinien-Wissensdatenbank nach "
        "exakten Empfehlungen, Referenzwerten und klinischen "
        "Ernährungsprotokollen abfragen."
    ),
}

_TECH_INPUT_FORMAT = {
    "en": (
        "A single precise question in professional English using correct "
        "terminology (e.g. nutrient reference values, g/kg body weight, "
        "specific conditions and life-stage groups). No references to 'the "
        "context' or 'the document'. Must anchor the question to a specific "
        "population — at least one of: a life-stage/age group, sex, or a "
        "clinical condition — rather than asking for a value in the abstract."
    ),
    "de": (
        "Eine einzelne präzise Frage auf Deutsch in Fachsprache (z.B. "
        "Referenzwerte, g/kg Körpergewicht, konkrete Erkrankungen und "
        "Altersgruppen). Kein Verweis auf 'den Kontext' oder 'das Dokument'. "
        "Muss die Frage an eine konkrete Population binden — mindestens eine "
        "Angabe zu Altersgruppe/Lebensphase, Geschlecht oder klinischem "
        "Zustand — statt einen Wert abstrakt zu erfragen."
    ),
}


def build_styling(profile: str, qlang: str, persona_text: str | None = None) -> StylingConfig:
    if qlang not in ("en", "de"):
        raise ValueError(f"unsupported question language: {qlang!r}")
    if profile == "lay":
        if not persona_text:
            raise ValueError("lay profile requires persona_text")
        return StylingConfig(
            scenario=_LAY_SCENARIO[qlang].format(persona=persona_text.strip()),
            task=_TASK[qlang],
            input_format=_LAY_INPUT_FORMAT[qlang],
            expected_output_format=_EXPECTED_OUTPUT[qlang],
        )
    if profile == "reference_lay":
        if not persona_text:
            raise ValueError("reference_lay profile requires persona_text")
        return StylingConfig(
            scenario=_REF_LAY_SCENARIO[qlang].format(persona=persona_text.strip()),
            task=_TASK[qlang],
            input_format=_REF_LAY_INPUT_FORMAT[qlang],
            expected_output_format=_EXPECTED_OUTPUT[qlang],
        )
    if profile == "technical":
        return StylingConfig(
            scenario=_TECH_SCENARIO[qlang],
            task=_TASK[qlang],
            input_format=_TECH_INPUT_FORMAT[qlang],
            expected_output_format=_EXPECTED_OUTPUT[qlang],
        )
    raise ValueError(f"unknown styling profile: {profile!r}")
