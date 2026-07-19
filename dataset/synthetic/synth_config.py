"""
Configuration for the deepeval synthetic-dataset generation.

Everything is env-overridable so the SLURM script can steer a run without code
changes, mirroring evaluation/eval_config_llamacpp.py.

Design decisions (full rationale in dataset/synthetic/README.md):

- Contexts are mixed-language groups (1 German DGE chunk + its bge-m3 nearest
  English chunks) built from richtlinien/all_chunks.json — the exact corpus
  the retriever serves — NOT via deepeval's generate_goldens_from_docs, which
  would re-chunk the raw documents and decouple goldens from the RAG corpus.
- Questions are generated in both languages over the SAME contexts (paired
  design): condition = <question_lang>Q_mixedC, tracked per golden.
- Styling is persona-conditioned: lay passes template an NHANES-derived NGQA
  user profile into the scenario; one Synthesizer pass per (persona, lang).
- FiltrationConfig's threshold only triggers rewrite-retries in deepeval
  3.9.4; low scorers are KEPT after max_quality_retries. Hard filtering
  happens in validate_synthetic.py using
  golden.additional_metadata["synthetic_input_quality"] plus a faithfulness
  check of expected_output against its own context.
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
CONTEXTS_FILE = os.getenv("SYNTH_CONTEXTS_FILE", "dataset/synthetic/contexts_mixed.json")
PERSONAS_FILE = os.getenv("SYNTH_PERSONAS_FILE", "dataset/synthetic/personas.json")
NGQA_FILE = os.getenv("SYNTH_NGQA_FILE", "dataset/NGQA/NGQA.jsonl")
OUTPUT_DIR = os.getenv("SYNTH_OUTPUT_DIR", "results/synthetic")

# ---------------------------------------------------------------------------
# Context construction (build_contexts.py — login node / locally)
# ---------------------------------------------------------------------------
# Number of mixed context groups. 40 contexts x 2 goldens x 2 question
# languages = 160 raw goldens -> ~150 after validation attrition.
SYNTH_MAX_CONTEXTS = int(os.getenv("SYNTH_MAX_CONTEXTS", "40"))
# English partner chunks retrieved per German chunk (context size = 1 + this).
SYNTH_EN_PARTNERS = int(os.getenv("SYNTH_EN_PARTNERS", "2"))
# Minimum cosine similarity for a de->en pairing; German chunks whose best
# English partners score below this are skipped (forced pairings produce
# incoherent MULTICONTEXT questions).
SYNTH_MIN_PAIR_SIM = float(os.getenv("SYNTH_MIN_PAIR_SIM", "0.45"))
# Cap on how many contexts a single English chunk may appear in.
SYNTH_MAX_EN_REUSE = int(os.getenv("SYNTH_MAX_EN_REUSE", "2"))
SYNTH_NUM_PERSONAS = int(os.getenv("SYNTH_NUM_PERSONAS", "10"))
SYNTH_SEED = int(os.getenv("SYNTH_SEED", "42"))

# ---------------------------------------------------------------------------
# Generation (generate_synthetic.py — single SLURM node)
# ---------------------------------------------------------------------------
SYNTH_MAX_GOLDENS_PER_CONTEXT = int(os.getenv("SYNTH_MAX_GOLDENS_PER_CONTEXT", "2"))
# Match llama-server --parallel (GEN_PARALLEL in slurm/run_synth.sh).
SYNTH_MAX_CONCURRENT = int(os.getenv("SYNTH_MAX_CONCURRENT", "3"))
# Comma-separated question languages; each language re-uses the same contexts
# (paired design) so per-language scores are directly comparable.
SYNTH_QUESTION_LANGS = [
    s.strip() for s in os.getenv("SYNTH_QUESTION_LANGS", "en,de").split(",") if s.strip()
]
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
SYNTH_QUALITY_THRESHOLD = float(os.getenv("SYNTH_QUALITY_THRESHOLD", "0.6"))
SYNTH_MAX_QUALITY_RETRIES = int(os.getenv("SYNTH_MAX_QUALITY_RETRIES", "2"))
# Hard post-hoc cutoffs applied in validate_synthetic.py. Revisit after
# inspecting the score histogram the validation step prints.
SYNTH_HARD_CUTOFF = float(os.getenv("SYNTH_HARD_CUTOFF", "0.6"))
SYNTH_FAITHFULNESS_CUTOFF = float(os.getenv("SYNTH_FAITHFULNESS_CUTOFF", "0.8"))

# ---------------------------------------------------------------------------
# Evolutions
# ---------------------------------------------------------------------------
# One evolution step per input: with local models every extra rewrite step
# compounds drift away from the source context, which shows up later as
# unanswerable questions and unsupported expected_outputs.
SYNTH_NUM_EVOLUTIONS = int(os.getenv("SYNTH_NUM_EVOLUTIONS", "1"))

# IN_BREADTH is deliberately absent: it broadens questions beyond the given
# context, which breaks the grounding guarantee the RAG evaluation relies on.
# MULTICONTEXT is weighted highest because forcing the question to span the
# German AND English chunks is exactly the cross-lingual behaviour under test.
EVOLUTION_DISTRIBUTION = {
    Evolution.MULTICONTEXT: 0.30,
    Evolution.REASONING: 0.20,
    Evolution.CONCRETIZING: 0.15,
    Evolution.COMPARATIVE: 0.15,
    Evolution.CONSTRAINED: 0.10,
    Evolution.HYPOTHETICAL: 0.10,
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
        "ask."
    ),
    "de": (
        "Eine einzelne Frage in der Ich-Form auf Deutsch in Alltagssprache, "
        "1-2 Sätze, ohne Fachjargon und ohne Verweis auf 'den Kontext' oder "
        "'das Dokument'. Darf leicht mehrdeutig oder unterspezifiziert sein, "
        "wie echte Patient:innen fragen."
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
        "context' or 'the document'."
    ),
    "de": (
        "Eine einzelne präzise Frage auf Deutsch in Fachsprache (z.B. "
        "Referenzwerte, g/kg Körpergewicht, konkrete Erkrankungen und "
        "Altersgruppen). Kein Verweis auf 'den Kontext' oder 'das Dokument'."
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
    if profile == "technical":
        return StylingConfig(
            scenario=_TECH_SCENARIO[qlang],
            task=_TASK[qlang],
            input_format=_TECH_INPUT_FORMAT[qlang],
            expected_output_format=_EXPECTED_OUTPUT[qlang],
        )
    raise ValueError(f"unknown styling profile: {profile!r}")
