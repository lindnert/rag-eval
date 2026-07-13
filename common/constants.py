"""Project-wide canonical constants shared across the generation, dataset, and
evaluation layers. Kept dependency-free so any module can import it without
pulling in heavy generation/retrieval dependencies.
"""

import os

# Run language, selected once from the environment. A single `export RAG_LANG=de`
# flips the abstention string here and the prompt bundle in rag.utils together,
# so the generation, dataset, and evaluation layers all agree on the language of
# a run. Defaults to English to preserve prior behaviour. German runs answer the
# (English) dataset queries in German — only the model-facing text changes.
RAG_LANG = os.getenv("RAG_LANG", "en").lower()

# Canonical abstention string, per language. The generation model is instructed
# to emit this verbatim when the context is insufficient, and the RAG pipeline
# substitutes it whenever a generation produces no answer text (the
# thinking-looped regen — see rag.utils._finalize_answer). A single fixed
# sentence makes rejection a clean, countable event downstream instead of a
# fuzzy family of "I don't know" phrasings. The out-of-domain MEDQA probe also
# uses it as its gold answer, so a well-behaved abstention scores as correct
# (see dataset/MEDQA/loader.py).
_REJECTION_ANSWERS = {
    "en": (
        "The provided context does not contain sufficient information to answer "
        "this question."
    ),
    "de": (
        "Die bereitgestellten Kontextinformationen enthalten keine ausreichenden "
        "Informationen, um diese Frage zu beantworten."
    ),
}

REJECTION_ANSWER = _REJECTION_ANSWERS[RAG_LANG]
