"""Project-wide canonical constants shared across the generation, dataset, and
evaluation layers. Kept dependency-free so any module can import it without
pulling in heavy generation/retrieval dependencies.
"""

import os

# Default run language, selected once from the environment. A single run now
# answers each query in its own language (German query → German prompt, English
# query → English), selected per query in rag.utils; RAG_LANG only seeds the
# fallback for untagged queries and the default-language singletons. Defaults to
# English to preserve prior behaviour.
RAG_LANG = os.getenv("RAG_LANG", "en").lower()

# Canonical abstention string, per language. The generation model is instructed
# to emit this verbatim when the context is insufficient, and the RAG pipeline
# substitutes it whenever a generation produces no answer text (the
# thinking-looped regen — see rag.utils._finalize_answer). A single fixed
# sentence makes rejection a clean, countable event downstream instead of a
# fuzzy family of "I don't know" phrasings. The out-of-domain MEDQA probe also
# uses it as its gold answer, so a well-behaved abstention scores as correct
# (see dataset/MEDQA/loader.py). Keyed by query language so the per-query prompt
# selection in rag.utils substitutes the matching-language rejection.
REJECTION_ANSWERS = {
    "en": (
        "The provided context does not contain sufficient information to answer "
        "this question."
    ),
    "de": (
        "Die bereitgestellten Kontextinformationen enthalten keine ausreichenden "
        "Informationen, um diese Frage zu beantworten."
    ),
}

# Default-language abstention string (used by the English-only MEDQA gold and as
# a fallback); per-query substitution goes through REJECTION_ANSWERS directly.
REJECTION_ANSWER = REJECTION_ANSWERS[RAG_LANG]
