"""Project-wide canonical constants shared across the generation, dataset, and
evaluation layers. Kept dependency-free so any module can import it without
pulling in heavy generation/retrieval dependencies.
"""

# Canonical abstention string. The generation model is instructed to emit this
# verbatim when the context is insufficient, and the RAG pipeline substitutes
# it whenever a generation produces no answer text (the thinking-looped regen —
# see rag.utils._finalize_answer). A single fixed sentence makes rejection a
# clean, countable event downstream instead of a fuzzy family of "I don't know"
# phrasings. The out-of-domain MEDQA probe also uses it as its gold answer, so
# a well-behaved abstention scores as correct (see dataset/MEDQA/loader.py).
REJECTION_ANSWER = (
    "Die bereitgestellten Kontextinformationen enthalten keine ausreichenden "
    "Informationen, um diese Frage zu beantworten."
)
