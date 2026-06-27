import re
import unicodedata

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.embeddings import PASSAGE_PREFIX, QUERY_PREFIX

# Tokenizer for the lexical (BM25) channel. We want to match on content
# keywords ("calcium"), not numbers or measurement units: digits ("1000",
# "1,5") and units ("mg", "kcal") appear in almost every nutrition chunk, so
# matching on them would pull in irrelevant documents. So we keep only
# letter-runs ([^\W\d_]+ drops digits and underscores) and then filter out a
# small unit stopword set. No stemming: it's language-specific and we have
# mixed German/English. NFKC + lowercase fold case and unicode width so the
# query and the corpus tokenise identically.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Measurement units / non-topical tokens excluded from lexical matching.
# (After NFKC the micro sign µ becomes Greek μ, hence "μg".)
_STOPWORDS = frozenset({
    "mg", "g", "kg", "µg", "μg", "ug", "mcg", "ng", "ml", "l", "dl", "cl",
    "kcal", "kj", "iu", "ie",
    # period words from dosage expressions ("mg/Tag", "per day")
    "tag", "tage", "day", "days",
})


def bm25_tokenize(text):
    tokens = _WORD_RE.findall(unicodedata.normalize("NFKC", text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _strip_prefix(text, prefix):
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


class HybridRetriever:
    """Weighted dense + lexical (BM25) retrieval over one shared chunk set.

    Final ranking score, higher = better:

        fused = alpha * cosine + (1 - alpha) * (bm25 / max_bm25)

    - `cosine` is the dense similarity (absolute, ~[0, 1]); the FAISS index is
      built L2-normalised with MAX_INNER_PRODUCT, so inner product == cosine.
    - the BM25 term is scaled by its per-query max so 0 means "no lexical
      overlap" and 1 means "best lexical match for this query".
    - `alpha = 1.0` reproduces pure dense retrieval (previous behaviour);
      `alpha = 0.0` is pure lexical.

    Rationale: the dense channel is the cross-lingual/semantic workhorse
    (bge-m3 maps an English query near a German passage). BM25 adds a
    discriminative term-overlap signal — a bland boilerplate chunk that drifts
    into the dense top-k usually contains none of a query's specific terms, so
    its BM25 score is ~0 and fusion demotes it. The trade-off is that genuinely
    relevant paraphrase / cross-language chunks also score low on BM25, so
    `alpha` must be tuned rather than assumed.
    """

    def __init__(self, vectorstore, alpha):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.embeddings = getattr(vectorstore, "embeddings", None) or vectorstore.embedding_function

        n = vectorstore.index.ntotal
        # Stored vectors are L2-normalised (index built with normalize_L2=True),
        # so reconstructing them and dotting with a normalised query yields
        # cosine similarity for the whole corpus in one matmul.
        self._doc_vectors = vectorstore.index.reconstruct_n(0, n)  # (n, d) float32

        # Recover documents in FAISS-position order so doc vectors, BM25 rows
        # and metadata share a single index space (0..n-1).
        id_map = vectorstore.index_to_docstore_id
        docs = [vectorstore.docstore.search(id_map[i]) for i in range(n)]
        self._texts = [_strip_prefix(d.page_content, PASSAGE_PREFIX) for d in docs]
        self._metadatas = [d.metadata for d in docs]

        self._bm25 = BM25Okapi([bm25_tokenize(t) for t in self._texts])

    def search_with_score(self, query, k, alpha=None):
        """Return [(Document, fused_score), ...] sorted by fused score desc."""
        alpha = self.alpha if alpha is None else alpha

        q = np.asarray(self.embeddings.embed_query(QUERY_PREFIX + query), dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        dense = self._doc_vectors @ q  # cosine per doc, absolute in [-1, 1]

        sparse = np.asarray(self._bm25.get_scores(bm25_tokenize(query)), dtype=np.float32)
        smax = float(sparse.max()) if sparse.size else 0.0
        sparse_norm = sparse / smax if smax > 0 else np.zeros_like(sparse)

        fused = alpha * dense + (1.0 - alpha) * sparse_norm

        top = np.argsort(-fused)[:k]
        return [
            (Document(page_content=self._texts[i], metadata=self._metadatas[i]), float(fused[i]))
            for i in top
        ]
