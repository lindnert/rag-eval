import re
import unicodedata

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.embeddings import PASSAGE_PREFIX, QUERY_PREFIX

# Tokenizer for the lexical (BM25) channel. We want to match on content
# keywords ("calcium"), not on tokens that carry no topical signal. We keep
# only letter-runs ([^\W\d_]+ drops digits and underscores, so "1000"/"1,5"
# never become tokens) and then drop two stopword layers (see below). No
# stemming: it's language-specific and we have mixed German/English. NFKC +
# lowercase fold case and unicode width so the query and the corpus tokenise
# identically.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize_tokens(text):
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text or "").lower())


# Layer 1 — measurement units / dosage words. Domain-specific: these are
# frequent enough in a nutrition corpus to pull in irrelevant chunks.
# (After NFKC the micro sign µ becomes Greek μ, hence "μg".)
_UNIT_STOPWORDS = frozenset({
    "mg", "g", "kg", "µg", "μg", "ug", "mcg", "ng", "ml", "l", "dl", "cl",
    "kcal", "kj", "iu", "ie",
    # period words from dosage expressions ("mg/Tag", "per day")
    "tag", "tage", "täglich", "day", "days", "daily", "per", "pro", "je",
})


# Layer 2 — English + German function words ("and", "der", ...) with no
# topical meaning. BM25's IDF already down-weights them, but rank_bm25 floors
# the negative IDF of very-common terms to a small positive value, so they
# still add uniform noise and inflate document length. Vendored from NLTK's
# english + german stopword lists, already run through _normalize_tokens (NFKC
# + lowercase + letter-runs) so they line up with corpus tokens and the code
# needs no NLTK data download at runtime. Regenerate with:
#   sorted({t for w in stopwords.words("english") + stopwords.words("german")
#           for t in _normalize_tokens(w)})
_LANGUAGE_STOPWORDS = frozenset({
    'a', 'aber', 'about', 'above', 'after', 'again', 'against', 'ain',
    'all', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am',
    'an', 'and', 'ander', 'andere', 'anderem', 'anderen', 'anderer',
    'anderes', 'anderm', 'andern', 'anderr', 'anders', 'any', 'are', 'aren',
    'as', 'at', 'auch', 'auf', 'aus', 'be', 'because', 'been', 'before',
    'bei', 'being', 'below', 'between', 'bin', 'bis', 'bist', 'both', 'but',
    'by', 'can', 'couldn', 'd', 'da', 'damit', 'dann', 'das', 'dass',
    'dasselbe', 'dazu', 'daß', 'dein', 'deine', 'deinem', 'deinen',
    'deiner', 'deines', 'dem', 'demselben', 'den', 'denn', 'denselben',
    'der', 'derer', 'derselbe', 'derselben', 'des', 'desselben', 'dessen',
    'dich', 'did', 'didn', 'die', 'dies', 'diese', 'dieselbe', 'dieselben',
    'diesem', 'diesen', 'dieser', 'dieses', 'dir', 'do', 'doch', 'does',
    'doesn', 'doing', 'don', 'dort', 'down', 'du', 'durch', 'during',
    'each', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig',
    'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er',
    'es', 'etwas', 'euch', 'euer', 'eure', 'eurem', 'euren', 'eurer',
    'eures', 'few', 'for', 'from', 'further', 'für', 'gegen', 'gewesen',
    'hab', 'habe', 'haben', 'had', 'hadn', 'has', 'hasn', 'hat', 'hatte',
    'hatten', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers',
    'herself', 'hier', 'him', 'himself', 'hin', 'hinter', 'his', 'how', 'i',
    'ich', 'if', 'ihm', 'ihn', 'ihnen', 'ihr', 'ihre', 'ihrem', 'ihren',
    'ihrer', 'ihres', 'im', 'in', 'indem', 'ins', 'into', 'is', 'isn',
    'ist', 'it', 'its', 'itself', 'jede', 'jedem', 'jeden', 'jeder',
    'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'just',
    'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines',
    'können', 'könnte', 'll', 'm', 'ma', 'machen', 'man', 'manche',
    'manchem', 'manchen', 'mancher', 'manches', 'me', 'mein', 'meine',
    'meinem', 'meinen', 'meiner', 'meines', 'mich', 'mightn', 'mir', 'mit',
    'more', 'most', 'muss', 'musste', 'mustn', 'my', 'myself', 'nach',
    'needn', 'nicht', 'nichts', 'no', 'noch', 'nor', 'not', 'now', 'nun',
    'nur', 'o', 'ob', 'oder', 'of', 'off', 'ohne', 'on', 'once', 'only',
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're',
    's', 'same', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner',
    'seines', 'selbst', 'shan', 'she', 'should', 'shouldn', 'sich', 'sie',
    'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches',
    'soll', 'sollte', 'some', 'sondern', 'sonst', 'such', 't', 'than',
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'through', 'to', 'too', 'um', 'und',
    'under', 'uns', 'unser', 'unsere', 'unserem', 'unseren', 'unseres',
    'unter', 'until', 'up', 've', 'very', 'viel', 'vom', 'von', 'vor',
    'war', 'waren', 'warst', 'was', 'wasn', 'we', 'weg', 'weil', 'weiter',
    'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde',
    'werden', 'were', 'weren', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst',
    'with', 'wo', 'wollen', 'wollte', 'won', 'wouldn', 'während', 'würde',
    'würden', 'y', 'you', 'your', 'yours', 'yourself', 'yourselves', 'zu',
    'zum', 'zur', 'zwar', 'zwischen', 'über',
})

_STOPWORDS = _UNIT_STOPWORDS | _LANGUAGE_STOPWORDS


def bm25_tokenize(text):
    return [t for t in _normalize_tokens(text) if t not in _STOPWORDS]


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
