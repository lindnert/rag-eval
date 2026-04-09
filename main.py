from dotenv import load_dotenv
load_dotenv()
import os

from rag_pipeline import run_rag_pipeline
from retrieval.utils import build_retriever
from evaluation.ragas_eval import run_ragas
from evaluation.deepeval_eval import run_deepeval
from evaluation.custom_eval import run_custom


def evaluate_query(query, retriever):
    sample = run_rag_pipeline(query, retriever)

    ragas_scores = run_ragas(sample)
    deepeval_scores = run_deepeval(sample)
    custom_scores = run_custom(sample)

    return {
        **sample,
        **ragas_scores,
        **deepeval_scores,
        **custom_scores
    }


if __name__ == "__main__":
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"
    os.environ["DEEPEVAL_MAX_RETRIES_OVERRIDE"] = "3"
    
    query = "Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?"
    retriever = build_retriever()
    result = evaluate_query(query, retriever)

    for k, v in result.items():
        print(f"{k}: {v}")