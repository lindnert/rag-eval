from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import run_rag_pipeline
from evaluation.ragas_eval import run_ragas
from evaluation.deepeval_eval import run_deepeval
from evaluation.custom_eval import run_custom


def evaluate_query(query):
    sample = run_rag_pipeline(query)

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
    query = "Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?"

    result = evaluate_query(query)

    for k, v in result.items():
        print(f"{k}: {v}")