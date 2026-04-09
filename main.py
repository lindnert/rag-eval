from dotenv import load_dotenv
load_dotenv()
import os
import asyncio

from rag_pipeline import run_rag_pipeline, run_rag_pipeline_batch_async
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


def evaluate_queries_batch_async(queries):
    return asyncio.run(run_rag_pipeline_batch_async(queries, batch_size=3))


if __name__ == "__main__":
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"
    os.environ["DEEPEVAL_MAX_RETRIES_OVERRIDE"] = "3"
    
    #query = "Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?"

    #result = evaluate_query(query)

    queries = [
    "Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?",
    "Wie kann ich meine Regeneration nach intensivem Training optimieren? Welche Lebensmittel und Timing sind dafür am wichtigsten?",
    "Ich habe starke Gelenkschmerzen und Entzündungen. Gibt es eine Ernährung, die mir helfen kann, diese zu reduzieren?",
    "Meine 8-jährige Tochter ist übergewichtig. Welche Ernährungsempfehlungen sind für Kinder mit Übergewicht geeignet?",
    "Ich bin 65 Jahre alt und möchte meine Knochendichte erhöhen und kognitiven Abbau verhindern. Welche Nährstoffe sind entscheidend?",
    "Ich bin Veganer und trainiere intensiv 6x pro Woche. Wie stelle ich sicher, dass ich genug Protein und alle essentiellen Aminosäuren bekomme?",
    "Nach meiner Gallenblasenoperation kann ich viele Lebensmittel nicht mehr essen. Welche Ernährungsstrategie hilft mir, wieder normal zu essen?",
    "Ich bin 45 Jahre alt, habe ADHS und Schlafprobleme. Kann die richtige Ernährung meine Symptome verbessern?",
    "Welche Lebensmittel helfen am besten gegen Migräne? Gibt es Trigger, die ich vermeiden sollte?"
    ]
    results = evaluate_queries_batch_async(queries)

    for result in results:
        for k, v in result.items():
            print(f"{k}: {v}")

