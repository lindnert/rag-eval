from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
import time
from datetime import datetime

from rag.utils import run_rag_pipeline_batch_async, OLLAMA_CONTEXT_LENGTH

# outdated function, not used in current pipeline but kept for reference
""" def evaluate_query(query):
    sample = run_rag_pipeline(query)

    ragas_scores = run_ragas(sample)
    deepeval_scores = run_deepeval(sample)
    custom_scores = run_custom(sample)

    return {
        **sample,
        **ragas_scores,
        **deepeval_scores,
        **custom_scores
    } """


def run_queries_batch_async(queries):
    return asyncio.run(run_rag_pipeline_batch_async(queries, batch_size=10))


if __name__ == "__main__":

    queries = [
    "Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?",
    "Wie kann ich meine Regeneration nach intensivem Training optimieren? Welche Lebensmittel und Timing sind dafür am wichtigsten?",
    "Ich habe starke Gelenkschmerzen und Entzündungen. Gibt es eine Ernährung, die mir helfen kann, diese zu reduzieren?",
    "Meine 8-jährige Tochter ist übergewichtig. Welche Ernährungsempfehlungen sind für Kinder mit Übergewicht geeignet?",
    "Ich bin 65 Jahre alt und möchte meine Knochendichte erhöhen und kognitiven Abbau verhindern. Welche Nährstoffe sind entscheidend?",
    "Ich bin Veganer und trainiere intensiv 6x pro Woche. Wie stelle ich sicher, dass ich genug Protein und alle essentiellen Aminosäuren bekomme?",
    "Nach meiner Gallenblasenoperation kann ich viele Lebensmittel nicht mehr essen. Welche Ernährungsstrategie hilft mir, wieder normal zu essen?",
    "Ich bin 45 Jahre alt, habe ADHS und Schlafprobleme. Kann die richtige Ernährung meine Symptome verbessern?",
    "Welche Lebensmittel helfen am besten gegen Migräne? Gibt es Trigger, die ich vermeiden sollte?",
    "Mit 22 Jahren und 60 kg fällt es mir schwer zuzunehmen. Welche Ernährungsstrategie hilft mir beim gesunden Gewichtaufbau?",
    "Als 35-jähriger Büroangestellter mit wenig Bewegung frage ich mich, wie ich meine Ernährung langfristig optimieren kann.",
    "Regelmäßiges Marathontraining gehört zu meinem Alltag (28 Jahre). Welche Lebensmittel verbessern gezielt meine Ausdauerleistung?",
    "Aufgrund von Bluthochdruck (50 Jahre) möchte ich meine Ernährung umstellen – worauf sollte ich besonders achten?",
    "Ich wiege 95 kg bei 40 Jahren und möchte nachhaltig abnehmen. Welche Rolle spielen Makronährstoffe dabei konkret?",
    "Seit einigen Jahren ernähre ich mich vegetarisch (19 Jahre). Wie kann ich mögliche Nährstoffdefizite vermeiden?",
    "Häufige Verdauungsprobleme beeinträchtigen meinen Alltag (33 Jahre). Welche Ernährungsweise könnte helfen?",
    "Mit 70 Jahren habe ich oft wenig Appetit. Wie kann ich dennoch eine ausreichende Nährstoffversorgung sicherstellen?",
    "Trotz Pflege habe ich mit Akne zu kämpfen (26 Jahre). Welche Ernährungsfaktoren könnten mein Hautbild beeinflussen?",
    "Ich trainiere regelmäßig im Fitnessstudio (31 Jahre, 80 kg). Wie bestimme ich meinen optimalen Proteinbedarf?",
    "Nach der Diagnose Diabetes Typ 2 (45 Jahre) möchte ich meine Ernährung anpassen – insbesondere bei Kohlenhydraten. Was ist sinnvoll?",
    "Schichtarbeit (38 Jahre) bringt meinen Essrhythmus durcheinander. Wie kann ich meine Mahlzeiten besser strukturieren?",
    "Während des Studiums fällt es mir schwer, konzentriert zu bleiben (24 Jahre). Welche Rolle spielt Ernährung dabei?",
    "Erhöhte Cholesterinwerte (55 Jahre) machen mir Sorgen. Welche Fette sollte ich bevorzugen oder vermeiden?",
    "Dauerhafter Stress (30 Jahre) wirkt sich auf mein Wohlbefinden aus. Gibt es Lebensmittel, die mich unterstützen können?",
    "Aufgrund einer Laktoseintoleranz (27 Jahre) suche ich nach Alternativen, um meinen Kalziumbedarf zu decken.",
    "Mit einer Schilddrüsenerkrankung (42 Jahre) frage ich mich, welche Ernährung förderlich ist.",
    "Ich praktiziere intermittierendes Fasten (36 Jahre). Wie sollte ich meine Mahlzeiten innerhalb des Essfensters gestalten?",
    "Heißhungerattacken treten bei mir regelmäßig auf (48 Jahre). Welche Ernährungsstrategien helfen dagegen?",
    "Als leistungsorientierter Fußballspieler (21 Jahre) interessiert mich, wie ich meine Regeneration durch Ernährung verbessern kann.",
    "Osteoporose wurde bei mir diagnostiziert (60 Jahre). Welche Nährstoffe sind jetzt besonders wichtig?",
    "Mit einer Glutenunverträglichkeit (34 Jahre) suche ich nach Möglichkeiten für eine ausgewogene Ernährung.",
    "Mein Ziel ist es, den Körperfettanteil zu reduzieren (29 Jahre). Welche Ernährungsansätze sind dafür effektiv?",
    "Wiederkehrende Migräneanfälle (41 Jahre) belasten mich. Welche Rolle spielen Ernährung und mögliche Trigger?",
    "Ich möchte gezielt meine Darmflora verbessern (37 Jahre). Welche Lebensmittel sind dafür besonders geeignet?",
    "Im Alltag fühle ich mich oft energielos (23 Jahre). Kann meine Ernährung daran schuld sein?",
    "Zur Verbesserung meiner Herzgesundheit (52 Jahre) möchte ich meine Essgewohnheiten anpassen – was ist empfehlenswert?",
    "Wechseljahresbeschwerden machen mir zu schaffen (46 Jahre). Welche Ernährung kann unterstützend wirken?",
    "In Vorbereitung auf einen Triathlon (32 Jahre) suche ich nach einer optimalen Ernährungsstrategie für Training und Wettkampf.",
    "Mit 65 Jahren möchte ich mein Immunsystem stärken. Welche Nährstoffe und Lebensmittel spielen dabei eine zentrale Rolle?",
    ]

    print(f"\n{'='*80}")
    print(f"Starting RAG evaluation pipeline")
    print(f"Total queries: {len(queries)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ollama context length: {OLLAMA_CONTEXT_LENGTH} tokens")
    print(f"{'='*80}\n", flush=True)

    pipeline_start = time.time()
    results = run_queries_batch_async(queries)
    pipeline_time = time.time() - pipeline_start

    print(f"\n{'='*80}")
    print(f"Printing results...")
    print(f"{'='*80}\n", flush=True)

    for idx, result in enumerate(results, 1):
        print(f"\n[Result {idx}/{len(results)}]")
        for k, v in result.items():
            print(f"{k}: {v}")
        print(f"-" * 80, flush=True)

    output_file = "rag_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {output_file}")

    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"Pipeline time: {pipeline_time:.1f}s ({pipeline_time/60:.1f}m)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

