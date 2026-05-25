from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import json
import random
import time
from datetime import datetime

from common.json_io import dump as dump_json
from rag.utils import run_rag_pipeline_async
from rag.llm_config import LLAMACPP_RAG_CONCURRENCY, LLAMACPP_RAG_MODEL

from dataset.NGQA.loader import load_ngqa, to_metadata as ngqa_to_metadata

if __name__ == "__main__":

    #queries = [
    #"Ich bin 29 Jahre alt, 71kg schwer und möchte Muskeln aufbauen. Wie sollte ich mich ernähren? Welche Mikro- und Makronährstoffe sollte ich einnehmen und wieviel?",
    #"Wie kann ich meine Regeneration nach intensivem Training optimieren? Welche Lebensmittel und Timing sind dafür am wichtigsten?",
    #"Ich habe starke Gelenkschmerzen und Entzündungen. Gibt es eine Ernährung, die mir helfen kann, diese zu reduzieren?",
    #"Meine 8-jährige Tochter ist übergewichtig. Welche Ernährungsempfehlungen sind für Kinder mit Übergewicht geeignet?",
    #"Ich bin 65 Jahre alt und möchte meine Knochendichte erhöhen und kognitiven Abbau verhindern. Welche Nährstoffe sind entscheidend?",
    #"Ich bin Veganer und trainiere intensiv 6x pro Woche. Wie stelle ich sicher, dass ich genug Protein und alle essentiellen Aminosäuren bekomme?",
    #"Nach meiner Gallenblasenoperation kann ich viele Lebensmittel nicht mehr essen. Welche Ernährungsstrategie hilft mir, wieder normal zu essen?",
    #"Ich bin 45 Jahre alt, habe ADHS und Schlafprobleme. Kann die richtige Ernährung meine Symptome verbessern?",
    #"Welche Lebensmittel helfen am besten gegen Migräne? Gibt es Trigger, die ich vermeiden sollte?",
    #"Mit 22 Jahren und 60 kg fällt es mir schwer zuzunehmen. Welche Ernährungsstrategie hilft mir beim gesunden Gewichtaufbau?",
    #"Als 35-jähriger Büroangestellter mit wenig Bewegung frage ich mich, wie ich meine Ernährung langfristig optimieren kann.",
    #"Regelmäßiges Marathontraining gehört zu meinem Alltag (28 Jahre). Welche Lebensmittel verbessern gezielt meine Ausdauerleistung?",
    #"Aufgrund von Bluthochdruck (50 Jahre) möchte ich meine Ernährung umstellen – worauf sollte ich besonders achten?",
    #"Ich wiege 95 kg bei 40 Jahren und möchte nachhaltig abnehmen. Welche Rolle spielen Makronährstoffe dabei konkret?",
    #"Seit einigen Jahren ernähre ich mich vegetarisch (19 Jahre). Wie kann ich mögliche Nährstoffdefizite vermeiden?",
    #"Häufige Verdauungsprobleme beeinträchtigen meinen Alltag (33 Jahre). Welche Ernährungsweise könnte helfen?",
    #"Mit 70 Jahren habe ich oft wenig Appetit. Wie kann ich dennoch eine ausreichende Nährstoffversorgung sicherstellen?",
    #"Trotz Pflege habe ich mit Akne zu kämpfen (26 Jahre). Welche Ernährungsfaktoren könnten mein Hautbild beeinflussen?",
    #"Ich trainiere regelmäßig im Fitnessstudio (31 Jahre, 80 kg). Wie bestimme ich meinen optimalen Proteinbedarf?",
    #"Nach der Diagnose Diabetes Typ 2 (45 Jahre) möchte ich meine Ernährung anpassen – insbesondere bei Kohlenhydraten. Was ist sinnvoll?",
    #"Schichtarbeit (38 Jahre) bringt meinen Essrhythmus durcheinander. Wie kann ich meine Mahlzeiten besser strukturieren?",
    #"Während des Studiums fällt es mir schwer, konzentriert zu bleiben (24 Jahre). Welche Rolle spielt Ernährung dabei?",
    #"Erhöhte Cholesterinwerte (55 Jahre) machen mir Sorgen. Welche Fette sollte ich bevorzugen oder vermeiden?",
    #"Dauerhafter Stress (30 Jahre) wirkt sich auf mein Wohlbefinden aus. Gibt es Lebensmittel, die mich unterstützen können?",
    #"Aufgrund einer Laktoseintoleranz (27 Jahre) suche ich nach Alternativen, um meinen Kalziumbedarf zu decken.",
    #"Mit einer Schilddrüsenerkrankung (42 Jahre) frage ich mich, welche Ernährung förderlich ist.",
    #"Ich praktiziere intermittierendes Fasten (36 Jahre). Wie sollte ich meine Mahlzeiten innerhalb des Essfensters gestalten?",
    #"Heißhungerattacken treten bei mir regelmäßig auf (48 Jahre). Welche Ernährungsstrategien helfen dagegen?",
    #"Als leistungsorientierter Fußballspieler (21 Jahre) interessiert mich, wie ich meine Regeneration durch Ernährung verbessern kann.",
    #"Osteoporose wurde bei mir diagnostiziert (60 Jahre). Welche Nährstoffe sind jetzt besonders wichtig?",
    # "Mit einer Glutenunverträglichkeit (34 Jahre) suche ich nach Möglichkeiten für eine ausgewogene Ernährung.",
    # "Mein Ziel ist es, den Körperfettanteil zu reduzieren (29 Jahre). Welche Ernährungsansätze sind dafür effektiv?",
    # "Wiederkehrende Migräneanfälle (41 Jahre) belasten mich. Welche Rolle spielen Ernährung und mögliche Trigger?",
    # "Ich möchte gezielt meine Darmflora verbessern (37 Jahre). Welche Lebensmittel sind dafür besonders geeignet?",
    # "Im Alltag fühle ich mich oft energielos (23 Jahre). Kann meine Ernährung daran schuld sein?",
    # "Zur Verbesserung meiner Herzgesundheit (52 Jahre) möchte ich meine Essgewohnheiten anpassen – was ist empfehlenswert?",
    # "Wechseljahresbeschwerden machen mir zu schaffen (46 Jahre). Welche Ernährung kann unterstützend wirken?",
    # "In Vorbereitung auf einen Triathlon (32 Jahre) suche ich nach einer optimalen Ernährungsstrategie für Training und Wettkampf.",
    # "Mit 65 Jahren möchte ich mein Immunsystem stärken. Welche Nährstoffe und Lebensmittel spielen dabei eine zentrale Rolle?",
    # "Nach einer Schwangerschaft (35 Jahre) möchte ich wieder fit werden. Welche Ernährung ist sinnvoll?",
    #]

    s1 = load_ngqa(difficulty="easy", has_conflict=False, limit=5)
    s2 = load_ngqa(difficulty="easy", has_conflict=True, limit=5)
    s3 = load_ngqa(difficulty="medium", has_conflict=False, limit=5)
    s4 = load_ngqa(difficulty="medium", has_conflict=True, limit=5)
    s5 = load_ngqa(difficulty="hard", summary_agrees_with_reference_answer=True, limit=5)
    s6 = load_ngqa(difficulty="hard", summary_agrees_with_reference_answer=False, limit=5)
    samples = s1 + s2 + s3 + s4 + s5 + s6

    # Shuffle the combined cross-stratum list with a fixed seed so each shard
    # gets a representative mix (and so reruns are reproducible).
    random.Random(0).shuffle(samples)

    # Build (query, metadata) pairs. The metadata dict is opaque to the
    # pipeline — each dataset's loader defines its own `to_metadata`, and the
    # fields it returns become top-level keys on every result row.
    items = [(s["query"], ngqa_to_metadata(s)) for s in samples]

    # Shard across SLURM array tasks. Global pipeline ids are assigned *before*
    # slicing so each shard's outputs carry their original index and can be
    # re-ordered correctly by rag.merge_shards.
    indexed_items = [(pid, q, meta) for pid, (q, meta) in enumerate(items)]
    shard_idx   = int(os.environ.get("RAG_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("RAG_SHARD_COUNT", "1"))
    shard_tag   = os.environ.get("RAG_SHARD_TAG", "local")
    if shard_count > 1:
        indexed_items = indexed_items[shard_idx::shard_count]

    print(f"\n{'='*80}")
    print(f"Starting RAG evaluation pipeline")
    print(f"Shard {shard_idx}/{shard_count}: {len(indexed_items)} of {len(items)} queries")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {LLAMACPP_RAG_MODEL}  (concurrency={LLAMACPP_RAG_CONCURRENCY})")
    print(f"{'='*80}\n", flush=True)

    pipeline_start = time.time()
    results = asyncio.run(run_rag_pipeline_async(indexed_items))
    pipeline_time = time.time() - pipeline_start

    print(f"\n{'='*80}")
    print(f"Printing results...")
    print(f"{'='*80}\n", flush=True)

    for idx, result in enumerate(results, 1):
        print(f"\n[Result {idx}/{len(results)}]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"-" * 80, flush=True)

    results_dir = os.environ.get("RESULTS_DIR", "results")
    # Group by array job id so the merge step can find all shards.
    shard_dir = os.path.join(results_dir, "_shards_rag", shard_tag.split("_")[0])
    os.makedirs(shard_dir, exist_ok=True)
    output_file = os.path.join(shard_dir, f"shard_{shard_tag}.json")
    dump_json(results, output_file)

    print(f"\n✓ Shard results saved to {output_file}")

    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"Pipeline time: {pipeline_time:.1f}s ({pipeline_time/60:.1f}m)")
    print(f"{'='*80}\n", flush=True)

