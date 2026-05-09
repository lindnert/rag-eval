import json
import time
import requests
from datetime import datetime
from evaluation.ragas_eval import run_ragas_batch
from evaluation.deepeval_eval import run_deepeval_batch
from evaluation.custom_eval import run_custom


def load_rag_results(results_file="rag_results.json"):
    """Load RAG results from file"""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_results(results, partial_path=None):
    """Evaluate all RAG results. Writes partial progress to partial_path after each item."""
    print(f"\n{'='*80}")
    print(f"Starting evaluation of {len(results)} results...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)

    start_time = time.time()

    # Phase 1: ragas scores in parallel (Ollama serves multiple slots concurrently
    # if started with OLLAMA_NUM_PARALLEL >= RAGAS_CONCURRENCY).
    print(f"[Phase 1/2] Running ragas concurrently for {len(results)} samples...", flush=True)
    ragas_start = time.time()
    ragas_done = {"n": 0}

    def _on_ragas_done(idx, sample, scores):
        ragas_done["n"] += 1
        n = ragas_done["n"]
        elapsed = time.time() - start_time
        rate = n / elapsed if elapsed > 0 else 0
        remaining = (len(results) - n) / rate if rate > 0 else 0
        preview = sample['query'][:60]
        faith = scores.get("ragas_faithfulness")
        relev = scores.get("ragas_answer_relevancy")
        err = scores.get("ragas_error")
        score_str = f"faith={faith} relev={relev}"
        if err:
            score_str += f" ERROR={err}"
        print(
            f"  [ragas {n}/{len(results)}] sample={idx} | {score_str} "
            f"| {preview}... | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s",
            flush=True,
        )
        results[idx]['ragas_scores'] = scores
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        pass #run_ragas_batch(results, on_done=_on_ragas_done) TODO: re-enable after testing
    except Exception as e:
        print(f"  ✗ ragas batch failed: {e}", flush=True)

    ragas_elapsed = time.time() - ragas_start
    print(
        f"\n[Phase 1/2] ragas total time: {ragas_elapsed:.1f}s "
        f"({ragas_elapsed/60:.1f}m) for {len(results)} samples "
        f"({ragas_elapsed/max(1, len(results)):.1f}s/sample avg)",
        flush=True,
    )

    print(requests.get("http://127.0.0.1:11434/api/ps").json(), flush=True)

    # Phase 2: deepeval concurrently (mirrors Phase 1 — Ollama serves multiple
    # slots in parallel when started with OLLAMA_NUM_PARALLEL >= DEEPEVAL_CONCURRENCY).
    print(f"\n[Phase 2/2] Running deepeval concurrently for {len(results)} samples...", flush=True)
    phase2_start = time.time()
    deepeval_done = {"n": 0}

    def _on_deepeval_done(idx, sample, scores):
        deepeval_done["n"] += 1
        n = deepeval_done["n"]
        elapsed = time.time() - phase2_start
        rate = n / elapsed if elapsed > 0 else 0
        remaining = (len(results) - n) / rate if rate > 0 else 0
        preview = sample['query'][:60]
        faith = scores.get("deepeval_faithfulness")
        relev = scores.get("deepeval_relevance")
        err = scores.get("deepeval_error")
        score_str = f"faith={faith} relev={relev}"
        if err:
            score_str += f" ERROR={err}"
        print(
            f"  [deepeval {n}/{len(results)}] sample={idx} | {score_str} "
            f"| {preview}... | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s",
            flush=True,
        )
        results[idx]['deepeval_scores'] = scores
        if partial_path:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        run_deepeval_batch(results, on_done=_on_deepeval_done)
    except Exception as e:
        print(f"  ✗ deepeval batch failed: {e}", flush=True)

    phase2_elapsed = time.time() - phase2_start
    print(
        f"\n[Phase 2/2] deepeval total time: {phase2_elapsed:.1f}s "
        f"({phase2_elapsed/60:.1f}m) for {len(results)} samples "
        f"({phase2_elapsed/max(1, len(results)):.1f}s/sample avg)",
        flush=True,
    )

    eval_time = time.time() - start_time
    print(f"{'='*80}")
    print(f"Evaluation complete!")
    print(f"Total time: {eval_time:.1f}s ({eval_time/60:.1f}m)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)
    
    return results


def save_evaluated_results(results, output_file="evaluated_results.json"):
    """Save fully evaluated results"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Evaluated results saved to {output_file}\n", flush=True)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"
    os.environ["DEEPEVAL_MAX_RETRIES_OVERRIDE"] = "3"

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    rag_input = os.environ.get(
        "RAG_RESULTS_FILE",
        os.path.join(results_dir, "rag_results_latest.json"),
    )
    results = load_rag_results(rag_input)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_file = os.path.join(results_dir, f"evaluated_results_{timestamp}.partial.json")
    evaluated_results = evaluate_results(results, partial_path=partial_file)

    out_file = os.path.join(results_dir, f"evaluated_results_{timestamp}.json")
    save_evaluated_results(evaluated_results, out_file)
    save_evaluated_results(
        evaluated_results,
        os.path.join(results_dir, "evaluated_results_latest.json"),
    )