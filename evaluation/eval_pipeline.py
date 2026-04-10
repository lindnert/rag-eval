import json
import time
from datetime import datetime
from evaluation.ragas_eval import run_ragas
from evaluation.deepeval_eval import run_deepeval
from evaluation.custom_eval import run_custom


def load_rag_results(results_file="rag_results.json"):
    """Load RAG results from file"""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_results(results):
    """Evaluate all RAG results"""
    print(f"\n{'='*80}")
    print(f"Starting evaluation of {len(results)} results...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n", flush=True)
    
    start_time = time.time()
    
    for idx, result in enumerate(results, 1):
        query_preview = result['query'][:60]
        print(f"[Eval {idx}/{len(results)}] {query_preview}...", flush=True)
        
        result['ragas_scores'] = run_ragas(result)
        result['deepeval_scores'] = run_deepeval(result)
        result['custom_scores'] = run_custom(result)
        
        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0
        remaining = (len(results) - idx) / rate if rate > 0 else 0
        
        print(f"  ✓ Complete | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s\n", flush=True)
    
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
    
    results = load_rag_results("rag_results.json")

    evaluated_results = evaluate_results(results)
    
    save_evaluated_results(evaluated_results, "evaluated_results.json")