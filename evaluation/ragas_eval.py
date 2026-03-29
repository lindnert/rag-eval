# evaluation/ragas_eval.py

from ragas import evaluate
from datasets import Dataset

def run_ragas(sample):
    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

    result = evaluate(dataset)

    return {
        "ragas_faithfulness": result["faithfulness"],
        "ragas_answer_relevance": result["answer_relevancy"],
    }