from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import os

def run_deepeval(sample):
    print("Using timeout: ", os.getenv("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"))
    print("Using max retries: ", os.getenv("DEEPEVAL_MAX_RETRIES_OVERRIDE"))
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"]
    )

    ollama_model = OllamaModel(model="gemma3:1b")
    faithfulness = FaithfulnessMetric(model=ollama_model)
    relevance = AnswerRelevancyMetric(model=ollama_model)

    faithfulness.measure(test_case)
    relevance.measure(test_case)

    return {
        "deepeval_faithfulness": faithfulness.score,
        "deepeval_relevance": relevance.score,
    }