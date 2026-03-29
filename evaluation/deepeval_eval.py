from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

def run_deepeval(sample):
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"]
    )

    faithfulness = FaithfulnessMetric()
    relevance = AnswerRelevancyMetric()

    faithfulness.measure(test_case)
    relevance.measure(test_case)

    return {
        "deepeval_faithfulness": faithfulness.score,
        "deepeval_relevance": relevance.score,
    }