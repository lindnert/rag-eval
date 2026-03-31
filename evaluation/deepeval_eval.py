from deepeval import prompt
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

class OllamaWrapper(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, **kwargs):
        prompt = f"""
        Respond ONLY with valid JSON. No explanation.

        {prompt}
        """
        try:
            response = self.llm.invoke(prompt).content
            if not response or response.strip() == "":
                return "{}"
            return response
        except Exception:
            return "{}"

    async def a_generate(self, prompt: str, **kwargs):
        return self.generate(prompt)

    def get_model_name(self):
        return "ollama-local"

def run_deepeval(sample):
    print("Using timeout: ", os.getenv("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"))
    print("Using max retries: ", os.getenv("DEEPEVAL_MAX_RETRIES_OVERRIDE"))
    test_case = LLMTestCase(
        input=sample["query"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"]
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url="http://localhost:11434"
    )
    ollama_model = OllamaWrapper(llm)
    faithfulness = FaithfulnessMetric(model=ollama_model)
    relevance = AnswerRelevancyMetric(model=ollama_model)

    faithfulness.measure(test_case)
    relevance.measure(test_case)

    return {
        "deepeval_faithfulness": faithfulness.score,
        "deepeval_relevance": relevance.score,
    }