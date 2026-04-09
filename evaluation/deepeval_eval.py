from deepeval import prompt
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import os

OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:2b")

class OllamaWrapper(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, **kwargs):
        prompt = f"""
            You MUST return valid JSON.

            IMPORTANT:
            - Always include the field "claims" as a list of strings.
            - If unsure, return: {{"claims": []}}

            NO explanation. ONLY JSON.

        {prompt}
        """
        try:
            response = self.llm.invoke(prompt).content

            if not response or response.strip() == "":
                return '{"claims": []}'

            return response

        except Exception:
            return '{"claims": []}'

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
        model=OLLAMA_EVAL_MODEL,
        base_url="http://localhost:11434"
    )
    ollama_model = OllamaWrapper(llm)
    faithfulness = FaithfulnessMetric(model=ollama_model)
    relevance = AnswerRelevancyMetric(model=ollama_model)

    try:
        faithfulness.measure(test_case)
        relevance.measure(test_case)

        return {
            "deepeval_faithfulness": faithfulness.score,
            "deepeval_relevance": relevance.score,
        }

    except Exception as e:
        return {
            "deepeval_faithfulness": None,
            "deepeval_relevance": None,
            "deepeval_error": str(e),
        }