from deepeval import prompt
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
import os
from utils import _prompt_to_text

from evaluation.eval_config import (
    OLLAMA_EVAL_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TOP_P,
    OLLAMA_CONTEXT_LENGTH,
    JSON_SYSTEM_PROMPT,
)

class OllamaWrapper(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, **kwargs):
        try:
            messages = [
                SystemMessage(content=JSON_SYSTEM_PROMPT),
                HumanMessage(content=_prompt_to_text(prompt)),
            ]
            response = self.llm.invoke(messages).content

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
        base_url="http://localhost:11434",
        format="json",
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
        top_p=OLLAMA_TOP_P,
        num_ctx=OLLAMA_CONTEXT_LENGTH,
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