import os
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas.llms import LangchainLLMWrapper

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text")


# --- Base LLM ---
base_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434"
)


# --- JSON-forcing wrapper (IMPORTANT) ---
class RagasJSONWrapper:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt, **kwargs):
        prompt = f"""
You MUST return valid JSON only.
No explanation. No text outside JSON.

{prompt}
"""
        response = self.llm.invoke(prompt).content

        # fallback if model returns empty / garbage
        if not response or response.strip() == "":
            return "{}"

        return response


wrapped_llm = RagasJSONWrapper(base_llm)
ragas_llm = LangchainLLMWrapper(wrapped_llm)


embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDINGS_MODEL,
    base_url="http://localhost:11434"
)


def run_ragas(sample):
    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

    try:
        result = cast(
            EvaluationResult,
            evaluate(
                dataset,
                llm=ragas_llm,
                embeddings=embeddings,
                metrics=[Faithfulness(), AnswerRelevancy()],
                return_executor=False,
            ),
        )

        return {
            "ragas_faithfulness": result["faithfulness"][0],
            "ragas_answer_relevancy": result["answer_relevancy"][0],
        }

    except Exception as e:

        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_error": str(e)
        }