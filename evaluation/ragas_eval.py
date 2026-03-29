import os
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from ragas.llms import LangchainLLMWrapper

ollama_llm = OllamaLLM(model="gemma3:1b")
ragas_llm = LangchainLLMWrapper(ollama_llm)


def _build_embeddings():
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "gemma3:1b")
    return OllamaEmbeddings(model=embedding_model)


def run_ragas(sample):
    dataset = Dataset.from_dict({
        "question": [sample["query"]],
        "answer": [sample["answer"]],
        "contexts": [sample["contexts"]],
    })

    result = cast(
        EvaluationResult,
        evaluate(
            dataset,
            llm=ragas_llm,
            embeddings=_build_embeddings(),
            metrics=[Faithfulness(), AnswerRelevancy()],
            return_executor=False,
        ),
    )

    return {
        "ragas_faithfulness": result["faithfulness"][0],
        "ragas_answer_relevancy": result["answer_relevancy"][0],
    }