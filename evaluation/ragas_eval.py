import os
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper

OLLAMA_MODEL = "phi3"
OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"

ollama_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434"
)
ragas_llm = LangchainLLMWrapper(ollama_llm)


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