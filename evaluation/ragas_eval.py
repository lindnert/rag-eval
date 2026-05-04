import os
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import SystemMessage, HumanMessage
from ragas.llms import LangchainLLMWrapper

OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "qwen3.5:2b")
OLLAMA_EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text")

JSON_SYSTEM_PROMPT = (
    "Follow the user's instructions and return your answer as a single JSON object "
    "that matches the schema given in the prompt. "
    "Do not wrap the JSON in markdown code fences and do not add commentary before or after it."
)

print(f"[ragas_eval] OLLAMA_EVAL_MODEL = {OLLAMA_EVAL_MODEL}", flush=True)

EVAL_DEBUG_LLM = os.getenv("EVAL_DEBUG_LLM", "1") == "1"
_call_counter = {"n": 0}


base_llm = ChatOllama(
    model=OLLAMA_EVAL_MODEL,
    base_url="http://localhost:11434",
)


class RagasJSONWrapper:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt, **kwargs):
        messages = [
            SystemMessage(content=JSON_SYSTEM_PROMPT),
            HumanMessage(content=str(prompt)),
        ]
        response = self.llm.invoke(messages).content or "{}"

        if EVAL_DEBUG_LLM:
            _call_counter["n"] += 1
            n = _call_counter["n"]
            prompt_str = str(prompt)
            print(f"\n--- ragas LLM call #{n} ---", flush=True)
            print(f"[PROMPT first 400] {prompt_str[:400]}", flush=True)
            print(f"[PROMPT  last 200] {prompt_str[-200:]}", flush=True)
            print(f"[RESPONSE len={len(response)}]\n{response[:1500]}", flush=True)
            print(f"--- end call #{n} ---\n", flush=True)

        return response

    async def agenerate(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def generate_prompt(self, prompts, **kwargs):
        outputs = []
        for p in prompts:
            text = self.generate(p, **kwargs)
            outputs.append([Generation(text=text)])
        return LLMResult(generations=outputs)

    async def agenerate_prompt(self, prompts, **kwargs):
        return self.generate_prompt(prompts, **kwargs)


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
            "ragas_faithfulness": result["faithfulness"][0] if result["faithfulness"][0] == result["faithfulness"][0] else None,
            "ragas_answer_relevancy": result["answer_relevancy"][0] if result["answer_relevancy"][0] == result["answer_relevancy"][0] else None,
        }

    except Exception as e:

        return {
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_error": str(e)
        }