"""
DeepEvalBaseLLM wrappers around the llama-server OpenAI endpoint for the
synthesizer. Mirrors evaluation/deepeval_eval.py's LlamaCppWrapper (schema
coercion + retries) without its per-call debug logging.

Two instances are used:
- generator: SYNTH_TEMPERATURE (mildly creative — question variety)
- critic:    temperature 0.0    (deterministic quality scores / validation)
"""

import json
import os
import re

from dotenv import load_dotenv

# Pick up the repo .env (external Ollama node URL + API key) BEFORE
# eval_config_llamacpp reads LLAMACPP_GEN_BASE_URL at import time. load_dotenv
# never overrides variables already in the environment, so SLURM runs (which
# export the localhost llama-server URL explicitly) are unaffected — and .env
# is gitignored, so it doesn't exist on the cluster anyway.
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from deepeval.models.base_model import DeepEvalBaseLLM

from evaluation.eval_config_llamacpp import (
    LLAMACPP_EVAL_MODEL,
    LLAMACPP_GEN_BASE_URL,
    LLAMACPP_NUM_PREDICT,
    LLAMACPP_REPEAT_LAST_N,
    LLAMACPP_REPEAT_PENALTY,
    LLAMACPP_TOP_P,
)
from evaluation.utils import _prompt_to_text, _strip_code_fences
from dataset.synthetic.synth_config import SYNTH_TEMPERATURE

SYNTH_SYSTEM_PROMPT = (
    "You are a synthetic dataset generation assistant for evaluating a "
    "nutrition question-answering system. Follow the user's instructions "
    "exactly and return your answer as a single JSON object that matches the "
    "schema given in the prompt. Do not wrap the JSON in markdown code fences "
    "and do not add commentary before or after it."
)

_MAX_SCHEMA_RETRIES = 3

# Bearer token for the generation endpoint. Unset -> llama-server placeholder
# (no auth needed); set (directly, or via OLLAMA_API_KEY in .env) -> the
# authenticated external Ollama node. Same env vars as rag/llm_config.py.
SYNTH_API_KEY = (
    os.getenv("LLAMACPP_GEN_API_KEY")
    or os.getenv("OLLAMA_API_KEY")
    or "sk-no-key-required"
)

# gemma4 via Ollama's /v1 emits <think> blocks inline in content even when
# thinking is toggled off (verified 2026-07-13 on gemma4:12b); llama-server
# keeps them in a separate reasoning_content field. Strip them defensively so
# JSON parsing works against either backend.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _clean_response(text: str) -> str:
    return _strip_code_fences(_THINK_RE.sub("", text or "").strip())


def _coerce_to_schema(text: str, schema):
    cleaned = _strip_code_fences(text or "").strip() or "{}"
    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ValueError(
            f"LLM returned non-JSON for schema {schema.__name__}: {e}\n--RAW--\n{text}"
        )
    try:
        if isinstance(data, dict):
            return schema(**data)
        return schema.model_validate(data)
    except Exception as e:
        # Single-field schemas: deepeval's evolution step expects
        # Response(response=...) but its prompt never names the JSON key, so
        # local models answer with keys like "rewritten_input". When both the
        # schema and the reply have exactly one field, the mapping is
        # unambiguous — remap instead of burning retries on a key name.
        fields = getattr(schema, "model_fields", {})
        if isinstance(data, dict) and len(data) == 1 and len(fields) == 1:
            try:
                return schema(**{next(iter(fields)): next(iter(data.values()))})
            except Exception:
                pass
        raise ValueError(
            f"LLM JSON did not match schema {schema.__name__}: {e}\n"
            f"--PARSED-- {data}\n--RAW-- {text}"
        )


class SynthLlamaCppWrapper(DeepEvalBaseLLM):
    def __init__(self, temperature: float, tag: str):
        self.temperature = temperature
        self.tag = tag
        self.llm = ChatOpenAI(
            model=LLAMACPP_EVAL_MODEL,
            base_url=LLAMACPP_GEN_BASE_URL,
            api_key=SecretStr(SYNTH_API_KEY),
            temperature=temperature,
            top_p=LLAMACPP_TOP_P,
            max_completion_tokens=LLAMACPP_NUM_PREDICT,
            streaming=False,
            model_kwargs={"response_format": {"type": "json_object"}},
            extra_body={
                "repeat_penalty": LLAMACPP_REPEAT_PENALTY,
                "repeat_last_n": LLAMACPP_REPEAT_LAST_N,
            },
        )

    def load_model(self):
        return self.llm

    def _messages(self, prompt):
        return [
            SystemMessage(content=SYNTH_SYSTEM_PROMPT),
            HumanMessage(content=_prompt_to_text(prompt)),
        ]

    def generate(self, prompt, schema=None, **kwargs):
        last_err: Exception | None = None
        for attempt in range(1, _MAX_SCHEMA_RETRIES + 1):
            response = _clean_response(self.llm.invoke(self._messages(prompt)).content or "")
            if schema is None:
                return response
            try:
                return _coerce_to_schema(response, schema)
            except ValueError as e:
                last_err = e
                print(
                    f"[synth_llm:{self.tag}] schema validation failed "
                    f"(attempt {attempt}/{_MAX_SCHEMA_RETRIES}) for "
                    f"{getattr(schema, '__name__', '?')}: {e}",
                    flush=True,
                )
        assert last_err is not None
        raise last_err

    async def a_generate(self, prompt, schema=None, **kwargs):
        last_err: Exception | None = None
        for attempt in range(1, _MAX_SCHEMA_RETRIES + 1):
            result = await self.llm.ainvoke(self._messages(prompt))
            response = _clean_response((result.content if result is not None else "") or "")
            if schema is None:
                return response
            try:
                return _coerce_to_schema(response, schema)
            except ValueError as e:
                last_err = e
                print(
                    f"[synth_llm:{self.tag}] schema validation failed "
                    f"(attempt {attempt}/{_MAX_SCHEMA_RETRIES}) for "
                    f"{getattr(schema, '__name__', '?')}: {e}",
                    flush=True,
                )
        assert last_err is not None
        raise last_err

    def get_model_name(self):
        return f"llamacpp-{LLAMACPP_EVAL_MODEL}-{self.tag}"


def build_generator() -> SynthLlamaCppWrapper:
    return SynthLlamaCppWrapper(temperature=SYNTH_TEMPERATURE, tag="gen")


def build_critic() -> SynthLlamaCppWrapper:
    return SynthLlamaCppWrapper(temperature=0.0, tag="critic")
