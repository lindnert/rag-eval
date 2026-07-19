"""
DeepEvalBaseLLM wrappers around the llama-server OpenAI endpoint for the
synthesizer. Mirrors evaluation/deepeval_eval.py's LlamaCppWrapper (schema
coercion + retries) without its per-call debug logging.

Two instances are used:
- generator: SYNTH_TEMPERATURE (mildly creative — question variety)
- critic:    temperature 0.0    (deterministic quality scores / validation)
"""

import json

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
            api_key=SecretStr("sk-no-key-required"),
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
            response = _strip_code_fences(self.llm.invoke(self._messages(prompt)).content or "")
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
            response = _strip_code_fences((result.content if result is not None else "") or "")
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
