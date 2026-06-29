import re
import shutil
import subprocess

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)

# Sentinel written to context-dependent metric fields for the no_rag variant,
# where running those metrics would be meaningless (no retrieved contexts).
NO_RAG_SENTINEL = "no rag - no retrieved contexts"

# Sentinel written to answer-relevancy fields when the generation abstained
# (row["rejected"] is True). Relevancy of the canonical REJECTION_ANSWER to the
# question is meaningless and would otherwise drag the metric down, so we skip
# scoring it. Distinct from None (genuine NaN/error) so the aggregation layer
# can filter abstentions out and analyse them as a separate outcome class.
REJECTED_SENTINEL = "rejected - answer relevancy not scored"


def _prompt_to_text(prompt) -> str:
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return str(prompt)


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    m = _CODE_FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def print_gpu_diagnostics(label: str = "after first call"):
    print(f"\n========== GPU diagnostics ({label}) ==========", flush=True)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            print(out.stdout, flush=True)
            if out.stderr:
                print(out.stderr, flush=True)
        except Exception as e:
            print(f"[diag] nvidia-smi failed: {e}", flush=True)
    else:
        print("[diag] nvidia-smi not on PATH", flush=True)
    print("========== end diagnostics ==========\n", flush=True)