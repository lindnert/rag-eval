import re
import shutil
import subprocess

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


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