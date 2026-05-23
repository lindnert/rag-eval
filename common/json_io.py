import json
import re

# Fields whose values are flat numeric arrays we want collapsed onto one line
# in the dumped JSON — keeps full precision but avoids long scroll walls in
# the result files (gen_logprobs is typically hundreds of tokens per sample).
_COMPACT_ARRAY_FIELDS = (
    "gen_logprobs",
    "retrieval_scores",
    "original_gen_logprobs",
    "original_retrieval_scores",
)

_COMPACT_PATTERNS = [
    re.compile(
        r'("' + re.escape(field) + r'":\s*)\[\s*([^\[\]]*?)\s*\]',
        re.DOTALL,
    )
    for field in _COMPACT_ARRAY_FIELDS
]


def _collapse(match):
    inner = re.sub(r"\s+", " ", match.group(2)).strip()
    return match.group(1) + "[" + inner + "]"


def dumps(obj, *, indent=2, ensure_ascii=False):
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    for pattern in _COMPACT_PATTERNS:
        text = pattern.sub(_collapse, text)
    return text


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(obj))
