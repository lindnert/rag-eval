def _prompt_to_text(prompt) -> str:
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return str(prompt)