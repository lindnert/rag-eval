import re

def extract_numbers(text):
    if isinstance(text, list):
        text = " ".join(text)
    return [int(x) for x in re.findall(r"\d+", text)]

def numerical_correctness(sample):
    numbers = extract_numbers(sample["answer"])

    if not numbers:
        return 0.0

    value = numbers[0]

    if 50 <= value <= 100:
        return 1.0
    else:
        return 0.0

def unsupported_precision(sample):
    numbers = extract_numbers(sample["answer"])

    if numbers and not extract_numbers(sample["contexts"]):
        return 1.0

    return 0.0

def run_custom(sample):
    return {
        "numerical_correctness": numerical_correctness(sample),
        "unsupported_precision": unsupported_precision(sample),
    }