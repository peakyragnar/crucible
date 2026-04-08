"""
Convert raw generated data into training-ready format for unsloth.

Unsloth expects a HuggingFace Dataset with a specific chat format.
This script reads our JSONL, validates it, and outputs a clean dataset.

Usage: python scripts/curate.py
"""

import json
from datetime import datetime


def load_raw_data(path: str) -> list:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            example = json.loads(line)
            examples.append(example)
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def validate_example(example: dict) -> list:
    """Check a single example for problems. Returns list of issues."""
    issues = []
    convos = example.get("conversations", [])

    if len(convos) != 3:
        issues.append(f"Expected 3 turns (system/user/assistant), got {len(convos)}")
        return issues

    if convos[0]["role"] != "system":
        issues.append(f"First turn should be 'system', got '{convos[0]['role']}'")
    if convos[1]["role"] != "user":
        issues.append(f"Second turn should be 'user', got '{convos[1]['role']}'")
    if convos[2]["role"] != "assistant":
        issues.append(f"Third turn should be 'assistant', got '{convos[2]['role']}'")

    # Check for empty content
    for turn in convos:
        if not turn.get("content", "").strip():
            issues.append(f"Empty content in {turn['role']} turn")

    # Check assistant response length (too short = probably bad)
    assistant_len = len(convos[2]["content"].split())
    if assistant_len < 50:
        issues.append(f"Assistant response very short ({assistant_len} words)")

    return issues


def convert_to_training_format(examples: list) -> list:
    """Convert our format to what unsloth expects.

    Unsloth uses the ShareGPT format:
    {"conversations": [{"from": "system", "value": "..."}, ...]}

    Our format uses "role" and "content".
    Unsloth expects "from" and "value".
    """
    training_data = []
    for example in examples:
        converted = {
            "conversations": [
                {"from": turn["role"], "value": turn["content"]}
                for turn in example["conversations"]
            ]
        }
        training_data.append(converted)
    return training_data


def main():
    raw_path = "data/raw/revenue_trend_analysis_v1.jsonl"
    output_path = "data/curated/revenue_trend_analysis_v1.jsonl"

    # Load
    examples = load_raw_data(raw_path)

    # Validate
    print("\nValidating...")
    valid = []
    for i, example in enumerate(examples):
        issues = validate_example(example)
        company = example.get("metadata", {}).get("company", f"Example {i+1}")
        if issues:
            print(f"  SKIP: {company}")
            for issue in issues:
                print(f"    - {issue}")
        else:
            valid.append(example)
            print(f"  OK:   {company}")

    print(f"\n{len(valid)}/{len(examples)} examples passed validation")

    # Convert format
    training_data = convert_to_training_format(valid)

    # Write
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(training_data)} training examples to {output_path}")

    # Summary stats
    word_counts = []
    for example in valid:
        words = len(example["conversations"][2]["content"].split())
        word_counts.append(words)

    print(f"\nResponse length stats:")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    print(f"  Avg: {sum(word_counts) // len(word_counts)} words")


if __name__ == "__main__":
    main()
