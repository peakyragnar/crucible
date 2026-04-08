"""
Compare base model vs fine-tuned model on the same prompts.

Runs both models on a set of test scenarios and prints outputs
side by side so you can judge whether fine-tuning improved anything.

This script runs on the GPU machine (RunPod) after training.

Usage: python scripts/evaluate.py
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

OUTPUT_DIR = "outputs/final"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# Test scenarios — different from training data so we're testing
# generalization, not memorization.
TEST_SCENARIOS = [
    {
        "name": "Subscription fatigue",
        "prompt": """Analyze the revenue trend for StreamMax Media (Digital Streaming).

| Quarter | Revenue ($M) |
|---------|-------------|
| Q1 2023 | 520 |
| Q2 2023 | 545 |
| Q3 2023 | 558 |
| Q4 2023 | 572 |
| Q1 2024 | 580 |
| Q2 2024 | 578 |
| Q3 2024 | 575 |
| Q4 2024 | 571 |

Additional context: Subscription SVOD model. Price increase of 12% in Q1 2024. Subscriber count declining 2% per quarter since Q2 2024. Ad-supported tier launched Q3 2023 growing 40% QoQ but only 8% of revenue. Churn rate increased from 4.2% to 5.8% monthly.

What's the story these numbers are telling? What would you want to dig into further?""",
    },
    {
        "name": "Geographic expansion",
        "prompt": """Analyze the revenue trend for SecureNet Cyber (Cybersecurity).

| Quarter | Revenue ($M) |
|---------|-------------|
| Q1 2023 | 95 |
| Q2 2023 | 102 |
| Q3 2023 | 108 |
| Q4 2023 | 118 |
| Q1 2024 | 125 |
| Q2 2024 | 140 |
| Q3 2024 | 152 |
| Q4 2024 | 168 |

Additional context: US revenue flat at ~$85M/quarter since Q3 2023. All growth from international expansion (EMEA + APAC). Opened London and Singapore offices in 2023. International gross margins 15 points below US. Hiring aggressively — headcount up 45% YoY.

What's the story these numbers are telling? What would you want to dig into further?""",
    },
]


def run_inference(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": "You are a financial analyst specializing in revenue trend analysis. Provide direct, quantified analysis focused on what matters for the investment case."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=800,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode only the new tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


def main():
    print("=" * 60)
    print("CRUCIBLE — Model Evaluation")
    print("=" * 60)

    # Load base model
    print("\nLoading base model...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    base_tokenizer = get_chat_template(base_tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(base_model)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(ft_model)

    # Run both models on each test scenario
    for scenario in TEST_SCENARIOS:
        print("\n" + "=" * 60)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 60)

        print("\n--- BASE MODEL ---")
        base_response = run_inference(base_model, base_tokenizer, scenario["prompt"])
        print(base_response)

        print("\n--- FINE-TUNED MODEL ---")
        ft_response = run_inference(ft_model, ft_tokenizer, scenario["prompt"])
        print(ft_response)

        print("\n--- YOUR JUDGMENT ---")
        print("Which response is better? What specifically improved or got worse?")
        print("(Note this down — it tells you whether the training data was good.)")


if __name__ == "__main__":
    main()
