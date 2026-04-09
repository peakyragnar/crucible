"""Quick evaluation of DPO-trained model on revenue analysis."""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

print("Loading DPO model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/dpo_final",
    max_seq_length=2048,
    load_in_4bit=True,
    local_files_only=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

# Test scenarios the model has NEVER seen
tests = [
    {
        "name": "Streaming company - subscription fatigue",
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
        "name": "Cybersecurity - international expansion",
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

system = "You are a financial analyst specializing in revenue trend analysis. Provide direct, quantified analysis focused on what matters for the investment case."

for t in tests:
    print("\n" + "=" * 60)
    print("TEST: " + t["name"])
    print("=" * 60)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": t["prompt"]},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=800, temperature=0.7, top_p=0.9,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(response)
