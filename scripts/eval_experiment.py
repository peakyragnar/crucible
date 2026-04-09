"""
Evaluate an experiment's model on extraction test cases.

Usage:
  python scripts/eval_experiment.py outputs/experiment_e3_r16_lr0.0002
  python scripts/eval_experiment.py outputs/experiment_e5_r16_lr0.0002
"""

import sys
import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

if len(sys.argv) < 2:
    print("Usage: python scripts/eval_experiment.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]

SYSTEM_PROMPT = "You are a financial data extraction assistant. Given a block of financial text, extract all available financial data into a structured JSON format. Use null for any fields not mentioned or not inferable from the text. Be precise — every number must match the source text exactly."

TEST_EXAMPLES = [
    {
        "name": "AMD semiconductor",
        "text": "Advanced Micro Devices reported third quarter 2024 revenue of $6.8 billion, an increase of 18% year-over-year from $5.8 billion. Data Center segment revenue was $3.5 billion, up 122% year-over-year. Client segment revenue increased 29% to $1.9 billion. Gaming segment revenue declined 69% to $462 million. Gross margin was 54%, up from 51% in the prior year quarter. Operating income was $1.7 billion. Net income was $1.5 billion, or $0.92 per diluted share. The company generated $628 million in free cash flow during the quarter.",
        "ground_truth": {
            "company": "Advanced Micro Devices",
            "period": "Q3 2024",
            "total_revenue_m": 6800,
            "total_revenue_prior_m": 5800,
            "revenue_yoy_pct": 18,
            "gross_margin_pct": 54,
            "gross_margin_change_bps": 300,
            "operating_income_m": 1700,
            "net_income_m": 1500,
            "eps": 0.92,
            "cash_from_operations_m": None,
            "free_cash_flow_m": 628,
            "guidance_revenue_m": None,
        }
    },
    {
        "name": "Dollar General retail",
        "text": "Dollar General Corporation reported net sales of $9.9 billion for the second quarter of fiscal 2024, an increase of 4.2% from $9.5 billion in the prior year period. Gross profit as a percentage of net sales was 30.0% compared to 31.1% in the prior year, a contraction of 110 basis points. Operating profit was $550 million. Diluted earnings per share were $1.70.",
        "ground_truth": {
            "company": "Dollar General Corporation",
            "period": "Q2 FY2024",
            "total_revenue_m": 9900,
            "total_revenue_prior_m": 9500,
            "revenue_yoy_pct": 4.2,
            "gross_margin_pct": 30.0,
            "gross_margin_change_bps": -110,
            "operating_income_m": 550,
            "net_income_m": None,
            "eps": 1.70,
            "cash_from_operations_m": None,
            "free_cash_flow_m": None,
            "guidance_revenue_m": None,
        }
    },
    {
        "name": "Vertex biotech minimal",
        "text": "Vertex Pharmaceuticals announced that total net product revenues for the first quarter were $2.6 billion, representing growth of 12% compared to the first quarter of the prior year. The company reaffirmed its full-year 2025 net product revenue guidance of $11.3 billion to $11.6 billion.",
        "ground_truth": {
            "company": "Vertex Pharmaceuticals",
            "period": "Q1 2025",
            "total_revenue_m": 2600,
            "total_revenue_prior_m": None,
            "revenue_yoy_pct": 12,
            "gross_margin_pct": None,
            "gross_margin_change_bps": None,
            "operating_income_m": None,
            "net_income_m": None,
            "eps": None,
            "cash_from_operations_m": None,
            "free_cash_flow_m": None,
            "guidance_revenue_m": "$11.3B to $11.6B",
        }
    },
]


def score(predicted, ground_truth):
    fields = [k for k in ground_truth.keys() if k not in ("segments", "revenue_by_type", "flags")]
    correct = 0
    total = len(fields)
    wrong = []

    for field in fields:
        expected = ground_truth[field]
        actual = predicted.get(field)

        if expected is None and actual is None:
            correct += 1
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(expected - actual) < 0.1:
                correct += 1
            else:
                wrong.append(f"  {field}: expected {expected}, got {actual}")
        elif isinstance(expected, str) and isinstance(actual, str):
            if expected.lower().strip() in actual.lower().strip() or actual.lower().strip() in expected.lower().strip():
                correct += 1
            else:
                wrong.append(f"  {field}: expected {expected}, got {actual}")
        elif expected is None and actual is not None:
            wrong.append(f"  {field}: expected null, got {actual} (hallucinated)")
        elif expected is not None and actual is None:
            wrong.append(f"  {field}: expected {expected}, got null (missed)")
        else:
            wrong.append(f"  {field}: expected {expected}, got {actual}")

    return correct, total, wrong


print(f"Loading model from {model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
    local_files_only=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

total_correct = 0
total_fields = 0

for test in TEST_EXAMPLES:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract the financial data from the following text:\n\n{test['text']}"},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=1000, temperature=0.1, top_p=0.9)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    # Parse JSON
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        predicted = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        predicted = {}
        print(f"\n  WARNING: Could not parse JSON for {test['name']}")

    c, t, wrong = score(predicted, test["ground_truth"])
    total_correct += c
    total_fields += t

    print(f"\n{test['name']}: {c}/{t} ({100*c/t:.0f}%)")
    for w in wrong:
        print(w)

pct = 100 * total_correct / total_fields
print(f"\n{'='*40}")
print(f"OVERALL: {total_correct}/{total_fields} ({pct:.1f}%)")
print(f"{'='*40}")
