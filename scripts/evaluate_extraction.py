"""
Evaluate base model vs fine-tuned model on financial data extraction.

Runs both models on held-out test examples and automatically scores
the extractions by comparing each field to the ground truth.

This script runs on the GPU machine (RunPod) after training.

Usage: python scripts/evaluate_extraction.py
"""

import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

OUTPUT_DIR = "outputs/final"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# System prompt — same as training
SYSTEM_PROMPT = "You are a financial data extraction assistant. Given a block of financial text, extract all available financial data into a structured JSON format. Use null for any fields not mentioned or not inferable from the text. Be precise — every number must match the source text exactly."

# Test examples — different from training data
TEST_EXAMPLES = [
    {
        "name": "Semiconductor earnings",
        "text": """Advanced Micro Devices reported third quarter 2024 revenue of $6.8 billion, an increase of 18% year-over-year from $5.8 billion. Data Center segment revenue was $3.5 billion, up 122% year-over-year driven by strong demand for Instinct MI300X GPUs. Client segment revenue increased 29% to $1.9 billion. Gaming segment revenue declined 69% to $462 million. Gross margin was 54%, up from 51% in the prior year quarter. Operating income was $1.7 billion compared to $224 million a year ago. Net income was $1.5 billion, or $0.92 per diluted share, compared to $299 million, or $0.18 per diluted share. The company generated $628 million in free cash flow during the quarter.""",
        "ground_truth": {
            "company": "Advanced Micro Devices",
            "period": "Q3 2024",
            "total_revenue_m": 6800,
            "total_revenue_prior_m": 5800,
            "revenue_yoy_pct": 18,
            "segments": [
                {"name": "Data Center", "revenue_m": 3500, "yoy_pct": 122},
                {"name": "Client", "revenue_m": 1900, "yoy_pct": 29},
                {"name": "Gaming", "revenue_m": 462, "yoy_pct": -69}
            ],
            "revenue_by_type": [],
            "gross_margin_pct": 54,
            "gross_margin_change_bps": 300,
            "operating_income_m": 1700,
            "operating_margin_pct": 25.0,
            "net_income_m": 1500,
            "eps": 0.92,
            "cash_from_operations_m": None,
            "free_cash_flow_m": 628,
            "guidance_revenue_m": None,
            "flags": []
        }
    },
    {
        "name": "Retail quarterly results",
        "text": """Dollar General Corporation reported net sales of $9.9 billion for the second quarter of fiscal 2024, an increase of 4.2% from $9.5 billion in the prior year period. Same-store sales increased 0.5%, driven by increases in average transaction amount partially offset by a decline in customer traffic. Gross profit as a percentage of net sales was 30.0% compared to 31.1% in the prior year, a contraction of 110 basis points primarily due to increased markdowns and higher inventory shrink. Operating profit was $550 million compared to $692 million in the prior year quarter. Diluted earnings per share were $1.70 compared to $2.13 in the prior year period. For fiscal 2024, the company updated its guidance, now expecting net sales growth of approximately 4.7% to 5.3% and diluted EPS of $5.50 to $6.20.""",
        "ground_truth": {
            "company": "Dollar General Corporation",
            "period": "Q2 FY2024",
            "total_revenue_m": 9900,
            "total_revenue_prior_m": 9500,
            "revenue_yoy_pct": 4.2,
            "segments": [],
            "revenue_by_type": [],
            "gross_margin_pct": 30.0,
            "gross_margin_change_bps": -110,
            "operating_income_m": 550,
            "operating_margin_pct": None,
            "net_income_m": None,
            "eps": 1.70,
            "cash_from_operations_m": None,
            "free_cash_flow_m": None,
            "guidance_revenue_m": "4.7% to 5.3% growth",
            "flags": []
        }
    },
    {
        "name": "Biotech with minimal data",
        "text": """Vertex Pharmaceuticals announced that total net product revenues for the first quarter were $2.6 billion, representing growth of 12% compared to the first quarter of the prior year. The company reaffirmed its full-year 2025 net product revenue guidance of $11.3 billion to $11.6 billion.""",
        "ground_truth": {
            "company": "Vertex Pharmaceuticals",
            "period": "Q1 2025",
            "total_revenue_m": 2600,
            "total_revenue_prior_m": None,
            "revenue_yoy_pct": 12,
            "segments": [],
            "revenue_by_type": [],
            "gross_margin_pct": None,
            "gross_margin_change_bps": None,
            "operating_income_m": None,
            "operating_margin_pct": None,
            "net_income_m": None,
            "eps": None,
            "cash_from_operations_m": None,
            "free_cash_flow_m": None,
            "guidance_revenue_m": "$11.3B to $11.6B",
            "flags": []
        }
    },
]


def score_extraction(predicted: dict, ground_truth: dict) -> dict:
    """Score a predicted extraction against ground truth.

    Returns a dict with total fields, correct fields, and details.
    """
    # Fields to check (skip segments/revenue_by_type/flags — scored separately)
    scalar_fields = [
        "company", "period", "total_revenue_m", "total_revenue_prior_m",
        "revenue_yoy_pct", "gross_margin_pct", "gross_margin_change_bps",
        "operating_income_m", "operating_margin_pct", "net_income_m",
        "eps", "cash_from_operations_m", "free_cash_flow_m", "guidance_revenue_m",
    ]

    results = []
    correct = 0
    total = len(scalar_fields)

    for field in scalar_fields:
        expected = ground_truth.get(field)
        actual = predicted.get(field)

        # Both null = correct
        if expected is None and actual is None:
            correct += 1
            results.append({"field": field, "status": "correct", "expected": expected, "actual": actual})
        # Number comparison with tolerance
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(expected - actual) < 0.1:
                correct += 1
                results.append({"field": field, "status": "correct", "expected": expected, "actual": actual})
            else:
                results.append({"field": field, "status": "WRONG", "expected": expected, "actual": actual})
        # String comparison (loose)
        elif isinstance(expected, str) and isinstance(actual, str):
            if expected.lower().strip() in actual.lower().strip() or actual.lower().strip() in expected.lower().strip():
                correct += 1
                results.append({"field": field, "status": "correct", "expected": expected, "actual": actual})
            else:
                results.append({"field": field, "status": "WRONG", "expected": expected, "actual": actual})
        # One is null, other isn't
        elif expected is None and actual is not None:
            results.append({"field": field, "status": "WRONG (hallucinated)", "expected": "null", "actual": actual})
        elif expected is not None and actual is None:
            results.append({"field": field, "status": "WRONG (missed)", "expected": expected, "actual": "null"})
        else:
            results.append({"field": field, "status": "WRONG", "expected": expected, "actual": actual})

    return {
        "correct": correct,
        "total": total,
        "pct": round(100 * correct / total, 1),
        "details": results,
    }


def run_inference(model, tokenizer, text: str) -> dict:
    """Generate an extraction from the model and parse it as JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract the financial data from the following text:\n\n{text}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1000,
        temperature=0.1,  # Low temperature for deterministic extraction
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    # Try to parse as JSON
    try:
        # Clean up common issues
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse model output as JSON")
        print(f"  Raw output: {response[:200]}...")
        return {}


def main():
    print("=" * 60)
    print("CRUCIBLE — Extraction Model Evaluation")
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

    # Track overall scores
    base_scores = []
    ft_scores = []

    for example in TEST_EXAMPLES:
        print("\n" + "=" * 60)
        print(f"TEST: {example['name']}")
        print("=" * 60)
        print(f"\nText: {example['text'][:150]}...")

        # Base model
        print("\n--- BASE MODEL ---")
        base_result = run_inference(base_model, base_tokenizer, example["text"])
        base_score = score_extraction(base_result, example["ground_truth"])
        base_scores.append(base_score["pct"])
        print(f"Score: {base_score['correct']}/{base_score['total']} ({base_score['pct']}%)")
        for d in base_score["details"]:
            if d["status"] != "correct":
                print(f"  {d['status']}: {d['field']} — expected {d['expected']}, got {d['actual']}")

        # Fine-tuned model
        print("\n--- FINE-TUNED MODEL ---")
        ft_result = run_inference(ft_model, ft_tokenizer, example["text"])
        ft_score = score_extraction(ft_result, example["ground_truth"])
        ft_scores.append(ft_score["pct"])
        print(f"Score: {ft_score['correct']}/{ft_score['total']} ({ft_score['pct']}%)")
        for d in ft_score["details"]:
            if d["status"] != "correct":
                print(f"  {d['status']}: {d['field']} — expected {d['expected']}, got {d['actual']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Base model average:       {sum(base_scores) / len(base_scores):.1f}%")
    print(f"Fine-tuned model average: {sum(ft_scores) / len(ft_scores):.1f}%")
    diff = sum(ft_scores) / len(ft_scores) - sum(base_scores) / len(base_scores)
    if diff > 0:
        print(f"Improvement:              +{diff:.1f}%")
    else:
        print(f"Change:                   {diff:.1f}%")


if __name__ == "__main__":
    main()
