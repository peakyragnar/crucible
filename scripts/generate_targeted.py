"""
Generate targeted training data to fix specific model weaknesses.

Based on error analysis from v1 evaluation:
1. Field confusion — cash_from_operations vs free_cash_flow
2. Number parsing — billions/millions conversions
3. Null discipline — should be null even when inference is possible

Usage: uv run python scripts/generate_targeted.py
"""

import anthropic
import json
import hashlib
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"
VERSION = "v2"

SCHEMA_BLOCK = """{
  "company": "string or null",
  "period": "string or null",
  "total_revenue_m": number or null,
  "total_revenue_prior_m": number or null,
  "revenue_yoy_pct": number or null,
  "segments": [{"name": "string", "revenue_m": number, "yoy_pct": number or null}],
  "revenue_by_type": [{"name": "string", "revenue_m": number, "yoy_pct": number or null}],
  "gross_margin_pct": number or null,
  "gross_margin_change_bps": number or null,
  "operating_income_m": number or null,
  "operating_margin_pct": number or null,
  "net_income_m": number or null,
  "eps": number or null,
  "cash_from_operations_m": number or null,
  "free_cash_flow_m": number or null,
  "guidance_revenue_m": "string or null",
  "flags": ["string"]
}"""

BASE_SYSTEM_PROMPT = """You are generating training data for a financial data extraction model.

Your task: create a realistic paragraph of financial text AND the correct structured JSON extraction.

Requirements for the JSON extraction:
- You MUST use EXACTLY these field names — no variations, no renaming, no additions:

{schema}

- Every response MUST include ALL of these fields. Use null for fields not mentioned in the text.
- Use empty arrays [] for segments and revenue_by_type if not mentioned.
- Numbers in millions. Percentages as numbers (17 not "17%"). Basis points as integers.
- The JSON must be perfectly accurate. Every number must match the text exactly.

Return your response in this exact format:

TEXT:
[the financial paragraph]

EXTRACTION:
[the JSON extraction]
"""

# Each target defines a weakness and how to generate examples that fix it
TARGETS = [
    # --- Target 1: Field confusion between cash flow types ---
    {
        "name": "cash_flow_both_present",
        "count": 5,
        "prompt": """Generate a financial paragraph that mentions BOTH operating cash flow AND free cash flow as separate numbers. The text must make it clear which is which. For example: "Cash from operations was $X million. After capital expenditures of $Y million, free cash flow was $Z million."

The extraction MUST have different values for cash_from_operations_m and free_cash_flow_m.
Sector: {sector}""",
    },
    {
        "name": "cash_flow_only_operating",
        "count": 5,
        "prompt": """Generate a financial paragraph that mentions operating cash flow / cash from operations but does NOT mention free cash flow or capital expenditures.

The extraction MUST have a value for cash_from_operations_m and null for free_cash_flow_m.
Sector: {sector}""",
    },
    {
        "name": "cash_flow_only_fcf",
        "count": 5,
        "prompt": """Generate a financial paragraph that mentions free cash flow but does NOT mention operating cash flow or cash from operations.

The extraction MUST have null for cash_from_operations_m and a value for free_cash_flow_m.
Sector: {sector}""",
    },
    # --- Target 2: Number parsing with billions ---
    {
        "name": "billions_parsing",
        "count": 5,
        "prompt": """Generate a financial paragraph for a LARGE company where revenue is stated in billions (e.g., "$4.2 billion" or "$12.8B"). Include at least 3 different numbers stated in billions.

CRITICAL: In the extraction, convert ALL numbers to millions. "$4.2 billion" = 4200, "$12.8B" = 12800. Do NOT use decimals for billion-scale numbers in the extraction.
Sector: {sector}""",
    },
    {
        "name": "mixed_billions_millions",
        "count": 5,
        "prompt": """Generate a financial paragraph where some numbers are in billions (revenue) and others in millions (operating income, cash flow). For example, revenue of "$8.5 billion" but operating income of "$342 million."

CRITICAL: In the extraction, ALL numbers must be in millions. "$8.5 billion" = 8500. "$342 million" = 342.
Sector: {sector}""",
    },
    # --- Target 3: Null discipline ---
    {
        "name": "null_inferable_revenue",
        "count": 5,
        "prompt": """Generate a financial paragraph that states current revenue and a growth percentage, but does NOT state the prior period revenue number. For example: "Revenue grew 15% to $450 million."

CRITICAL: Even though prior revenue COULD be calculated (450/1.15 = ~391), the extraction MUST have total_revenue_prior_m as null because the text does not explicitly state it. Only extract what is directly stated.
Sector: {sector}""",
    },
    {
        "name": "null_inferable_margin",
        "count": 5,
        "prompt": """Generate a financial paragraph that states gross profit dollars and revenue, but does NOT state gross margin as a percentage. For example: "Revenue was $500M with gross profit of $175M."

CRITICAL: Even though margin COULD be calculated (175/500 = 35%), the extraction MUST have gross_margin_pct as null because the text does not explicitly state it as a percentage. Only extract what is directly stated.
Sector: {sector}""",
    },
    {
        "name": "null_missing_fields",
        "count": 5,
        "prompt": """Generate a short financial paragraph (2-3 sentences) that ONLY mentions revenue and EPS. Nothing else — no margins, no cash flow, no segments, no guidance.

CRITICAL: Every field except company, period, total_revenue_m, revenue_yoy_pct (if mentioned), and eps MUST be null or empty arrays. Do not infer or calculate any missing fields.
Sector: {sector}""",
    },
]

SECTORS = [
    "Enterprise SaaS", "Retail", "Biotech/Pharma", "Industrial Manufacturing",
    "Banking/Financial Services", "Healthcare Services", "Semiconductors",
    "Consumer Packaged Goods", "Telecom", "Energy/Oil & Gas",
    "Aerospace/Defense", "Payments/Fintech", "Cloud Infrastructure",
    "Medical Devices", "Cybersecurity",
]


def generate_example(target: dict, index: int) -> dict:
    """Generate one targeted training example."""
    sector = SECTORS[index % len(SECTORS)]
    prompt = target["prompt"].format(sector=sector)

    system = BASE_SYSTEM_PROMPT.format(schema=SCHEMA_BLOCK)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text

    try:
        text_part = raw_text.split("TEXT:")[1].split("EXTRACTION:")[0].strip()
        json_part = raw_text.split("EXTRACTION:")[1].strip()

        if json_part.startswith("```json"):
            json_part = json_part[7:]
        if json_part.startswith("```"):
            json_part = json_part[3:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()

        extraction = json.loads(json_part)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"\n  WARNING: Failed to parse: {e}")
        return None

    return {
        "conversations": [
            {
                "role": "system",
                "content": "You are a financial data extraction assistant. Given a block of financial text, extract all available financial data into a structured JSON format. Use null for any fields not mentioned or not inferable from the text. Be precise — every number must match the source text exactly."
            },
            {
                "role": "user",
                "content": f"Extract the financial data from the following text:\n\n{text_part}"
            },
            {
                "role": "assistant",
                "content": json.dumps(extraction, indent=2)
            }
        ],
        "metadata": {
            "sector": sector,
            "target": target["name"],
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "version": VERSION,
        },
        "ground_truth": extraction,
        "source_text": text_part,
    }


def main():
    output_path = f"data/raw/extraction_targeted_{VERSION}.jsonl"
    examples = []
    failures = 0

    total = sum(t["count"] for t in TARGETS)
    print(f"Generating {total} targeted training examples...")
    print(f"Version: {VERSION}")
    print(f"Model: {MODEL}")
    print()

    example_index = 0
    for target in TARGETS:
        print(f"\n  Target: {target['name']} ({target['count']} examples)")
        for i in range(target["count"]):
            sector = SECTORS[example_index % len(SECTORS)]
            print(f"    [{example_index+1}/{total}] {sector}...", end=" ", flush=True)

            example = generate_example(target, example_index)
            if example:
                examples.append(example)
                print("done")
            else:
                failures += 1
                print("FAILED")

            example_index += 1

    # Write
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(examples)} examples to {output_path}")
    print(f"Failed: {failures}")

    # Save config
    config = {
        "version": VERSION,
        "model": MODEL,
        "targets": [{"name": t["name"], "count": t["count"], "prompt": t["prompt"]} for t in TARGETS],
        "num_generated": len(examples),
        "num_failed": failures,
        "generated_at": datetime.now().isoformat(),
    }
    config_path = f"runs/config_extraction_targeted_{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Summary by target
    print("\nExamples by target:")
    for target in TARGETS:
        count = sum(1 for e in examples if e["metadata"]["target"] == target["name"])
        print(f"  {target['name']}: {count}")


if __name__ == "__main__":
    main()
