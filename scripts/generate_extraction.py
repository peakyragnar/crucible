"""
Generate training data for financial data extraction.

Creates realistic financial text paragraphs paired with structured
JSON extractions. The model learns to pull specific data points
from unstructured text — a task where correctness is measurable.

Usage: python scripts/generate_extraction.py
"""

import anthropic
import json
import hashlib
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"
VERSION = "v1"
NUM_EXAMPLES = 50

# --- Extraction schema ---
# These are the fields the model should extract from any financial text.
# null means "not mentioned in the text."
SCHEMA_DESCRIPTION = """
{
  "company": "string — company name",
  "period": "string — fiscal period (e.g., 'FY 2024', 'Q3 2024')",
  "total_revenue_m": "number or null — total revenue in millions",
  "total_revenue_prior_m": "number or null — prior period revenue in millions",
  "revenue_yoy_pct": "number or null — year-over-year revenue growth percentage",
  "segments": [
    {"name": "string", "revenue_m": "number", "yoy_pct": "number or null"}
  ],
  "revenue_by_type": [
    {"name": "string", "revenue_m": "number", "yoy_pct": "number or null"}
  ],
  "gross_margin_pct": "number or null",
  "gross_margin_change_bps": "number or null — positive = expansion, negative = contraction",
  "operating_income_m": "number or null",
  "operating_margin_pct": "number or null",
  "net_income_m": "number or null",
  "eps": "number or null — earnings per share",
  "cash_from_operations_m": "number or null",
  "free_cash_flow_m": "number or null",
  "guidance_revenue_m": "string or null — forward guidance if mentioned",
  "flags": ["string — notable items, risks, or anomalies worth flagging"]
}
"""

# System prompt for generating the training examples
GENERATION_PROMPT = """You are generating training data for a financial data extraction model.

Your task: create a realistic paragraph of financial text (like you'd find in a 10-K, 10-Q,
earnings release, or analyst report) AND the correct structured JSON extraction from that text.

Requirements for the text:
- Make it sound like real financial writing — dense, specific, with actual numbers
- Vary the style: some formal (SEC filing), some conversational (earnings call), some analytical (research note)
- Vary the sector and company type
- Vary which data points are present — NOT every text should have every field
- Some numbers should be stated directly ("revenue of $450M")
- Some should require inference ("revenue grew 15% to $450M" — prior was ~$391M)
- Some fields should be genuinely absent from the text
- Include realistic noise: management commentary, qualitative statements mixed with numbers
- Length: 100-250 words per paragraph

Requirements for the JSON extraction:
- You MUST use EXACTLY these field names — no variations, no renaming, no additions:

{
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
}

- Every response MUST include ALL of these fields. Use null for fields not mentioned in the text.
- Use empty arrays [] for segments and revenue_by_type if not mentioned.
- Use empty array [] for flags if nothing notable.
- Numbers in millions. Percentages as numbers (17 not "17%"). Basis points as integers (-180 not "-1.8%").
- IMPORTANT: The JSON must be perfectly accurate. Every number must match the text exactly.
- If a number requires inference, show the correct inferred value.
"""

USER_PROMPT_TEMPLATE = """Generate training example {n} of {total}.

Requirements for this specific example:
- Sector: {sector}
- Style: {style}
- Complexity: {complexity}
- Missing fields: {missing}

Return your response in this exact format:

TEXT:
[the financial paragraph]

EXTRACTION:
[the JSON extraction]
"""

# Vary these across examples to get diverse training data
SECTORS = [
    "Enterprise SaaS", "Retail", "Biotech/Pharma", "Industrial Manufacturing",
    "Banking/Financial Services", "Healthcare Services", "Semiconductors",
    "Consumer Packaged Goods", "Telecom", "Energy/Oil & Gas",
    "Real Estate/REITs", "Media/Entertainment", "Insurance",
    "Aerospace/Defense", "Restaurants/Hospitality", "Auto/Transportation",
    "Agricultural Products", "Specialty Chemicals", "E-commerce",
    "Cybersecurity", "Medical Devices", "Payments/Fintech",
    "Cloud Infrastructure", "Education Technology", "Waste Management",
]

STYLES = [
    "10-K annual report filing",
    "10-Q quarterly filing",
    "Earnings press release",
    "Earnings call transcript excerpt",
    "Sell-side analyst report excerpt",
    "Management discussion and analysis (MD&A)",
]

COMPLEXITIES = [
    "simple — 3-4 data points, clearly stated",
    "moderate — 5-7 data points, some require inference",
    "complex — 8+ data points, multiple segments, some inference required",
]

MISSING_FIELDS = [
    "no segment data, no guidance",
    "no margin data",
    "no cash flow data, no EPS",
    "no prior period comparisons",
    "no segment data, no revenue by type",
    "has everything — comprehensive disclosure",
    "minimal — just revenue and one other metric",
    "no operating income, no net income",
]


def generate_example(n: int, total: int) -> dict:
    """Generate one training example."""
    # Cycle through variations to ensure diversity
    sector = SECTORS[n % len(SECTORS)]
    style = STYLES[n % len(STYLES)]
    complexity = COMPLEXITIES[n % len(COMPLEXITIES)]
    missing = MISSING_FIELDS[n % len(MISSING_FIELDS)]

    prompt = USER_PROMPT_TEMPLATE.format(
        n=n + 1, total=total,
        sector=sector, style=style,
        complexity=complexity, missing=missing,
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=GENERATION_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text

    # Parse the response into text and JSON
    try:
        text_part = raw_text.split("TEXT:")[1].split("EXTRACTION:")[0].strip()
        json_part = raw_text.split("EXTRACTION:")[1].strip()

        # Clean up JSON — remove markdown code fences if present
        json_part = json_part.strip()
        if json_part.startswith("```json"):
            json_part = json_part[7:]
        if json_part.startswith("```"):
            json_part = json_part[3:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()

        extraction = json.loads(json_part)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"\n  WARNING: Failed to parse example {n+1}: {e}")
        return None

    # Format as training conversation
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
            "style": style,
            "complexity": complexity,
            "missing_fields": missing,
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "version": VERSION,
        },
        # Store separately for automated scoring later
        "ground_truth": extraction,
        "source_text": text_part,
    }


def save_run_config(run_dir: str, num_generated: int, num_failed: int) -> None:
    """Save config snapshot for this run."""
    config = {
        "version": VERSION,
        "task": "financial_data_extraction",
        "model": MODEL,
        "generation_prompt": GENERATION_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "schema": SCHEMA_DESCRIPTION,
        "num_requested": NUM_EXAMPLES,
        "num_generated": num_generated,
        "num_failed": num_failed,
        "sectors": SECTORS,
        "styles": STYLES,
        "complexities": COMPLEXITIES,
        "generated_at": datetime.now().isoformat(),
        "input_hash": hashlib.md5(
            json.dumps({"prompt": GENERATION_PROMPT, "schema": SCHEMA_DESCRIPTION},
                       sort_keys=True).encode()
        ).hexdigest(),
    }

    config_path = f"{run_dir}/config_extraction_{VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Run config saved to {config_path}")


def main():
    output_path = f"data/raw/extraction_{VERSION}.jsonl"
    run_dir = "runs"
    examples = []
    failures = 0

    print(f"Generating {NUM_EXAMPLES} extraction training examples...")
    print(f"Version: {VERSION}")
    print(f"Model: {MODEL}")
    print()

    for i in range(NUM_EXAMPLES):
        sector = SECTORS[i % len(SECTORS)]
        print(f"  [{i+1}/{NUM_EXAMPLES}] {sector}...", end=" ", flush=True)

        example = generate_example(i, NUM_EXAMPLES)
        if example:
            examples.append(example)
            print("done")
        else:
            failures += 1
            print("FAILED")

    # Write training data
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(examples)} examples to {output_path}")
    print(f"Failed: {failures}")

    # Save config
    save_run_config(run_dir, len(examples), failures)

    # Cost estimate: ~$0.01-0.02 per example with Sonnet
    print(f"Estimated API cost: ~${len(examples) * 0.015:.2f}")

    # Show a sample
    if examples:
        sample = examples[0]
        print("\n" + "=" * 60)
        print("SAMPLE EXAMPLE:")
        print("=" * 60)
        print(f"\nSector: {sample['metadata']['sector']}")
        print(f"Style: {sample['metadata']['style']}")
        print(f"\nTEXT:\n{sample['source_text'][:300]}...")
        print(f"\nEXTRACTION:\n{json.dumps(sample['ground_truth'], indent=2)[:500]}...")


if __name__ == "__main__":
    main()
