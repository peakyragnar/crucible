"""
Generate training data for revenue trend analysis.

Uses Claude to create high-quality (prompt, response) pairs
that demonstrate how a skilled analyst thinks about revenue trends.

Output: JSONL file where each line is a conversation in chat format.
"""

import anthropic
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

# --- Revenue scenarios to generate training data for ---
# Each scenario gives Claude enough context to produce a realistic,
# specific financial analysis. We vary the patterns so the fine-tuned
# model learns to handle different situations, not just one template.

SCENARIOS = [
    {
        "company": "CloudServe Inc.",
        "sector": "Enterprise SaaS",
        "revenue_data": {
            "Q1 2023": 142, "Q2 2023": 156, "Q3 2023": 168, "Q4 2023": 185,
            "Q1 2024": 198, "Q2 2024": 215, "Q3 2024": 228, "Q4 2024": 245
        },
        "context": "Subscription-based model, 90% recurring revenue. Shifted from on-prem licenses to cloud in 2022. Net retention rate 118%. Sales cycles lengthening from 45 to 62 days.",
        "pattern": "steady_growth_decelerating"
    },
    {
        "company": "MedDevice Corp.",
        "sector": "Medical Devices",
        "revenue_data": {
            "Q1 2023": 320, "Q2 2023": 315, "Q3 2023": 298, "Q4 2023": 340,
            "Q1 2024": 355, "Q2 2024": 348, "Q3 2024": 310, "Q4 2024": 380
        },
        "context": "Sells surgical instruments and disposables. Q4 is seasonally strong (hospital budget flush). New product line launched Q1 2024. Reimbursement rates under pressure from CMS.",
        "pattern": "seasonal_with_growth"
    },
    {
        "company": "PetroChem Holdings",
        "sector": "Specialty Chemicals",
        "revenue_data": {
            "Q1 2023": 890, "Q2 2023": 845, "Q3 2023": 810, "Q4 2023": 780,
            "Q1 2024": 750, "Q2 2024": 725, "Q3 2024": 740, "Q4 2024": 760
        },
        "context": "Revenue tied to commodity pricing. Volumes flat but ASPs declining through 2023. Management guided to 'trough pricing' in Q2 2024 call. New contract wins in specialty segment starting Q3 2024.",
        "pattern": "decline_then_stabilization"
    },
    {
        "company": "QuickDeliver Logistics",
        "sector": "Last-Mile Delivery",
        "revenue_data": {
            "Q1 2023": 210, "Q2 2023": 195, "Q3 2023": 225, "Q4 2023": 310,
            "Q1 2024": 185, "Q2 2024": 178, "Q3 2024": 205, "Q4 2024": 290
        },
        "context": "Heavy Q4 seasonality from e-commerce peak. Lost a major retail client in Q1 2024. Expanding into grocery delivery. Revenue per delivery declining due to competitive pricing.",
        "pattern": "seasonal_with_client_loss"
    },
    {
        "company": "FinGuard Software",
        "sector": "RegTech / Compliance Software",
        "revenue_data": {
            "Q1 2023": 45, "Q2 2023": 52, "Q3 2023": 61, "Q4 2023": 58,
            "Q1 2024": 72, "Q2 2024": 88, "Q3 2024": 105, "Q4 2024": 94
        },
        "context": "Sells compliance automation to banks. Revenue lumpy due to large contract timing. New SEC reporting rules driving demand. Professional services revenue 30% of total, declining as product matures.",
        "pattern": "lumpy_high_growth"
    },
    {
        "company": "HomeFirst Retail",
        "sector": "Home Improvement Retail",
        "revenue_data": {
            "Q1 2023": 1250, "Q2 2023": 1480, "Q3 2023": 1520, "Q4 2023": 1180,
            "Q1 2024": 1190, "Q2 2024": 1395, "Q3 2024": 1430, "Q4 2024": 1120
        },
        "context": "Same-store sales declining. Spring/summer seasonal peak. Opened 12 new stores in 2024. Comp sales -3.2% but total revenue flat due to new stores. Ticket size up 4%, traffic down 7%.",
        "pattern": "flat_masking_decline"
    },
    {
        "company": "NovaBio Therapeutics",
        "sector": "Biotech / Early Commercial",
        "revenue_data": {
            "Q1 2023": 8, "Q2 2023": 14, "Q3 2023": 22, "Q4 2023": 31,
            "Q1 2024": 38, "Q2 2024": 52, "Q3 2024": 64, "Q4 2024": 71
        },
        "context": "First commercial product launched Q4 2022. Revenue is product sales only (no milestones). Treating rare disease with ~15,000 addressable patients in US. Payer coverage expanding. EU launch expected mid-2025.",
        "pattern": "early_launch_ramp"
    },
    {
        "company": "DataVault Systems",
        "sector": "Data Infrastructure",
        "revenue_data": {
            "Q1 2023": 180, "Q2 2023": 192, "Q3 2023": 205, "Q4 2023": 218,
            "Q1 2024": 210, "Q2 2024": 205, "Q3 2024": 215, "Q4 2024": 222
        },
        "context": "Consumption-based pricing model. Growth stalled as customers optimized cloud spend in 2024. Signed several large contracts not yet ramping. Management says 'optimization headwinds largely behind us' as of Q4 2024 call.",
        "pattern": "growth_stall_consumption"
    },
]

# The system prompt tells Claude what kind of analyst it should emulate.
# This is critical — it shapes the quality and style of every response.
SYSTEM_PROMPT = """You are a senior equity research analyst with 15+ years of experience covering public companies. You are analyzing revenue trends for a company.

Your analysis style:
- Lead with the most important insight, not a summary of the data
- Distinguish between reported numbers and underlying trends (organic vs. inorganic, volume vs. price, recurring vs. one-time)
- Quantify everything — growth rates, sequential changes, run-rates
- Flag what the numbers DON'T tell you and what you'd want to dig into
- Be direct. If the trend is bad, say so. If it's good, say why it might not last
- Think in terms of what matters for the investment case, not academic analysis
- Keep it under 400 words — this is a working analyst's note, not a research report"""


def format_revenue_table(revenue_data: dict) -> str:
    """Format revenue data as a readable table."""
    lines = ["| Quarter | Revenue ($M) |", "|---------|-------------|"]
    for quarter, revenue in revenue_data.items():
        lines.append(f"| {quarter} | {revenue} |")
    return "\n".join(lines)


def generate_user_prompt(scenario: dict) -> str:
    """Create the user question for a given scenario."""
    table = format_revenue_table(scenario["revenue_data"])

    return f"""Analyze the revenue trend for {scenario['company']} ({scenario['sector']}).

{table}

Additional context: {scenario['context']}

What's the story these numbers are telling? What would you want to dig into further?"""


def generate_example(scenario: dict) -> dict:
    """Generate one training example from a scenario."""
    user_prompt = generate_user_prompt(scenario)

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    assistant_response = response.content[0].text

    # Format as a chat conversation for SFT training
    return {
        "conversations": [
            {"role": "system", "content": "You are a financial analyst specializing in revenue trend analysis. Provide direct, quantified analysis focused on what matters for the investment case."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ],
        "metadata": {
            "company": scenario["company"],
            "sector": scenario["sector"],
            "pattern": scenario["pattern"],
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
        }
    }


def main():
    output_path = "data/raw/revenue_trend_analysis_v1.jsonl"
    examples = []

    print(f"Generating {len(SCENARIOS)} training examples...")
    print(f"Using model: {MODEL}")
    print()

    for i, scenario in enumerate(SCENARIOS):
        print(f"  [{i+1}/{len(SCENARIOS)}] {scenario['company']} ({scenario['pattern']})...", end=" ", flush=True)
        example = generate_example(scenario)
        examples.append(example)
        print("done")

    # Write to JSONL
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(examples)} examples to {output_path}")
    print(f"\nEstimated API cost: ~${len(examples) * 0.01:.2f}")

    # Print one example so you can review quality
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (first example):")
    print("="*60)
    print(f"\nCompany: {examples[0]['metadata']['company']}")
    print(f"Pattern: {examples[0]['metadata']['pattern']}")
    print(f"\nUSER PROMPT:\n{examples[0]['conversations'][1]['content'][:200]}...")
    print(f"\nANALYST RESPONSE:\n{examples[0]['conversations'][2]['content']}")


if __name__ == "__main__":
    main()
