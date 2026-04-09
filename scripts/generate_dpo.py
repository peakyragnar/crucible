"""
Generate DPO (preference) training data for revenue trend analysis.

For each scenario, generates:
- A "chosen" response: direct, quantified, insight-first analyst writing (from Claude)
- A "rejected" response: generic, hedging, summary-style writing (from Claude, prompted to be bad)

Usage: uv run python scripts/generate_dpo.py
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

# The good analyst prompt — same as our SFT training
CHOSEN_SYSTEM_PROMPT = """You are a senior equity research analyst with 15+ years of experience covering public companies. You are analyzing revenue trends for a company.

Your analysis style:
- Lead with the most important insight, not a summary of the data
- Distinguish between reported numbers and underlying trends (organic vs. inorganic, volume vs. price, recurring vs. one-time)
- Quantify everything — growth rates, sequential changes, run-rates
- Flag what the numbers DON'T tell you and what you'd want to dig into
- Be direct. If the trend is bad, say so. If it's good, say why it might not last
- Think in terms of what matters for the investment case, not academic analysis
- Keep it under 400 words"""

# The bad analyst prompt — deliberately produces the kind of output we want to suppress
REJECTED_SYSTEM_PROMPT = """You are a junior financial analyst writing a generic summary. Your style:
- Start with "Key Takeaways" or "Revenue Analysis" as a header
- Summarize the data in bullet points, restating the numbers from the table
- Hedge everything: "could potentially", "may indicate", "it remains to be seen"
- Don't calculate growth rates yourself — just mention if things went up or down
- End with generic recommendations like "monitor closely" or "further analysis is needed"
- Be balanced to the point of saying nothing — every negative has an equal positive
- Don't take a position on what the numbers mean for the investment case
- Include a "Positive Trends" and "Negative Trends" section
- Keep it under 400 words"""

USER_PROMPT_TEMPLATE = """Analyze the revenue trend for {company} ({sector}).

{revenue_table}

Additional context: {context}

What's the story these numbers are telling? What would you want to dig into further?"""

SCENARIOS = [
    {
        "company": "CloudServe Inc.",
        "sector": "Enterprise SaaS",
        "revenue_data": {
            "Q1 2023": 142, "Q2 2023": 156, "Q3 2023": 168, "Q4 2023": 185,
            "Q1 2024": 198, "Q2 2024": 215, "Q3 2024": 228, "Q4 2024": 245
        },
        "context": "Subscription-based model, 90% recurring revenue. Shifted from on-prem licenses to cloud in 2022. Net retention rate 118%. Sales cycles lengthening from 45 to 62 days."
    },
    {
        "company": "MedDevice Corp.",
        "sector": "Medical Devices",
        "revenue_data": {
            "Q1 2023": 320, "Q2 2023": 315, "Q3 2023": 298, "Q4 2023": 340,
            "Q1 2024": 355, "Q2 2024": 348, "Q3 2024": 310, "Q4 2024": 380
        },
        "context": "Sells surgical instruments and disposables. Q4 is seasonally strong (hospital budget flush). New product line launched Q1 2024. Reimbursement rates under pressure from CMS."
    },
    {
        "company": "PetroChem Holdings",
        "sector": "Specialty Chemicals",
        "revenue_data": {
            "Q1 2023": 890, "Q2 2023": 845, "Q3 2023": 810, "Q4 2023": 780,
            "Q1 2024": 750, "Q2 2024": 725, "Q3 2024": 740, "Q4 2024": 760
        },
        "context": "Revenue tied to commodity pricing. Volumes flat but ASPs declining through 2023. Management guided to 'trough pricing' in Q2 2024 call. New contract wins in specialty segment starting Q3 2024."
    },
    {
        "company": "QuickDeliver Logistics",
        "sector": "Last-Mile Delivery",
        "revenue_data": {
            "Q1 2023": 210, "Q2 2023": 195, "Q3 2023": 225, "Q4 2023": 310,
            "Q1 2024": 185, "Q2 2024": 178, "Q3 2024": 205, "Q4 2024": 290
        },
        "context": "Heavy Q4 seasonality from e-commerce peak. Lost a major retail client in Q1 2024. Expanding into grocery delivery. Revenue per delivery declining due to competitive pricing."
    },
    {
        "company": "FinGuard Software",
        "sector": "RegTech / Compliance Software",
        "revenue_data": {
            "Q1 2023": 45, "Q2 2023": 52, "Q3 2023": 61, "Q4 2023": 58,
            "Q1 2024": 72, "Q2 2024": 88, "Q3 2024": 105, "Q4 2024": 94
        },
        "context": "Sells compliance automation to banks. Revenue lumpy due to large contract timing. New SEC reporting rules driving demand. Professional services revenue 30% of total, declining as product matures."
    },
    {
        "company": "HomeFirst Retail",
        "sector": "Home Improvement Retail",
        "revenue_data": {
            "Q1 2023": 1250, "Q2 2023": 1480, "Q3 2023": 1520, "Q4 2023": 1180,
            "Q1 2024": 1190, "Q2 2024": 1395, "Q3 2024": 1430, "Q4 2024": 1120
        },
        "context": "Same-store sales declining. Spring/summer seasonal peak. Opened 12 new stores in 2024. Comp sales -3.2% but total revenue flat due to new stores. Ticket size up 4%, traffic down 7%."
    },
    {
        "company": "NovaBio Therapeutics",
        "sector": "Biotech / Early Commercial",
        "revenue_data": {
            "Q1 2023": 8, "Q2 2023": 14, "Q3 2023": 22, "Q4 2023": 31,
            "Q1 2024": 38, "Q2 2024": 52, "Q3 2024": 64, "Q4 2024": 71
        },
        "context": "First commercial product launched Q4 2022. Revenue is product sales only (no milestones). Treating rare disease with ~15,000 addressable patients in US. Payer coverage expanding. EU launch expected mid-2025."
    },
    {
        "company": "DataVault Systems",
        "sector": "Data Infrastructure",
        "revenue_data": {
            "Q1 2023": 180, "Q2 2023": 192, "Q3 2023": 205, "Q4 2023": 218,
            "Q1 2024": 210, "Q2 2024": 205, "Q3 2024": 215, "Q4 2024": 222
        },
        "context": "Consumption-based pricing model. Growth stalled as customers optimized cloud spend in 2024. Signed several large contracts not yet ramping. Management says 'optimization headwinds largely behind us' as of Q4 2024 call."
    },
    {
        "company": "Pinnacle Insurance Group",
        "sector": "Specialty Insurance",
        "revenue_data": {
            "Q1 2023": 520, "Q2 2023": 545, "Q3 2023": 610, "Q4 2023": 580,
            "Q1 2024": 590, "Q2 2024": 625, "Q3 2024": 710, "Q4 2024": 660
        },
        "context": "Net premiums written growing 14% YoY. Combined ratio improved from 96% to 93%. Reserve releases contributed $40M in Q3 2024. Cat losses minimal in 2024. Competitor exiting E&S lines creating pricing opportunity."
    },
    {
        "company": "GreenHarvest AgTech",
        "sector": "Agricultural Technology",
        "revenue_data": {
            "Q1 2023": 35, "Q2 2023": 82, "Q3 2023": 95, "Q4 2023": 28,
            "Q1 2024": 42, "Q2 2024": 98, "Q3 2024": 115, "Q4 2024": 33
        },
        "context": "Sells precision agriculture subscriptions and hardware. Revenue extremely seasonal — planting season (Q2-Q3) drives 75% of annual revenue. Hardware margins 15%, software margins 80%. Transitioning to subscription model."
    },
    {
        "company": "TechBridge Semiconductors",
        "sector": "Analog Semiconductors",
        "revenue_data": {
            "Q1 2023": 410, "Q2 2023": 385, "Q3 2023": 362, "Q4 2023": 345,
            "Q1 2024": 330, "Q2 2024": 338, "Q3 2024": 355, "Q4 2024": 370
        },
        "context": "Automotive end-market 45% of revenue. Industrial 30%, consumer 25%. Inventory correction ongoing through 2023. Book-to-bill ratio crossed 1.0 in Q3 2024. Lead times normalizing to 8-10 weeks from peak of 52 weeks."
    },
    {
        "company": "MetroHealth Systems",
        "sector": "Hospital Operations",
        "revenue_data": {
            "Q1 2023": 2100, "Q2 2023": 2150, "Q3 2023": 2180, "Q4 2023": 2220,
            "Q1 2024": 2280, "Q2 2024": 2350, "Q3 2024": 2310, "Q4 2024": 2400
        },
        "context": "Same-hospital revenue growth 5%. Acquired two facilities in Q2 2024 adding $120M quarterly. Labor costs 52% of revenue, down from 55% as travel nurse usage normalizes. Medicaid redetermination impacting payer mix. Case mix index improving."
    },
    {
        "company": "SwiftPay Financial",
        "sector": "Payments Processing",
        "revenue_data": {
            "Q1 2023": 165, "Q2 2023": 178, "Q3 2023": 185, "Q4 2023": 210,
            "Q1 2024": 195, "Q2 2024": 208, "Q3 2024": 218, "Q4 2024": 248
        },
        "context": "Take rate declining from 2.8% to 2.5% due to enterprise mix shift. Payment volume growing 22% but revenue only growing 15%. Won three Fortune 500 clients in H2 2024. International expansion into LatAm beginning Q1 2025."
    },
    {
        "company": "Atlas Mining Corp",
        "sector": "Copper Mining",
        "revenue_data": {
            "Q1 2023": 680, "Q2 2023": 720, "Q3 2023": 750, "Q4 2023": 710,
            "Q1 2024": 810, "Q2 2024": 850, "Q3 2024": 920, "Q4 2024": 880
        },
        "context": "Revenue driven by copper price which rose 28% YoY. Production volumes flat at 180kt. New mine (Cerro Verde expansion) expected to add 40kt starting H2 2025. All-in sustaining cost $2.85/lb vs. current price $4.20/lb. Chinese demand uncertain."
    },
    {
        "company": "BrightEdge Solar",
        "sector": "Residential Solar",
        "revenue_data": {
            "Q1 2023": 280, "Q2 2023": 310, "Q3 2023": 295, "Q4 2023": 240,
            "Q1 2024": 195, "Q2 2024": 210, "Q3 2024": 225, "Q4 2024": 200
        },
        "context": "Revenue declined 24% YoY as interest rates crushed residential solar demand. Shifted from cash sales to lease/PPA model — lower upfront revenue but higher lifetime value. Customer acquisition cost up 35%. ITC extension provides policy certainty through 2032."
    },
]


def format_revenue_table(revenue_data: dict) -> str:
    lines = ["| Quarter | Revenue ($M) |", "|---------|-------------|"]
    for quarter, revenue in revenue_data.items():
        lines.append(f"| {quarter} | {revenue} |")
    return "\n".join(lines)


def generate_pair(scenario: dict) -> dict:
    """Generate a chosen/rejected pair for one scenario."""
    table = format_revenue_table(scenario["revenue_data"])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company=scenario["company"],
        sector=scenario["sector"],
        revenue_table=table,
        context=scenario["context"],
    )

    # Generate chosen (good) response
    chosen_response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=CHOSEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Generate rejected (bad) response
    rejected_response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=REJECTED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return {
        "prompt": user_prompt,
        "chosen": chosen_response.content[0].text,
        "rejected": rejected_response.content[0].text,
        "metadata": {
            "company": scenario["company"],
            "sector": scenario["sector"],
            "generated_at": datetime.now().isoformat(),
            "model": MODEL,
            "version": VERSION,
        }
    }


def main():
    output_path = f"data/preferences/dpo_{VERSION}.jsonl"

    print(f"Generating {len(SCENARIOS)} DPO preference pairs...")
    print(f"Version: {VERSION}")
    print(f"Model: {MODEL}")
    print()

    examples = []
    for i, scenario in enumerate(SCENARIOS):
        print(f"  [{i+1}/{len(SCENARIOS)}] {scenario['company']}...", end=" ", flush=True)
        pair = generate_pair(scenario)
        examples.append(pair)
        print("done")

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nWrote {len(examples)} preference pairs to {output_path}")
    print(f"Estimated API cost: ~${len(examples) * 0.03:.2f}")

    # Show one pair so you can see the contrast
    print("\n" + "=" * 60)
    print("SAMPLE PAIR:")
    print("=" * 60)
    sample = examples[0]
    print(f"\nCompany: {sample['metadata']['company']}")
    print(f"\n--- CHOSEN (good) ---")
    print(sample["chosen"][:500])
    print(f"\n--- REJECTED (bad) ---")
    print(sample["rejected"][:500])


if __name__ == "__main__":
    main()
