"""Quick evaluation of fine-tuned model only (no HuggingFace connection needed)."""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json

print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/final",
    max_seq_length=2048,
    load_in_4bit=True,
    local_files_only=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

tests = [
    {"name": "AMD - cash flow fields", "text": "Advanced Micro Devices reported third quarter 2024 revenue of $6.8 billion, an increase of 18% year-over-year from $5.8 billion. Data Center segment revenue was $3.5 billion, up 122% year-over-year. Client segment revenue increased 29% to $1.9 billion. Gaming segment revenue declined 69% to $462 million. Gross margin was 54%, up from 51% in the prior year quarter. Operating income was $1.7 billion. Net income was $1.5 billion, or $0.92 per diluted share. The company generated $628 million in free cash flow during the quarter."},
    {"name": "Vertex - null discipline", "text": "Vertex Pharmaceuticals announced that total net product revenues for the first quarter were $2.6 billion, representing growth of 12% compared to the first quarter of the prior year. The company reaffirmed its full-year 2025 net product revenue guidance of $11.3 billion to $11.6 billion."},
    {"name": "Dollar General - billions parsing", "text": "Dollar General Corporation reported net sales of $9.9 billion for the second quarter of fiscal 2024, an increase of 4.2% from $9.5 billion in the prior year period. Gross profit as a percentage of net sales was 30.0% compared to 31.1% in the prior year, a contraction of 110 basis points. Operating profit was $550 million. Diluted earnings per share were $1.70."},
]

system = "You are a financial data extraction assistant. Given a block of financial text, extract all available financial data into a structured JSON format. Use null for any fields not mentioned or not inferable from the text. Be precise - every number must match the source text exactly."

for t in tests:
    print("\n" + "=" * 60)
    print("TEST: " + t["name"])
    print("=" * 60)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Extract the financial data from the following text:\n\n" + t["text"]},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=1000, temperature=0.1, top_p=0.9
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(response)
