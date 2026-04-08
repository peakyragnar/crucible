"""
Fine-tune a base model using unsloth with LoRA.

This script runs on a GPU machine (RunPod), not locally.

Usage: python scripts/train.py

What this does:
1. Loads a base model (Llama 3.1 8B) with unsloth (fast LoRA setup)
2. Loads our curated training data
3. Trains a LoRA adapter on it
4. Saves the adapter weights

Key concepts:
- LoRA: Instead of updating all 8 billion parameters, we freeze the base
  model and train small adapter layers (~1-2% of total parameters).
  This is faster, cheaper, and needs less VRAM.
- QLoRA: Same as LoRA but the base model is loaded in 4-bit precision.
  Uses even less VRAM. Quality difference is minimal for our use case.
- SFT: Supervised Fine-Tuning. We show the model input/output pairs
  and train it to produce the outputs given the inputs.
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- Model Configuration ---
# These are the knobs you can turn. Defaults are conservative and safe.

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# Why this model: Llama 3.1 8B Instruct is a strong general-purpose model.
# The "bnb-4bit" variant is pre-quantized for QLoRA — uses ~5GB VRAM.
# "Instruct" means it already knows how to follow instructions,
# so we're refining its behavior, not teaching it from scratch.

MAX_SEQ_LENGTH = 2048
# Maximum tokens per training example. Our examples are ~500-800 tokens,
# so 2048 gives plenty of headroom.

# --- LoRA Configuration ---
LORA_RANK = 16
# How many parameters the adapter adds. Higher = more capacity to learn,
# but also more risk of overfitting on small datasets.
# 16 is a safe default. Range: 8 (minimal) to 64 (aggressive).

LORA_ALPHA = 16
# Scaling factor for LoRA. Usually set equal to rank.
# Higher alpha = adapter has more influence on output.

LORA_DROPOUT = 0
# Dropout for regularization. 0 is fine with unsloth's optimizations.

# --- Training Configuration ---
NUM_EPOCHS = 3
# How many times to train on the full dataset.
# With small datasets (<100 examples): 3-5 epochs is typical.
# Too many epochs = the model memorizes instead of learning patterns.

BATCH_SIZE = 2
# How many examples to process at once. Limited by VRAM.
# 2 is safe for 24GB VRAM with 8B model.

GRADIENT_ACCUMULATION = 4
# Simulates a larger batch size without using more VRAM.
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION = 8

LEARNING_RATE = 2e-4
# How big each update step is. This is the standard for LoRA fine-tuning.
# Too high = training is unstable. Too low = model barely changes.

WARMUP_STEPS = 5
# Gradually ramp up learning rate at the start. Prevents early instability.

OUTPUT_DIR = "outputs"
# Where to save the trained adapter and checkpoints.


def main():
    print("=" * 60)
    print("CRUCIBLE — Fine-Tuning Pipeline")
    print("=" * 60)

    # Step 1: Load the base model with LoRA adapters attached
    print(f"\n[1/4] Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Attach LoRA adapters to the model
    # target_modules: which layers of the model get LoRA adapters.
    # These are the standard attention and feed-forward layers.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Set up the chat template so the model knows how to parse conversations
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Step 2: Load training data
    print("\n[2/4] Loading training data...")
    dataset = load_dataset(
        "json",
        data_files="data/curated/revenue_trend_analysis_v1.jsonl",
        split="train",
    )
    print(f"  Loaded {len(dataset)} training examples")

    # Format conversations into the token format the model expects
    def format_conversations(batch):
        texts = []
        for convos in batch["conversations"]:
            # Convert from our format to chat messages
            messages = []
            for turn in convos:
                messages.append({
                    "role": "system" if turn["from"] == "system" else turn["from"],
                    "content": turn["value"],
                })
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_conversations, batched=True)

    # Step 3: Train
    print(f"\n[3/4] Training for {NUM_EPOCHS} epochs...")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            seed=42,
            fp16=True,
        ),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    trainer.train()

    # Step 4: Save the adapter
    print(f"\n[4/4] Saving adapter to {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\nDone. Adapter saved.")
    print(f"To test: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
