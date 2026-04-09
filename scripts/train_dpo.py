"""
DPO (Direct Preference Optimization) training on top of an SFT model.

This takes a model that already knows HOW to do the task (from SFT)
and teaches it WHICH style of output is preferred.

Requires preference pairs: each example has a prompt, a chosen (good)
response, and a rejected (bad) response.

This script runs on a GPU machine (RunPod), not locally.

Usage: python scripts/train_dpo.py
"""

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# Patch DPO trainer for unsloth speed optimizations
PatchDPOTrainer()

# --- Model Configuration ---

# We start from the BASE model, not the SFT adapter.
# DPO training applies LoRA on top of the base model
# and uses the base model as the "reference" to measure
# how much the model's preferences are shifting.
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

MAX_SEQ_LENGTH = 2048

# --- LoRA Configuration ---
# Same structure as SFT, but these are NEW adapter weights
# specifically for preference learning.
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# --- DPO Configuration ---

# Beta controls how much the model is allowed to deviate from
# the base model's behavior. Think of it as a "leash length."
#   - High beta (0.5+): short leash, model stays close to base behavior
#   - Low beta (0.01-0.05): long leash, model can change a lot
#   - Default (0.1): balanced — learns preferences without going off the rails
DPO_BETA = 0.1

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-5  # Lower than SFT — DPO needs gentler updates

OUTPUT_DIR = "outputs/dpo_final"


def main():
    print("=" * 60)
    print("CRUCIBLE — DPO Training Pipeline")
    print("=" * 60)

    # Step 1: Load the base model
    print(f"\n[1/4] Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Attach LoRA adapters
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

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Step 2: Load preference data
    print("\n[2/4] Loading preference data...")
    dataset = load_dataset(
        "json",
        data_files="data/preferences/dpo_v1.jsonl",
        split="train",
    )
    print(f"  Loaded {len(dataset)} preference pairs")

    # Format into what DPOTrainer expects.
    # DPO needs three fields: prompt, chosen, rejected.
    # Each must be formatted as a full chat conversation.
    def format_for_dpo(batch):
        formatted_prompts = []
        formatted_chosen = []
        formatted_rejected = []

        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            # The system message that frames the task
            system_msg = "You are a financial analyst specializing in revenue trend analysis. Provide direct, quantified analysis focused on what matters for the investment case."

            # Format prompt as chat messages
            prompt_msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]

            # Format chosen response as chat messages
            chosen_msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]

            # Format rejected response as chat messages
            rejected_msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected},
            ]

            # Apply chat template to convert to token format
            formatted_prompts.append(
                tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            )
            formatted_chosen.append(
                tokenizer.apply_chat_template(chosen_msgs, tokenize=False, add_generation_prompt=False)
            )
            formatted_rejected.append(
                tokenizer.apply_chat_template(rejected_msgs, tokenize=False, add_generation_prompt=False)
            )

        return {
            "prompt": formatted_prompts,
            "chosen": formatted_chosen,
            "rejected": formatted_rejected,
        }

    dataset = dataset.map(format_for_dpo, batched=True)

    # Step 3: Train with DPO
    print(f"\n[3/4] DPO training for {NUM_EPOCHS} epochs...")
    print(f"  Beta: {DPO_BETA}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            beta=DPO_BETA,
            logging_steps=1,
            save_strategy="epoch",
            optim="adamw_8bit",
            seed=42,
            bf16=True,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=1024,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Step 4: Save
    print(f"\n[4/4] Saving DPO adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone. DPO adapter saved.")


if __name__ == "__main__":
    main()
