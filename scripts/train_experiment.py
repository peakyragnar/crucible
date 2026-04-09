"""
Run training experiments with different parameters.

Change one parameter at a time, measure the impact.
Results are saved to runs/ for comparison.

Usage:
  python scripts/train_experiment.py                          # baseline defaults
  python scripts/train_experiment.py --epochs 5               # more epochs
  python scripts/train_experiment.py --lora-rank 32           # bigger adapter
  python scripts/train_experiment.py --lr 5e-5                # lower learning rate
  python scripts/train_experiment.py --name "my_experiment"   # custom name

This script runs on a GPU machine (RunPod).
"""

import argparse
import json
from datetime import datetime

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Training experiment")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--data", type=str, default="data/curated/extraction_v2.jsonl", help="Training data path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Generate experiment name if not provided
    if args.name is None:
        args.name = f"e{args.epochs}_r{args.lora_rank}_lr{args.lr}"

    output_dir = f"outputs/experiment_{args.name}"

    print("=" * 60)
    print(f"EXPERIMENT: {args.name}")
    print("=" * 60)
    print(f"  Epochs:         {args.epochs}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Grad accum:     {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Data:           {args.data}")
    print(f"  Output:         {output_dir}")

    # Step 1: Load model
    print(f"\n[1/4] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # Keep alpha = rank
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Step 2: Load data
    print("\n[2/4] Loading training data...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"  Loaded {len(dataset)} examples")

    def format_conversations(batch):
        texts = []
        for convos in batch["conversations"]:
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
    print(f"\n[3/4] Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=5,
            logging_steps=1,
            save_strategy="no",  # Don't save checkpoints, just final
            optim="adamw_8bit",
            seed=42,
            bf16=True,
        ),
        dataset_text_field="text",
        max_seq_length=2048,
    )

    result = trainer.train()

    # Step 4: Save model and experiment results
    print(f"\n[4/4] Saving...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save experiment config and results
    experiment = {
        "name": args.name,
        "params": {
            "epochs": args.epochs,
            "lora_rank": args.lora_rank,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "data": args.data,
            "num_examples": len(dataset),
        },
        "results": {
            "final_loss": result.metrics["train_loss"],
            "runtime_seconds": result.metrics["train_runtime"],
            "total_steps": result.global_step,
        },
        "timestamp": datetime.now().isoformat(),
    }

    import os
    os.makedirs("runs", exist_ok=True)
    results_path = f"runs/experiment_{args.name}.json"
    with open(results_path, "w") as f:
        json.dump(experiment, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Final loss: {result.metrics['train_loss']:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.1f}s")
    print(f"\nTo evaluate: python scripts/eval_quick.py  (update OUTPUT_DIR to '{output_dir}')")


if __name__ == "__main__":
    main()
