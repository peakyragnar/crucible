⏺ # Project: Crucible — Frontier-to-Open Investment Research Model

  ## What This Project Is
    The goal of this project is to teach the operator to become world class at post training models.  The operator is not a physics/math major.  He will be using frontier models for this.  The goal is to become world class using frontier models to improve models to specific objectives.  

  We will build as well A pipeline for using frontier AI models (Claude, etc.) to generate high-quality training data and fine-tune open-source language models for investment research. The operator is not an ML researcher — the frontier model is the researcher. The operator's job is data curation,pipeline execution, and evaluation.

  ## Core Principles

  - We do NOT pre-train from scratch. We start from existing open base models (Llama, Qwen, Mistral).
  - The frontier model (Claude) is the data engine — it generates, evaluates, and iterates on training data.
  - Quality of data matters more than quantity. Always filter, never bulk-dump.
  - Every fine-tuning run must be evaluated against the base model. If you can't measure improvement, don't train.
  - Keep it simple. No custom training loops. Use established tooling (unsloth, trl, axolotl).
  - Domain expertise is the moat. Training pipeline is commodity infrastructure.

  ## Tech Stack

  | Component | Tool | Purpose |
  |-----------|------|---------|
  | Fine-tuning | unsloth | Fast LoRA/QLoRA fine-tuning, single GPU |
  | Base models | HuggingFace Hub | Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B |
  | Evaluation | lm-evaluation-harness | Standardized benchmarks (MMLU, GSM8K, HumanEval, etc.) |
  | Data generation | Claude API / Anthropic SDK | Synthetic training data, preference pairs, reasoning traces |
  | Preference training | trl (HuggingFace) | DPO and RLHF post-training |
  | Model merging | mergekit | Combine specialist fine-tunes |
  | Inference/serving | vllm or llama.cpp | Run and test models locally or on GPU |
  | Compute | RunPod / Lambda / local | GPU rental as needed |

  ## Repository Structure

  data/               # Training datasets (generated, curated, filtered)
    raw/              # Raw generations from frontier model
    curated/          # Filtered and validated training data
    preferences/      # DPO preference pairs (chosen/rejected)
  configs/            # Training configs for unsloth/axolotl
  scripts/
    generate.py       # Data generation via Claude API
    curate.py         # Data filtering and quality checks
    train.py          # Fine-tuning wrapper
    evaluate.py       # Evaluation runner
    merge.py          # Model merging scripts
    serve.py          # Local inference server
  evals/              # Custom evaluation tasks specific to investment research
  models/             # Model cards, notes on each training run
  runs/               # Logs and results from training runs

  ## Workflow

  ### Phase 1: Data Generation
  1. Define the target capability (domain, task type, difficulty level)
  2. Use Claude to generate training examples (conversations, reasoning traces, tool use)
  3. Operator reviews and filters for quality — domain expertise is applied HERE
  4. Format as JSONL conversations for SFT or preference pairs for DPO

  ### Phase 2: Fine-Tuning
  1. Start from a base model (default: Llama 3.1 8B or Qwen 2.5 7B)
  2. Run SFT with unsloth using generated data
  3. Optionally run DPO with preference pairs
  4. Save adapter weights and merged model

  ### Phase 3: Evaluation
  1. Run lm-evaluation-harness on base model AND fine-tuned model
  2. Compare scores — if no improvement, the data was bad, not the training
  3. Run domain-specific evals from evals/ directory
  4. Log everything to runs/

  ### Phase 4: Iteration
  1. Analyze failures — where does the fine-tuned model still fall short?
  2. Generate targeted data for failure modes
  3. Retrain, re-evaluate
  4. When multiple specialists exist, merge with mergekit

  ## Target Domain: Investment Research

  ### What This Means Specifically
  A model that assists with equity and market research workflows — not trading signals or price prediction, but the analytical work that precedes investment decisions.

  ### Trainable Capabilities (prioritized)

  1. **Financial statement analysis** — read 10-K/10-Q data, identify trends, flag anomalies,compute and interpret ratios. The model should think like an analyst, not a calculator.

  2. **SEC filing comprehension** — parse dense legal/financial language from proxy statements, 8-Ks, S-1s. Extract what matters, ignore boilerplate.

  3. **Earnings call analysis** — summarize transcripts, identify management tone shifts,flag guidance changes, compare language across quarters.

  4. **Investment thesis generation** — given a company and data, produce a structured bull/bear case with supporting evidence. Not opinions — frameworks.

  5. **Risk identification** — surface risks from filings, industry context, and macro conditions that a generalist model would miss or underweight.

  6. **Comparable company analysis** — identify relevant comps, explain why they're
     comparable, note where the analogy breaks down.

  ### What We Are NOT Building
  - A trading bot or signal generator
  - A price prediction model
  - A replacement for Bloomberg terminal data
  - A model that gives investment advice

  ### Why Fine-Tuning Beats Prompting Here
  - **Cost**: Analyzing 500 filings via Claude API is expensive. A local 8B model is near-free.
  - **Privacy**: Proprietary research stays local.
  - **Consistency**: A fine-tuned model produces structured output in a predictable format every time, without prompt engineering.
  - **Speed**: Local inference on a single GPU is faster than API round-trips at scale.
  - **Domain calibration**: A general model hedges everything. A research model should be direct about what the numbers say.

  ### Operator's Edge
  20 years at leading hedge funds and investment banks. Can evaluate quality of:
  - Financial statement analysis (knows what matters vs. textbook ratios)
  - Earnings call interpretation (has read thousands, knows what "real" analysis looks like)
  - Investment thesis construction (has written and defended them professionally)
  - Risk assessment (has seen what actually blows up vs. what models flag as risky)

  This means: the operator can aggressively filter generated training data for quality. A mediocre earnings summary that would fool a software engineer will not fool this operator.  The quality ceiling on our training data is high because the curator is a domain expert.

  ## Guidelines for Claude (the assistant in this repo)

  - When asked to generate training data, prioritize quality and diversity over volume.
  - When suggesting training approaches, prefer the simplest method that could work.
    SFT before DPO. DPO before RLHF. LoRA before full fine-tune.
  - Always recommend evaluation before and after any training run.
  - Do not over-engineer. A Python script that works beats a framework that's elegant.
  - When the operator describes a domain, help them decompose it into trainable capabilities.
  - Flag when a task is better solved by prompting the base model vs. fine-tuning.
    Not everything needs training.
  - Keep costs visible. Estimate API spend for data generation, GPU hours for training.

  ## Current Phase

  Phase 0 — Repository setup and tooling installation.