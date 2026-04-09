#!/bin/bash
# Setup script for RunPod GPU pods.
# Run this once after cloning the repo on a fresh pod.
#
# Usage: bash scripts/setup_pod.sh

set -e

echo "=== Crucible Pod Setup ==="

# Install unsloth first — it pins compatible versions of torch, trl, etc.
echo "[1/3] Installing unsloth and dependencies..."
pip install unsloth datasets jsonlines python-dotenv tqdm 2>&1 | tail -3

# Install llm-blender with a fix for the transformers compatibility issue
# trl >= 0.24 requires it, but it breaks with new transformers
echo "[2/3] Fixing DPO dependencies..."
pip install llm-blender mergekit 2>&1 | tail -1
# Patch the broken import — find the file directly since we can't import it
BLENDER_FILE=$(python -c "import site; print(site.getsitepackages()[0])")/llm_blender/blender/blender_utils.py
if [ -f "$BLENDER_FILE" ]; then
    sed -i 's/from transformers.utils.hub import TRANSFORMERS_CACHE/try:\n    from transformers.utils.hub import TRANSFORMERS_CACHE\nexcept ImportError:\n    TRANSFORMERS_CACHE = None/' "$BLENDER_FILE"
    echo "  Patched llm_blender for transformers compatibility"
else
    echo "  llm_blender file not found at $BLENDER_FILE — skipping patch"
fi

echo "[3/3] Verifying installation..."
python -c "from unsloth import FastLanguageModel, PatchDPOTrainer; PatchDPOTrainer(); from trl import DPOTrainer, DPOConfig; print('  All imports OK')"

echo ""
echo "=== Setup complete ==="
