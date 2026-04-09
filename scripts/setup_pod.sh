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
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
# Patch ALL llm_blender files with the broken import
for f in $(find "$SITE_PACKAGES/llm_blender" -name "*.py" -exec grep -l "from transformers.utils.hub import TRANSFORMERS_CACHE" {} \; 2>/dev/null); do
    sed -i 's/from transformers.utils.hub import TRANSFORMERS_CACHE/TRANSFORMERS_CACHE = None/' "$f"
    echo "  Patched $f"
done

echo "[3/3] Verifying installation..."
python -c "from unsloth import FastLanguageModel, PatchDPOTrainer; PatchDPOTrainer(); from trl import DPOTrainer, DPOConfig; print('  All imports OK')"

echo ""
echo "=== Setup complete ==="
