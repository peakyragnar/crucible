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
echo "[2/3] Fixing llm-blender compatibility..."
pip install llm-blender 2>&1 | tail -1
# Patch the broken import
python -c "
import llm_blender.blender.blender_utils as mod
import inspect, pathlib
src = pathlib.Path(inspect.getfile(mod))
code = src.read_text()
if 'TRANSFORMERS_CACHE' in code:
    code = code.replace(
        'from transformers.utils.hub import TRANSFORMERS_CACHE',
        'try:\n    from transformers.utils.hub import TRANSFORMERS_CACHE\nexcept ImportError:\n    from transformers.utils import TRANSFORMERS_CACHE'
    )
    src.write_text(code)
    print('  Patched llm_blender for transformers compatibility')
else:
    print('  llm_blender already compatible')
"

echo "[3/3] Verifying installation..."
python -c "from unsloth import FastLanguageModel, PatchDPOTrainer; PatchDPOTrainer(); from trl import DPOTrainer, DPOConfig; print('  All imports OK')"

echo ""
echo "=== Setup complete ==="
