#!/usr/bin/env bash
set -e

ENV_NAME="florence"

echo "============================================================"
echo " [1/6] Removing old conda env '$ENV_NAME' (if exists)..."
echo "============================================================"
conda env remove -n "$ENV_NAME" --yes >/dev/null 2>&1 || echo "No existing env named '$ENV_NAME', OK."

echo
echo "============================================================"
echo " [2/6] Creating new conda env '$ENV_NAME' with Python 3.10..."
echo "============================================================"
conda create -n "$ENV_NAME" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo
echo "============================================================"
echo " [3/6] Installing core packages: numpy<2, torch, torchvision..."
echo "============================================================"
pip install "numpy<2.0"

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

echo
echo "============================================================"
echo " [4/6] Installing Florence-2 dependencies (einops, timm, pillow)..."
echo "============================================================"
pip install einops timm pillow

echo
echo "============================================================"
echo " [5/6] Installing transformers + pydantic (compatible versions)..."
echo "============================================================"
pip install "transformers==4.37.2" "pydantic==1.10.13" accelerate safetensors

echo
echo "============================================================"
echo " [6/6] (Optional) Trying to install flash-attn..."
echo "============================================================"

pip install "flash-attn>=2.5.9" --no-build-isolation || echo "WARNING: flash-attn failed to install, continuing without it."

echo
echo "============================================================"
echo " ✅ Environment '$ENV_NAME' is ready."
echo "    Next steps:"
echo "      1) conda activate $ENV_NAME"
echo "      2) python test_florence_single.py  # 用来测试 Florence-2 是否正常"
echo "============================================================"
