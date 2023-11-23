#!/bin/bash

################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: Automates the setup of a virtual environment and installs project
# requirements and handles model conversion.
################################################################################

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <models_folder>"
    exit 1
fi

# Define directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_FOLDER="$1"
LLAMA_HF_MODEL_DIR="$MODELS_FOLDER/llama-2-7b-hf"
LLAMA_ONNX_MODEL_DIR="$MODELS_FOLDER/llama-2-7b-onnx"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR"/requirements.txt
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

get_device() {
  if command -v nvidia-smi &> /dev/null; then
    echo "cuda"
  else
    echo "cpu"
  fi
}

# Check and create llama-2-7b-onnx model
if [ ! -d "$LLAMA_ONNX_MODEL_DIR" ]; then
    echo "optimum-cli export onnx \
        --model $LLAMA_HF_MODEL_DIR --task text-generation --framework pt \
        --opset 17 --sequence_length 1024 --batch_size 1 --device $(get_device) --fp16 \
        $LLAMA_ONNX_MODEL_DIR"
    optimum-cli export onnx \
        --model "$LLAMA_HF_MODEL_DIR" --task text-generation --framework pt \
        --opset 17 --sequence_length 1024 --batch_size 1 --device "$(get_device)" --fp16 \
        "$LLAMA_ONNX_MODEL_DIR"
else
    echo "Model llama-2-7b-onnx already exists!"
fi
