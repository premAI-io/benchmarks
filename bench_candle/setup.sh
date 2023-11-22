#!/bin/bash

################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: This script automates the setup of a virtual environment,
# installs project requirements, converts model.
################################################################################

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <models_folder>"
    exit 1
fi

# Define directory paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_FOLDER="$1"
LLAMA_HF_MODEL_DIR="$MODELS_FOLDER/llama-2-7b-hf"
LLAMA_ST_MODEL_DIR="$MODELS_FOLDER/llama-2-7b-st"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR"/bin/activate
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR"/requirements.txt > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR"/bin/activate
fi

if [ ! -d "$LLAMA_ST_MODEL_DIR" ]; then
    echo "Storing llama-2-7b-hf in safetensors format..."
    python "$SCRIPT_DIR"/convert_to_safetensors.py --input_dir "$LLAMA_HF_MODEL_DIR" --output_dir "$LLAMA_ST_MODEL_DIR"
else
    echo "Model llama-2-7b-hf in safetensors format already exists!"
fi
