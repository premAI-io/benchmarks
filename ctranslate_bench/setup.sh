#!/bin/bash

################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: This script automates the setup of a virtual environment,
# installs project requirements, converts and stores models.
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

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    source $VENV_DIR/bin/activate
    $VENV_DIR/bin/pip install --upgrade pip > /dev/null
else
    source $VENV_DIR/bin/activate
fi

$VENV_DIR/bin/pip install -r $SCRIPT_DIR/requirements.txt > /dev/null

if [ ! -d "$LLAMA_HF_MODEL_DIR-float16" ]; then
    echo "Creating llama-2-7b-hf-float16 model..."
    ct2-transformers-converter --model "$LLAMA_HF_MODEL_DIR/" --quantization float16 --output_dir "$LLAMA_HF_MODEL_DIR-float16" --copy_files tokenizer.model
else
    echo "Model llama-2-7b-hf-float16 already exists!"
fi
