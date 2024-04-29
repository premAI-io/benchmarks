#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

# Define directory paths
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Make the default dirs
LLAMA2_EXLLAMA_WEIGHTS_FOLDER="$CURRENT_DIR/models/llama-2-7b-chat-exllamav2"
MISTRAL_EXLLAMA_WEIGHTS_FOLDER="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-exllamav2"

check_python() {
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Python is not installed."
        exit 1
    fi
}


setup_exllamav2_and_quantize() {
    local MODEL_NAME="$1"
    local QUANTIZATION="$2"

    if [ "$MODEL_NAME" = "llama" ]; then
        EXLLAMA_WEIGHTS_FOLDER="$LLAMA2_EXLLAMA_WEIGHTS_FOLDER-$QUANTIZATION-bit"
        HF_WEIGHTS_FOLDER="$CURRENT_DIR/models/llama-2-7b-chat-hf"
    elif [ "$MODEL_NAME" = "mistral" ]; then
        EXLLAMA_WEIGHTS_FOLDER="$MISTRAL_EXLLAMA_WEIGHTS_FOLDER-$QUANTIZATION-bit"
        HF_WEIGHTS_FOLDER="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-hf"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    # do the conversion if the ExLlamaV2
    if [ -d "$EXLLAMA_WEIGHTS_FOLDER" ] && [ "$(ls -A "$EXLLAMA_WEIGHTS_FOLDER")" ]; then
        echo "EXLLAMA_WEIGHTS_FOLDER already exists and is not empty."
    else
        # clone the repo, if not exists
        if [ -d "$SCRIPT_DIR/exllamav2" ]; then
            echo "exllamav2 folder already exists."
        else
            git clone https://github.com/turboderp/exllamav2.git "$SCRIPT_DIR/exllamav2"
        fi

        mkdir -p "$EXLLAMA_WEIGHTS_FOLDER"
        echo "Going for conversion to exllamav2 format from .safetensors in $QUANTIZATION bit quantization."
        "$PYTHON_CMD" "$SCRIPT_DIR/exllamav2/convert.py" \
        -i "$HF_WEIGHTS_FOLDER" \
        -o "$EXLLAMA_WEIGHTS_FOLDER" \
        -cf "$EXLLAMA_WEIGHTS_FOLDER" \
        -b "$QUANTIZATION"

        # once done sync with other folders
        rm -rf "$EXLLAMA_WEIGHTS_FOLDER/out_tensor"
        rsync -av --exclude='*.safetensors' --exclude='.*' --exclude='*.bin' "$HF_WEIGHTS_FOLDER" "$EXLLAMA_WEIGHTS_FOLDER"
    fi

    # Delete ExllamaV2 repo
    rm -rf "$SCRIPT_DIR/exllamav2"
}


check_python

# CLI Args
MODEL_NAME="$1"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

echo "Converting HuggingFace Llama2 model pytorch .bin file to .safetensors format"

setup_exllamav2_and_quantize "$MODEL_NAME" 4.0
setup_exllamav2_and_quantize "$MODEL_NAME" 8.0
