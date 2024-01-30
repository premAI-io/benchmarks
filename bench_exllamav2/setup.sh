#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

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

convert_bin_to_safetensor() {
    local HF_MODEL_FOLDER_PATH="$1"

    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
    "$PYTHON_CMD" "$SCRIPT_DIR"/convert.py \
        "$HF_MODEL_FOLDER_PATH"
}


convert_safetensor_to_exllamav2() {
    local HF_WEIGHTS_FOLDER="$1"
    local EXLLAMA_WEIGHTS_FOLDER="$2"
    local QUANTIZATION="$3"

    # clone the repo, if not exists
    if [ -d "$SCRIPT_DIR/exllamav2" ]; then
        echo "exllamav2 folder already exists."
    else
        git clone https://github.com/turboderp/exllamav2 "$SCRIPT_DIR/exllamav2"
    fi

    # download the sample file if not exists
    if [ -f "$SCRIPT_DIR/wikitext-test.parquet" ]; then
        echo "wikitext-test.parquet file already exists."
    else
        wget -P "$SCRIPT_DIR" https://huggingface.co/datasets/wikitext/resolve/9a9e482b5987f9d25b3a9b2883fc6cc9fd8071b3/wikitext-103-v1/wikitext-test.parquet
    fi

    # do the conversion if the exllamav2 folder does not exists
    if [ -d "$EXLLAMA_WEIGHTS_FOLDER" ] && [ "$(ls -A "$EXLLAMA_WEIGHTS_FOLDER")" ]; then
        echo "EXLLAMA_WEIGHTS_FOLDER already exists and is not empty."
    else
        mkdir -p "$EXLLAMA_WEIGHTS_FOLDER"
        echo "Going for conversion to exllamav2 format from .safetensors in $QUANTIZATION bit quantization."
        "$PYTHON_CMD" "$SCRIPT_DIR/exllamav2/convert.py" \
        -i "$HF_WEIGHTS_FOLDER" \
        -o "$EXLLAMA_WEIGHTS_FOLDER" \
        -c "$SCRIPT_DIR/wikitext-test.parquet" \
        -b "$QUANTIZATION"

        # once done, delete the un-necessary files
        rm -rf "$EXLLAMA_WEIGHTS_FOLDER/out_tensor"
        rsync -av --exclude='*.safetensors' --exclude='.*' --exclude='*.bin' "$HF_WEIGHTS_FOLDER" "$EXLLAMA_WEIGHTS_FOLDER"
    fi
}


check_python

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="${MODELS_DIR:-"models/llama-2-7b-hf"}"
EXLLAMA_BASE_MODEL_DIR="${EXLLAMA_BASE_MODEL_DIR:-"./models/llama-2-7b-exllamav2"}"

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
convert_bin_to_safetensor "$CURRENT_DIR/$MODELS_DIR"

# do one for q4
convert_safetensor_to_exllamav2 "$MODELS_DIR/" "$EXLLAMA_BASE_MODEL_DIR-q4" 4
# do one for q8
convert_safetensor_to_exllamav2 "$MODELS_DIR/" "$EXLLAMA_BASE_MODEL_DIR-q8" 8
