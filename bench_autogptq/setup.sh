#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Main script starts here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HF_MODELS_DIR="${MODELS_DIR:-"models/llama-2-7b-hf"}"
GPTQ_WEIGHTS_FOLDER="${GPTQ_WEIGHTS_FOLDER:-"./models/llama-2-7b-gptq"}"

convert_hf_to_gptq() {
    local HF_WEIGHTS_FOLDER="$1"
    local GPTQ_WEIGHTS_FOLDER="$2"
    local QUANTIZATION="$3"

    # download the sample file if not exists
    if [ -f "$SCRIPT_DIR/wikitext-test.parquet" ]; then
        echo "wikitext-test.parquet file already exists."
    else
        wget -P "$SCRIPT_DIR" https://huggingface.co/datasets/wikitext/resolve/9a9e482b5987f9d25b3a9b2883fc6cc9fd8071b3/wikitext-103-v1/wikitext-test.parquet
    fi

    # do the conversion if the exllamav2 folder does not exists
    if [ -d "$GPTQ_WEIGHTS_FOLDER" ] && [ "$(ls -A "$GPTQ_WEIGHTS_FOLDER")" ]; then
        echo "GPTQ_WEIGHTS_FOLDER already exists and is not empty."
    else
        mkdir -p "$GPTQ_WEIGHTS_FOLDER"
        python "$SCRIPT_DIR/quantize.py" \
        --hf_dir "$HF_WEIGHTS_FOLDER" \
        --q_dir "$GPTQ_WEIGHTS_FOLDER" \
        --precision "$QUANTIZATION" \
        --parquet "$SCRIPT_DIR/wikitext-test.parquet"
    fi

}


if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
    git clone https://github.com/PanQiWei/AutoGPTQ.git "$SCRIPT_DIR"
    pip install -v "$SCRIPT_DIR/AutoGPTQ" .
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# do for q4 and q8
convert_hf_to_gptq "$HF_MODELS_DIR/" "$GPTQ_WEIGHTS_FOLDER-q4" 4
convert_hf_to_gptq "$HF_MODELS_DIR/" "$GPTQ_WEIGHTS_FOLDER-q8" 8
