#!/bin/bash

################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: This script automates the setup of a virtual environment,
# installs project requirements, converts model.
################################################################################

set -euo pipefail

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

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


build_and_compile_model () {
    local MODEL_NAME="$1"
    local PRECISION="$2"

    valid_precisions=("float32" "float16" "int8")

    # shellcheck disable=SC2199
    # shellcheck disable=SC2076
    if [[ ! " ${valid_precisions[@]} " =~ " $PRECISION " ]]; then
        echo "Invalid PRECISION value. Supported values are ${valid_precisions[*]}."
        exit 1
    fi

    if [[ "$MODEL_NAME" == "llama" ]]; then
        local model_download_path="$CURRENT_DIR/models/llama-2-7b-chat-ctranslate2-$PRECISION"
        local model_to_convert="$CURRENT_DIR/models/llama-2-7b-chat-hf"

    elif [[ "$MODEL_NAME" == "mistral" ]]; then
        local model_download_path="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-ctranslate2-$PRECISION"
        local model_to_convert="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-hf"
    else
        echo "No such model is supported"
        exit 1
    fi


    if [ ! -d "$model_download_path" ]; then
        ct2-transformers-converter --model "$model_to_convert" --quantization "$PRECISION" --output_dir "$model_download_path" --copy_files tokenizer.model tokenizer_config.json tokenizer.json special_tokens_map.json --force
        echo "Model Build for model: $MODEL_NAME and precision: $PRECISION ran successfully"
    else
        echo "Download folder already exists"
    fi

}


build_and_compile_models() {
    local MODEL_NAME="$1"
    local PRECISIONS=("float32" "float16" "int8")

    for PRECISION in "${PRECISIONS[@]}"; do
        build_and_compile_model "$MODEL_NAME" "$PRECISION"
    done
}


MODEL_NAME="${1:-"llama"}"

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."

    # Activate virtual environment using specified activation scripts
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi

    "$PYTHON_CMD" -m pip install --upgrade pip > /dev/null
    "$PYTHON_CMD" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # Activate virtual environment using specified activation scripts
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi
fi


build_and_compile_models "$MODEL_NAME"
