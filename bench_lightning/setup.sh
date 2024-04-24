#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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


setup_environment() {
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        echo "Virtual environment '$VENV_DIR' created."
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip > /dev/null

        # install everything
        pip install 'litgpt[all] @ git+https://github.com/Lightning-AI/litgpt'
        pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
        echo "Successfully installed lit-gpt and it's dependencies"
    else
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
}

convert_hf_to_litgpt() {
    local MODEL_NAME="$1"

    # This trick is done because LitGPT expects specific folder name / checkpoint_dir
    # Llama-2-7b-chat-hf or Mistral-7B-Instruct-v0.1
    TEMP_DIR=""
    LITGPT_DIR=""
    BACK_TO_DIR=""

    if [ "$MODEL_NAME" = "llama" ]; then
        TEMP_DIR="$CURRENT_DIR/models/Llama-2-7b-chat-hf"
        LITGPT_DIR="$CURRENT_DIR/models/llama-2-7b-chat-litgpt"
        BACK_TO_DIR="$CURRENT_DIR/models/llama-2-7b-chat-hf"
    elif [ "$MODEL_NAME" = "mistral" ]; then
        TEMP_DIR="$CURRENT_DIR/models/Mistral-7B-Instruct-v0.1"
        LITGPT_DIR="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-litgpt"
        BACK_TO_DIR="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-hf"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    if [ -d "$LITGPT_DIR" ]; then
        echo "Already converted"
        exit 0
    else
        mv "$BACK_TO_DIR" "$TEMP_DIR"
        mkdir -p "$LITGPT_DIR"
        litgpt convert to_litgpt --checkpoint_dir "$TEMP_DIR"
        mv "$TEMP_DIR/model_config.yaml" "$TEMP_DIR/lit_model.pth" "$LITGPT_DIR/"
        cp -r "$TEMP_DIR/tokenizer.model" "$TEMP_DIR/tokenizer_config.json" "$LITGPT_DIR/"
        mv "$TEMP_DIR" "$BACK_TO_DIR"
    fi
}


MODEL_NAME="$1"

check_python
setup_environment
convert_hf_to_litgpt "$MODEL_NAME"
