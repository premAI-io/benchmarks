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
CURRENT_DIR="$(pwd)"

# Set default folder paths for GPTQ weights
LLAMA2_GPTQ_WEIGHTS_FOLDER="$CURRENT_DIR/models/llama-2-7b-chat-autogptq"
MISTRAL_GPTQ_WEIGHTS_FOLDER="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-autogptq"


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

download_gptq_weights() {
    local MODEL_NAME="$1"

    # Set download directory based on MODEL_NAME
    if [ "$MODEL_NAME" = "llama" ]; then
        DOWNLOAD_DIR="$LLAMA2_GPTQ_WEIGHTS_FOLDER"
        MODEL_IDENTIFIER="TheBloke/Llama-2-7B-Chat-GPTQ"
    elif [ "$MODEL_NAME" = "mistral" ]; then
        DOWNLOAD_DIR="$MISTRAL_GPTQ_WEIGHTS_FOLDER"
        MODEL_IDENTIFIER="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    # Check if weights folder exists
    echo "$DOWNLOAD_DIR"

    if [ ! -d "$DOWNLOAD_DIR" ]; then
        # Download weights using huggingface-cli
        echo "Downloading weights to $DOWNLOAD_DIR..."
        huggingface-cli download "$MODEL_IDENTIFIER" --local-dir "$DOWNLOAD_DIR" --exclude "*.git*" "*.md" "Notice" "LICENSE"
    else
        echo "Weights already downloaded"
    fi
}

install_autogptq() {
    if [ -d "$SCRIPT_DIR/AutoGPTQ" ]; then
        echo "Removing existing AutoGPTQ directory..."
        rm -rf "$SCRIPT_DIR"/AutoGPTQ
    fi

    git clone https://github.com/PanQiWei/AutoGPTQ.git "$SCRIPT_DIR"/AutoGPTQ
    cd "$SCRIPT_DIR"/AutoGPTQ

    # Now build

    "$PYTHON_CMD" setup.py install

    # come out of the dir
    cd "$SCRIPT_DIR"
}

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."

    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi

    "$PYTHON_CMD" -m pip install --upgrade pip > /dev/null
    "$PYTHON_CMD" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null

    "$PYTHON_CMD" -m pip uninstall -y fsspec

    # Install the required version of fsspec
    "$PYTHON_CMD" -m pip install 'fsspec[http]>=2023.1.0,<=2024.2.0'

    install_autogptq
else
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi
fi


MODEL_NAME="${1:-"llama"}"  # Use the first argument as MODEL_NAME if provided
download_gptq_weights "$MODEL_NAME"
