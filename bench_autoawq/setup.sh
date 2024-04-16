#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Main script starts here.
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Set default folder paths for AWQ weights
LLAMA2_AWQ_WEIGHTS_FOLDER="$CURRENT_DIR/models/llama-2-7b-chat-autoawq"
MISTRAL_AWQ_WEIGHTS_FOLDER="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-autoawq"

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

download_awq_weights() {
    local MODEL_NAME="$1"

    # Set download directory based on MODEL_NAME
    if [ "$MODEL_NAME" = "llama" ]; then
        DOWNLOAD_DIR="$LLAMA2_AWQ_WEIGHTS_FOLDER"
        MODEL_IDENTIFIER="TheBloke/Llama-2-7B-Chat-AWQ"
    elif [ "$MODEL_NAME" = "mistral" ]; then
        DOWNLOAD_DIR="$MISTRAL_AWQ_WEIGHTS_FOLDER"
        MODEL_IDENTIFIER="TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
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

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."

    # Activate virtual environment using specified activation scripts
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/Scripts/activate"
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
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/Scripts/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi
fi


MODEL_NAME="${1:-"llama"}"  # Use the first argument as MODEL_NAME if provided
download_awq_weights "$MODEL_NAME"
