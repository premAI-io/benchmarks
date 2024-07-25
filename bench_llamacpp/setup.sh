#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE> <MODEL_NAME>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Set default folder paths for AWQ weights
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$CURRENT_DIR/models"
LLAMA2_GGUF_WEIGHTS_DIR="$MODELS_DIR/llama-2-7b-chat-gguf"
MISTRAL_GGUF_WEIGHTS_DIR="$MODELS_DIR/mistral-7b-v0.1-instruct-gguf"

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

download_gguf_weights() {
    local MODEL_NAME="$1"
    local DOWNLOAD_DIR

    case "$MODEL_NAME" in
        llama)
            DOWNLOAD_DIR="$LLAMA2_GGUF_WEIGHTS_DIR"
            MODEL_IDENTIFIER="TheBloke/Llama-2-7B-Chat-GGUF"
            MODEL_FILE_4BIT="llama-2-7b-chat.Q4_K_M.gguf"
            MODEL_FILE_8BIT="llama-2-7b-chat.Q8_0.gguf"
            ;;
        mistral)
            DOWNLOAD_DIR="$MISTRAL_GGUF_WEIGHTS_DIR"
            MODEL_IDENTIFIER="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
            MODEL_FILE_4BIT="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            MODEL_FILE_8BIT="mistral-7b-instruct-v0.1.Q8_0.gguf"
            ;;
        *)
            echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
            exit 1
            ;;
    esac

    if [ ! -d "$DOWNLOAD_DIR" ]; then
        huggingface-cli download "$MODEL_IDENTIFIER" "$MODEL_FILE_4BIT" --local-dir "$DOWNLOAD_DIR" --local-dir-use-symlinks False
        huggingface-cli download "$MODEL_IDENTIFIER" "$MODEL_FILE_8BIT" --local-dir "$DOWNLOAD_DIR" --local-dir-use-symlinks False
    else
        echo "Weights for $MODEL_NAME already downloaded."
    fi
}

clone_and_build_llama() {
    local DEVICE="$1"

    echo "Building llama.cpp..."
    if [ "$DEVICE" == "cuda" ]; then
        pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
    else
        pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    fi
}

# CLI Args
DEVICE="$1"
MODEL_NAME="$2"

# Define directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    clone_and_build_llama "$DEVICE"
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
    pip install numpy --upgrade 
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    
fi
 
download_gguf_weights "$MODEL_NAME"
