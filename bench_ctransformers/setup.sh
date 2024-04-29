#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements including CTransformers and GGUF weights.
################################################################################

set -euo pipefail

# Define constants and paths
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$CURRENT_DIR/models"
LLAMA2_GGUF_WEIGHTS_DIR="$MODELS_DIR/llama-2-7b-chat-gguf"
MISTRAL_GGUF_WEIGHTS_DIR="$MODELS_DIR/mistral-7b-v0.1-instruct-gguf"

# Check if Python is installed
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

install_ctransformers_cuda() {
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \(.*\),.*/\1/p')

    if [ -z "$CUDA_VERSION" ]; then
        echo "CUDA is not installed or not found."
        exit 1
    fi

    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

   if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 2 ]; }; then
        echo "Detected CUDA version >= 12.2"
        pip install ctransformers[cuda] > /dev/null
    else
        echo "Detected CUDA version < 12.2"
        CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$(which nvcc)" CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers > /dev/null
    fi
}

# Install CTransformers based on the specified device
install_ctransformers() {
    local DEVICE="$1"

    case "$DEVICE" in
        cuda)
            echo "Installing CTransformers for CUDA."
            install_ctransformers_cuda
            ;;
        metal)
            echo "Installing CTransformers for Metal."
            pip uninstall ctransformers --yes
            CT_METAL=1 pip install ctransformers --no-binary ctransformers
            ;;
        cpu)
            echo "Installing CTransformers for CPU."
            pip install ctransformers > /dev/null
            ;;
        *)
            echo "Unsupported DEVICE: $DEVICE"
            exit 1
            ;;
    esac
}

# Download GGUF weights for the specified model
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

# Main script starts here

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <DEVICE> <MODEL_NAME>"
    exit 1
fi

check_python

# Define command line arguments
DEVICE="$1"
MODEL_NAME="$2"

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
    install_ctransformers "$DEVICE"
else
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: Unable to find virtual environment activation script."
        exit 1
    fi
fi


download_gguf_weights "$MODEL_NAME"
