#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

check_python

install_vllm_cuda() {
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \(.*\),.*/\1/p')

    if [ -z "$CUDA_VERSION" ]; then
        echo "CUDA is not installed or not found."
        exit 1
    fi

    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

   if [ "$CUDA_MAJOR" -ge 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 0 ]; }; then
        echo "Detected CUDA version >= 12.2"
        "$PYTHON_CMD" -m pip install vllm==0.4.0 transformers==4.39.2
    else
        echo "Detected CUDA version < 12.2"
        PY_VERSION=$(get_python_version)
        if [ -z "$PY_VERSION" ]; then
            echo "Python version not found."
            exit 1
        fi
        echo "Installing vllm for CUDA 11.8 with Python version: $PY_VERSION"
        # Download vllm for CUDA 11.8 and specified Python version
        "$PYTHON_CMD" -m pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-"$PY_VERSION"-"$PY_VERSION"-manylinux1_x86_64.whl
        "$PYTHON_CMD" -m pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
        "$PYTHON_CMD" -m pip install huggingface-cli==0.1 transformers==4.39.2
    fi
}

get_python_version() {
    # Fetch Python version
    PY_VER=$("$PYTHON_CMD" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

    case $PY_VER in
        3.10) echo "cp310";;
        3.8) echo "cp38";;
        3.9) echo "cp39";;
        3.11) echo "cp311";;
        *) echo "Unknown Python version"; exit 1;;
    esac
}


install_device_specific_vllm() {
    local DEVICE="$1"

    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <DEVICE>"
        exit 1
    fi

    case "$DEVICE" in
        cuda)
            echo "Installing VLLM for CUDA."
            install_vllm_cuda
            ;;
        metal)
            echo "VLLM for metal is not supported yet."
            echo "For more information, checkout this issue: https://github.com/vllm-project/vllm/issues/1441"
            return 1
            ;;
        cpu)
            echo "VLLM for CPU is not supported yet."
            echo "For more information, checkout this issue: https://github.com/vllm-project/vllm/issues/176"
            ;;
        *)
            echo "Unsupported DEVICE: $DEVICE"
            return 1
            ;;
    esac
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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

DEVICE="$1"
MODEL_NAME="$2"


# Build and activate the virtual environment.

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    "$PYTHON_CMD" -m pip install --upgrade pip > /dev/null
    install_device_specific_vllm "$DEVICE"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

download_awq_weights "$MODEL_NAME"
