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

check_docker() {
    if command -v docker &> /dev/null; then
        return 0
    else
        return 1
    fi
}


build_docker_image () {
    # Check if the Docker image exists
    if docker image inspect huggingface/optimum-nvidia:latest &> /dev/null; then
        echo "Image 'huggingface/optimum-nvidia:latest' already exists."
    else
        docker pull huggingface/optimum-nvidia:latest
    fi
}

build_and_compile_model () {
    echo "Running and building the model inside Docker..."
    local MODEL_NAME="$1"
    local PRECISION="$2"

    # Set the default folder paths for HF and engines
    LLAMA2_WEIGHTS_FOLDER="/mnt/models/llama-2-7b-chat"
    MISTRAL_WEIGHTS_FOLDER="/mnt/models/mistral-7b-v0.1-instruct"

    if [ "$MODEL_NAME" = "llama" ]; then
        HF_DIR="$LLAMA2_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$LLAMA2_WEIGHTS_FOLDER-optimum-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/llama-2-7b-chat-optimum-$PRECISION"

    elif [ "$MODEL_NAME" = "mistral" ]; then
        HF_DIR="$MISTRAL_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$MISTRAL_WEIGHTS_FOLDER-optimum-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-optimum-$PRECISION"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    if [ ! -d "$OUT_DIR" ]; then
        docker run --gpus all \
            --ipc=host \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$SCRIPT_DIR/converter.py":/mnt/converter.py \
            huggingface/optimum-nvidia:latest \
            python3 /mnt/converter.py --hf_dir "$HF_DIR" --out_dir "$ENGINE_DIR" --dtype "$PRECISION"
    else
        echo "Engine file already exists"
    fi

}


MODEL_NAME="${1:-"llama"}"

if check_docker; then
    build_docker_image
    build_and_compile_model "$MODEL_NAME" "float32"
    build_and_compile_model "$MODEL_NAME" "float16"
else
    echo "Docker is not installed or not in the PATH"
fi
