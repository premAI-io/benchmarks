#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Main script starts here.

CURRENT_DIR="$(pwd)"

check_docker() {
    if command -v docker &> /dev/null; then
        return 0
    else
        return 1
    fi
}


build_docker_image () {
    # Check if the Docker image exists
    if docker image inspect anindyadeep/onnxruntime:latest &> /dev/null; then
        echo "Image 'anindyadeep/onnxruntime:latest' already exists."
    else
        docker pull anindyadeep/onnxruntime:latest
    fi
}

build_and_compile_model () {
    echo "Running and building the model inside Docker..."
    local MODEL_NAME="$1"
    local PRECISION="$2"
    local DEVICE="$3"

    # Set the default folder paths for HF and engines
    LLAMA2_WEIGHTS_FOLDER="/mnt/models/llama-2-7b-chat"
    MISTRAL_WEIGHTS_FOLDER="/mnt/models/mistral-7b-v0.1-instruct"

    if [ "$MODEL_NAME" = "llama" ]; then
        HF_DIR="$LLAMA2_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$LLAMA2_WEIGHTS_FOLDER-onnx-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/llama-2-7b-chat-onnx-$PRECISION"

    elif [ "$MODEL_NAME" = "mistral" ]; then
        HF_DIR="$MISTRAL_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$MISTRAL_WEIGHTS_FOLDER-onnx-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-onnx-$PRECISION"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    if [ "$PRECISION" = "float32" ]; then
        ONNX_PRECISION="fp32"
    elif [ "$PRECISION" = "float16" ]; then
        ONNX_PRECISION="fp16"
    else
        echo "Supported precision: 'float32' and 'float16'"
        exit 1
    fi

    if [ ! -d "$OUT_DIR" ]; then
        docker run --gpus all \
            --ipc=host \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v "$CURRENT_DIR"/models:/mnt/models \
            anindyadeep/onnxruntime:latest \
            optimum-cli export onnx --model "$HF_DIR" \
                --task text-generation --framework pt \
                --opset 17 --sequence_length 1024 \
                --batch_size 1 --device "$DEVICE" \
                --dtype "$ONNX_PRECISION" "$ENGINE_DIR"
    else
        echo "Engine file already exists"
    fi

}


MODEL_NAME="${1:-"llama"}"
DEVICE="$2"

if check_docker; then
    build_docker_image
    build_and_compile_model "$MODEL_NAME" "float32" "$DEVICE"
    build_and_compile_model "$MODEL_NAME" "float16" "$DEVICE"
else
    echo "Docker is not installed or not in the PATH"
fi
