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
    if docker image inspect huggingface/optimum-nvidia:latest &> /dev/null; then
        echo "Image 'huggingface/optimum-nvidia:latest' already exists."
    else
        docker pull huggingface/optimum-nvidia
    fi

    cd "$CURRENT_DIR"
}

build_and_compile_model () {
    echo "Running and building the model inside Docker..."
    local model_build_path="$CURRENT_DIR/models/llama-2-7b-optimum_nvidia_build"
    if [ ! -d "$model_build_path" ]; then
        mkdir "$model_build_path"
    fi

    if [ -z "$(ls -A "$model_build_path")" ]; then
        docker run --gpus all \
            --ipc=host \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v "$CURRENT_DIR"/models:/models \
            -v "$model_build_path":/optimum_nvidia_build \
            huggingface/optimum-nvidia:latest \
            python3 ./text-generation/llama.py /models/llama-2-7b-hf /optimum_nvidia_build
    else
        echo "Engine file already exists"
    fi

}

if check_docker; then
    build_docker_image
    build_and_compile_model
else
    echo "Docker is not installed or not in the PATH"
fi
