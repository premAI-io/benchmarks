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
    local repo_name="optimum-nvidia"

    # Check if the Docker image exists
    if docker image inspect prem/optimum-nvidia:base &> /dev/null; then
        echo "Image 'prem/optimum-nvidia:base' already exists."
        exit 0
    else

        if [ -d "$SCRIPT_DIR/$repo_name" ]; then
            echo "Repo already cloned"
        else
            git clone --recursive --depth=1 https://github.com/huggingface/optimum-nvidia.git "$SCRIPT_DIR/$repo_name"
        fi
        cd "$SCRIPT_DIR/$repo_name/third-party/tensorrt-llm"
        make -C docker release_build
        cd ../.. && docker build -t prem/optimum-nvidia:base -f docker/Dockerfile .
    fi

    cd "$CURRENT_DIR"
}

build_and_compile_model () {
    echo "Running and building the model inside Docker..."

    if docker image inspect prem/optimum-nvidia:base &> /dev/null; then
        echo "Image 'prem/optimum-nvidia:base' already exists."
        exit 0
    elif docker image inspect prem/optimum-nvidia:base &> /dev/null; then
        local model_build_path="$CURRENT_DIR/models/llama-2-7b-optimum_nvidia_build"
        if [ ! -d "$model_build_path" ]; then
            mkdir "$model_build_path"
        fi

        if [ -z "$(ls -A "$model_build_path")" ]; then
            docker run \
                --gpus all \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path":/optimum_nvidia_build \
                prem/optimum-nvidia:base \
                python3 ./text-generation/llama.py /models/llama-2-7b-hf /optimum_nvidia_build
        else
            echo "Engine file already exists"
        fi
    else
        echo "The base image does not exist locally. Exiting..."
        exit 0
    fi
}



if check_docker; then
    build_docker_image
    build_and_compile_model
else
    echo "Docker is not installed or not in the PATH"
fi
