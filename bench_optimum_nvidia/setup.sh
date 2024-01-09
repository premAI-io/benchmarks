#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Main script starts here.

CURRENT_DIR="$(pwd)"

check_docker () {
    pass
}

build_docker_image () {
    # Check if the Docker image exists
    if docker image inspect prem/optimum-nvidia:latest &> /dev/null; then
        echo "Image 'prem/optimum-nvidia:latest' already exists."
        exit 1
    else
        # Clone the repository and build the Docker image
        git clone --recursive --depth=1 https://github.com/huggingface/optimum-nvidia.git
        cd optimum-nvidia/third-party/tensorrt-llm
        make -C docker release_build
        cd ../.. && docker build -t prem/optimum-nvidia:base -f docker/Dockerfile .
    fi

    cd "$CURRENT_DIR"
}

build_and_compile_model () {
    echo "Running ... "
    if docker image inspect prem/optimum-nvidia:latest &> /dev/null; then
        docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/paperspace/workspace/benchmarks/models:/models prem/optimum-nvidia:latest python3 ./text-generation/llama.py /models/llama-2-7b-hf /build
        docker tag prem/optimum-nvidia:latest prem/optimum-nvidia:v1  # Tagging the new image as 'latest'
        docker image rm prem/optimum-nvidia:latest -f  # Remove the old 'base' image
    else
        echo "The base image does not exist locally. Exiting..."
        exit 1
    fi
}



# build_docker_image
build_and_compile_model
