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
    # Todo: might require to clone a Patched version.
    local repo_name="TensorRT-LLM"

    # Check if the Docker image exists
    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        echo "Image 'tensorrt_llm/release:latest' already exists."
    else

        if [ -d "$SCRIPT_DIR/$repo_name" ]; then
            echo "Repo already cloned"
        else
            sudo apt-get update && sudo apt-get -y install git git-lfs
            sudo apt-get -y install  openmpi-bin libopenmpi-dev

            git clone https://github.com/NVIDIA/TensorRT-LLM.git "$SCRIPT_DIR/$repo_name"

        fi
        cd "$SCRIPT_DIR/$repo_name"
        git submodule update --init --recursive
        git lfs install
        git lfs pull
        make -C docker release_build
    fi

    cd "$CURRENT_DIR"
}

build_and_compile_model () {
    set -e  # Exit on error

    echo "Running and building the model inside Docker..."

    local model_build_path_32="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_32"
    local model_build_path_16="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_16"

    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        if [ ! -d "$model_build_path_32" ]; then
            mkdir -p "$model_build_path_32"
        fi

        if [ ! -d "$model_build_path_16" ]; then
            mkdir -p "$model_build_path_16"
        fi

        if [ -z "$(ls -A "$model_build_path_32")" ] || [ -z "$(ls -A "$model_build_path_16")" ]; then
            echo "Building model with Docker..."

            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_32":/tensorrt_nvidia_build_32 \
                -v "$model_build_path_16":/tensorrt_nvidia_build_16 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/llama/build.py --model_dir /models/llama-2-7b-hf --dtype float32 --max_batch_size 1 --max_input_len 3000 --max_output_len 1024 --output_dir /tensorrt_nvidia_build_32

            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                --ipc=host \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_32":/tensorrt_nvidia_build_32 \
                -v "$model_build_path_16":/tensorrt_nvidia_build_16 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/llama/build.py --model_dir /models/llama-2-7b-hf --dtype float16 --max_batch_size 1 --max_input_len 3000 --max_output_len 1024 --output_dir /tensorrt_nvidia_build_16

            echo "Model built successfully."
        else
            echo "Engine file already exists. Skipping build."
        fi
    else
        echo "The base image does not exist locally. Exiting..."
        exit 1
    fi
}

if check_docker; then
    build_docker_image
    build_and_compile_model
else
    echo "Docker is not installed or not in the PATH"
    exit 1
fi
