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

# Build the docker image
build_docker_image () {
    local repo_name="TensorRT-LLM"

    # Check if the Docker image exists
    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        echo "Image 'tensorrt_llm/release:latest' already exists."
    else
        if [ -d "$SCRIPT_DIR/$repo_name" ]; then
            echo "Repo already cloned"
        else
            sudo apt-get update && sudo apt-get -y install git git-lfs
            sudo apt-get -y install openmpi-bin libopenmpi-dev

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

build_engine () {
    local MODEL_NAME="$1"
    local PRECISION="$2"

    # Set the default folder paths for HF and engines
    LLAMA2_WEIGHTS_FOLDER="/mnt/models/llama-2-7b-chat"
    MISTRAL_WEIGHTS_FOLDER="/mnt/models/mistral-7b-v0.1-instruct"

    # Files to run inside docker
    CONVERT_CHECKPOINT_PATH="/app/tensorrt_llm/examples/llama/convert_checkpoint.py"
    QUANT_PATH="/app/tensorrt_llm/examples/quantization/quantize.py"

    HF_MODEL_DIR=""
    ENGINE_DIR=""
    OUT_DIR=""

    if [ "$MODEL_NAME" = "llama" ]; then
        HF_MODEL_DIR="$LLAMA2_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$LLAMA2_WEIGHTS_FOLDER-trt-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/llama-2-7b-chat-trt-$PRECISION"

    elif [ "$MODEL_NAME" = "mistral" ]; then
        HF_MODEL_DIR="$MISTRAL_WEIGHTS_FOLDER-hf"
        ENGINE_DIR="$MISTRAL_WEIGHTS_FOLDER-trt-$PRECISION"
        OUT_DIR="$CURRENT_DIR/models/mistral-7b-v0.1-instruct-trt-$PRECISION"
    else
        echo "Invalid MODEL_NAME. Supported values: 'llama', 'mistral'"
        exit 1
    fi

    if [ ! -d "$OUT_DIR" ]; then
        echo "=> Converting first to .safetensorts format"
        if [ "$PRECISION" = "float16" ]; then
            docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                -v "$CURRENT_DIR/models":/mnt/models \
                tensorrt_llm/release:latest \
                python3 "$CONVERT_CHECKPOINT_PATH" --model_dir "$HF_MODEL_DIR" \
                    --output_dir "$ENGINE_DIR" \
                    --dtype float16

        elif [ "$PRECISION" = "float32" ]; then
            echo "Float32 is not currently support"
            echo "checkout issue: https://github.com/NVIDIA/TensorRT-LLM/issues/1485"
            exit 1

        elif [ "$PRECISION" = "int8" ]; then
            docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                -v "$CURRENT_DIR/models":/mnt/models \
                tensorrt_llm/release:latest \
                python3 "$CONVERT_CHECKPOINT_PATH" --model_dir "$HF_MODEL_DIR" \
                    --output_dir "$ENGINE_DIR" \
                    --dtype float16 \
                    --use_weight_only \
                    --weight_only_precision int8

        elif [ "$PRECISION" = "int4" ]; then
            docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                -v "$CURRENT_DIR/models":/mnt/models \
                tensorrt_llm/release:latest \
                python3 "$QUANT_PATH" --model_dir "$HF_MODEL_DIR" \
                    --dtype float16 \
                    --qformat int4_awq \
                    --awq_block_size 128 \
                    --output_dir "$ENGINE_DIR" \
                    --calib_size 32
        else
            echo "No such precision exists."
            exit 1
        fi

        # Now build the engine
        echo "Finally converting to .engine format"

        docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "$CURRENT_DIR/models":/mnt/models \
            tensorrt_llm/release:latest \
            trtllm-build --checkpoint_dir "$ENGINE_DIR" --output_dir "$ENGINE_DIR" --gemm_plugin float16
    else
        echo "Engine file already exists"
    fi
}

build_and_compile_all_engines () {
    local MODEL_NAME="$1"
    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        build_engine "$MODEL_NAME" "float16"
        build_engine "$MODEL_NAME" "int8"
        build_engine "$MODEL_NAME" "int4"
    else
        echo "Docker image does not exist, please build the docker image first ..."
    fi
}

# Main entrypoint

MODEL_NAME="${1:-"llama"}"

if check_docker; then
    build_docker_image
    build_and_compile_all_engines "$MODEL_NAME"
else
    echo "Docker is not installed or not in the PATH, please make sure docker is installed properly ..."
    exit 1
fi
