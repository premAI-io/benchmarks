#!/bin/bash

################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: This script automates the setup of a virtual environment,
# installs project requirements, converts model.
################################################################################

set -euo pipefail

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
    if docker image inspect prem-ctranslate2:latest &> /dev/null; then
        echo "Image prem-ctranslate2 already exists"
    else
        docker build -t prem-ctranslate2 "$SCRIPT_DIR/."
    fi
}

build_and_compile_model () {
    set -e  # Exit on error
    echo "Running and building the model inside Docker..."

    local model_build_path_32="$CURRENT_DIR/models/llama-2-7b-ctranslate2-fp32"
    local model_build_path_16="$CURRENT_DIR/models/llama-2-7b-ctranslate2-fp16"
    local model_build_path_08="$CURRENT_DIR/models/llama-2-7b-ctranslate2-int8"

    if docker image inspect prem-ctranslate2:latest &> /dev/null; then
        if [ ! -d "$model_build_path_32" ]; then
            docker run -it --rm \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                prem-ctranslate2:latest \
                ct2-transformers-converter --model /models/llama-2-7b-hf --quantization float32 --output_dir /models/llama-2-7b-ctranslate2-fp32 --copy_files tokenizer.model --force
            echo "Model build for FP32 ran successfully ... "
        fi

        if [ ! -d "$model_build_path_16" ]; then
            docker run -it --rm \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                prem-ctranslate2:latest \
                ct2-transformers-converter --model /models/llama-2-7b-hf --quantization float16 --output_dir /models/llama-2-7b-ctranslate2-fp16 --copy_files tokenizer.model --force
            echo "Model build for FP32 ran successfully ... "
        fi

        if [ ! -d "$model_build_path_08" ]; then
            docker run -it --rm \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                prem-ctranslate2:latest \
                ct2-transformers-converter --model /models/llama-2-7b-hf --quantization int8_float16 --output_dir /models/llama-2-7b-ctranslate2-int8 --copy_files tokenizer.model --force
            echo "Model build for FP32 ran successfully ... "
        fi
    else
        echo "Image does not exist locally. Exiting ... "
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
