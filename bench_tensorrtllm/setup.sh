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
    local model_build_path_08="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_08"
    local model_build_path_04="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_04"


    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        if [ ! -d "$model_build_path_32" ]; then
            mkdir -p "$model_build_path_32"
            echo "Building model build (FP32 precision) with Docker..."
            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_32":/tensorrt_nvidia_build_32 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/llama/build.py --model_dir /models/llama-2-7b-hf --dtype float32 --max_batch_size 1 --max_input_len 3000 --max_output_len 1024 --output_dir /tensorrt_nvidia_build_32
        else
            echo "Engine file for Llama 2 build FP32 already exists. Skipping ..."
        fi

        if [ ! -d "$model_build_path_16" ]; then
            mkdir -p "$model_build_path_16"
            echo "Building model build (FP16 precision) with Docker..."
            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
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
        else
            echo "Engine file for Llama 2 build FP16 already exists. Skipping ..."
        fi

        if [ ! -d "$model_build_path_08" ]; then
            mkdir -p "$model_build_path_08"
            echo "Generating binaries for each of the model layers"

            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_08":/tensorrt_nvidia_build_08 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/llama/hf_llama_convert.py -i /models/llama-2-7b-hf -o /tensorrt_nvidia_build_08 --calibrate-kv-cache -t fp16 \

        elif [ ! "$(find "$model_build_path_08" -maxdepth 1 | wc -l)" -gt 2 ]; then
            echo "Building model build (FP08 precision) with Docker..."
            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_08":/tensorrt_nvidia_build_08 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/llama/build.py --bin_model_dir /tensorrt_nvidia_build_08/1-gpu --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --int8_kv_cache --output_dir /tensorrt_nvidia_build_08 --use_weight_only
        else
            if [ -d "$model_build_path_08" ] && [ -d "$model_build_path_08/1-gpu" ]; then
                echo "Engine file for Llama 2 build INT-8 already exists. Skipping ..."
            else
                echo "There is a problem with the model build directories. Please retry."
            fi
        fi

        if [ ! -d "$model_build_path_04" ]; then
            docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                -v "$CURRENT_DIR"/models:/models \
                -v "$model_build_path_04":/tensorrt_nvidia_build_04 \
                -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
                --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
                --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
                --workdir /code/tensorrt_llm \
                --hostname psqh4m1l0zhx-release \
                --name tensorrt_llm-release-paperspace \
                --tmpfs /tmp:exec \
                tensorrt_llm/release:latest \
                python3 ./examples/quantization/quantize.py --model_dir /models/llama-2-7b-hf --dtype float16 --qformat int4_awq --export_path /tensorrt_nvidia_build_04 --calib_size 32
        fi
    else
        echo "Docker image does not exist ... "
    fi
}

if check_docker; then
    build_docker_image
    build_and_compile_model
else
    echo "Docker is not installed or not in the PATH"
    exit 1
fi
