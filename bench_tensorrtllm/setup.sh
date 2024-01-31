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


# build and compile different models

build_engine_float32 () {
    local model_build_path_32="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_fp32"

    if [ ! -d "$model_build_path_32" ]; then
        mkdir -p "$model_build_path_32"
        echo "Building the model engine file for fp32 precision ..."
        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_32":/tensorrt_nvidia_build_32 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/llama/build.py \
                --model_dir /mnt/models/llama-2-7b-hf \
                --dtype float32 \
                --max_batch_size 1 \
                --max_input_len 3000 \
                --max_output_len 1024 \
                --output_dir /tensorrt_nvidia_build_32
    else
        echo "Engine file for Llama 2 fp32 precision already exists. Skipping ..."
    fi
}

build_engine_float16 () {
    local model_build_path_16="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_fp16"

    if [ ! -d "$model_build_path_16" ]; then
        mkdir -p "$model_build_path_16"
        echo "Building the model engine file for fp16 precision ..."
        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_16":/tensorrt_nvidia_build_16 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/llama/build.py \
                --model_dir /mnt/models/llama-2-7b-hf \
                --dtype float16 \
                --max_batch_size 1 \
                --max_input_len 3000 \
                --max_output_len 1024 \
                --output_dir /tensorrt_nvidia_build_16
    else
        echo "Engine file for Llama 2 fp16 precision already exists. Skipping ..."
    fi
}

build_engine_int8 () {

    local model_build_path_08="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_int8"

    if [ ! -d "$model_build_path_08" ]; then
        mkdir -p "$model_build_path_08"
        echo "Generating binaries for each model layers in mixed fp16-int8 precision ..."

        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_08":/tensorrt_nvidia_build_08 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/llama/hf_llama_convert.py -i /mnt/models/llama-2-7b-hf \
                -o /tensorrt_nvidia_build_08 \
                --calibrate-kv-cache -t fp16
    fi


    # now check if the folder exists but not the engine file
    if [ -d "$model_build_path_08" ] && [ ! "$(find "$model_build_path_08" -maxdepth 1 | wc -l)" -gt 2 ]; then
        echo "Building the model engine file for fp16-int8 mixed precision ..."
        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_08":/tensorrt_nvidia_build_08 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/llama/build.py \
                --bin_model_dir /tensorrt_nvidia_build_08/1-gpu \
                --dtype float16 \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --int8_kv_cache \
                --use_weight_only \
                --output_dir /tensorrt_nvidia_build_08
    else
        if [ -d "$model_build_path_08" ] && [ -d "$model_build_path_08/1-gpu" ]; then
            echo "Engine file for Llama 2 build INT-8 already exists. Skipping ..."
        else
            echo "There is a problem with the model build directories. Please retry."
        fi
    fi
}

build_engine_int4 () {
    local model_build_path_04="$CURRENT_DIR/models/llama-2-7b-nvidia_tensorrt_build_int4"

    if [ ! -d "$model_build_path_04" ]; then
        mkdir -p "$model_build_path_04"
        echo "Generating binaries for each model layers in mixed fp16-int4 precision ..."

        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_04":/tensorrt_nvidia_build_04 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/quantization/quantize.py --model_dir /mnt/models/llama-2-7b-hf \
                --dtype float16 \
                --qformat int4_awq \
                --export_path /tensorrt_nvidia_build_04 \
                --calib_size 32

    fi

    # now build the engine file
    if [ -d "$model_build_path_04" ] && [ ! "$(find "$model_build_path_04" -maxdepth 1 | wc -l)" -gt 3 ]; then
        echo "Building the model engine file for fp16-int4 mixed precision ..."
        docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
            --gpus all \
            -v "$CURRENT_DIR"/models:/mnt/models \
            -v "$model_build_path_04":/tensorrt_nvidia_build_04 \
            -v "$SCRIPT_DIR"/TensorRT-LLM:/code/tensorrt_llm \
            --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
            --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
            --workdir /app/tensorrt_llm \
            --hostname psqh4m1l0zhx-release \
            --name tensorrt_llm-release-prem \
            --tmpfs /tmp:exec \
            tensorrt_llm/release:latest \
            python3 ./examples/llama/build.py --model_dir /mnt/models/llama-2-7b-hf \
                --quant_ckpt_path /tensorrt_nvidia_build_04/llama_tp1_rank0.npz \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --output_dir /tensorrt_nvidia_build_04
    else
        if [ -d "$model_build_path_04" ] && [ -d "$model_build_path_04" ]; then
            echo "Engine file for Llama 2 build int4 already exists. Skipping ..."
        else
            echo "There is a problem with the model build directories. Please retry ..."
        fi
    fi
}


# Build all the engines one by one

build_and_compile_all_engines () {
    if docker image inspect tensorrt_llm/release:latest &> /dev/null; then
        build_engine_float32
        build_engine_float16
        build_engine_int8
        build_engine_int4
    else
        echo "Docker image does not exist, please build the docker image first ..."
    fi
}

# Main entrypoint

if check_docker; then
    build_docker_image
    build_and_compile_all_engines
else
    echo "Docker is not installed or not in the PATH, please make sure, docker is installed properly ..."
    exit 1
fi
