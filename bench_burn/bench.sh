#!/bin/bash

########################################################################################################
# Script: bench.sh
# Description: This script runs benchmarks Burn Llama-2 benchmark.
#
# Usage: ./bench.sh [OPTIONS]
# OPTIONS:
#   -p, --prompt      Prompt for benchmarks (default: 'Explain what is a transformer')
#   -r, --repetitions Number of repetitions for benchmarks (default: 2)
#   -m, --max_tokens  Maximum number of tokens for benchmarks (default: 100)
#   -d, --device      Device for benchmarks (possible values: 'metal', 'gpu', and 'cpu', default: 'cpu')
#   -lf, --log_file   Logging file name.
#   -md, --models_dir Models directory.
#   -h, --help        Show this help message
########################################################################################################

set -euo pipefail

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  -p, --prompt        Prompt for benchmarks (default: 'Explain what is a transformer')"
    echo "  -r, --repetitions   Number of repetitions for benchmarks (default: 2)"
    echo "  -m, --max_tokens    Maximum number of tokens for benchmarks (default: 100)"
    echo "  -d, --device        Device for benchmarks (possible values: 'metal', 'gpu', and 'cpu', default: 'cpu')"
    echo "  -lf, --log_file     Logging file name."
    echo "  -md, --models_dir   Models directory."
    echo "  -h, --help          Show this help message"
    exit 1
}

check_cuda() {
    if command -v nvcc &> /dev/null
    then
        echo -e "\nUsing CUDA"
        nvcc --version
    else
        echo -e "\nCUDA is not available."
        exit 1
    fi
}

check_rust() {
    if which cargo &>/dev/null ; then
        echo -e "\nRust is installed. Using $(which cargo)"
    else
        echo -e "\nRust is not installed. Please install Rust before proceeding."
        exit 1  # Error exit code
    fi
}

check_platform() {
    local platform
    platform=$(uname -s)
    if [[ "$platform" == "Linux" ]]; then
        echo "Running on Linux."
    elif [[ "$platform" == "Darwin" ]]; then
        echo "Running on Mac OS."
    else
        echo "Unknown platform."
        exit 1
    fi
}

check_python() {
    if command -v python &> /dev/null || command -v python3 &> /dev/null; then
        echo "Python is installed."
    else
        echo "Python is not installed."
        exit 1
    fi
}


setup() {

    # Check if Logs folder exists else Make the logs folder
    LOGS_FOLDER="$CURRENT_DIR/Logs"

    if [ -d "$LOGS_FOLDER" ]; then
        echo "Folder '$LOGS_FOLDER' already exists. Skipping."
    else
        # Create the folder
        mkdir "$LOGS_FOLDER"
        echo "'$LOGS_FOLDER' created."
    fi

    echo -e "\nSetting up with $SCRIPT_DIR/setup.sh..."
    bash "$SCRIPT_DIR/setup.sh" "$1"
}

run_benchmarks() {
    local PROMPT="$1"
    local REPETITIONS="$2"
    local MAX_TOKENS="$3"
    local DEVICE="$4"
    local LOG_FILENAME="$5"
    local MODELS_DIR="$6"

    cargo clean --manifest-path="$SCRIPT_DIR/llama2-burn/Cargo.toml"

    echo "Building burn"
    if [ "$DEVICE" == "cuda" ]; then
        export TORCH_CUDA_VERSION=cu117
        DEVICE=gpu
    fi
    cargo build --release --manifest-path="$SCRIPT_DIR/llama2-burn/Cargo.toml"
    echo "Running benchmarks"

    benchmark_output=$(
        cargo run --release --bin benchmark \
            --manifest-path="$SCRIPT_DIR/llama2-burn/Cargo.toml" \
            "$MODELS_DIR/llama-2-7b-burn/llama-2-7b-burn" \
            "$MODELS_DIR/llama-2-7b-burn/tokenizer.model" \
            "$PROMPT" \
            "$MAX_TOKENS" \
            "$DEVICE" \
            "$REPETITIONS"
    )
    mean=$(echo "$benchmark_output" | grep -oP '\d+\.\d+ ± \d+\.\d+' | awk -F ' ± ' '{print $1}')
    std=$(echo "$benchmark_output" | grep -oP '\d+\.\d+ ± \d+\.\d+' | awk -F ' ± ' '{print $2}')
    echo "burn, float16 : $(printf "%.2f" "$mean") ± $(printf "%.2f" "$std")" >> "$LOG_FILENAME"
}
# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -r|--repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        -m|--max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            case "$DEVICE" in
                "cuda" | "metal" | "cpu")
                    ;;
                *)
                    echo "Invalid value for --device. Please use 'cuda', 'gpu' or 'cpu'."
                    print_usage
                    ;;
            esac
            if [ "$DEVICE" == "cuda" ]; then
                check_cuda
            fi
            if [ "$DEVICE" == "metal" ]; then
                echo "Metal not supported!"
                exit 0
            fi
            shift 2
            ;;
        -lf|--log_file)
            LOG_FILENAME="$2"
            shift 2
            ;;
        -md|--models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

MODELS_DIR="${MODELS_DIR:-"./models"}"

check_platform
check_rust
check_python
setup "$MODELS_DIR"

# Set default values if not provided
PROMPT="${PROMPT:-"Write an essay about the transformer model architecture"}"
REPETITIONS="${REPETITIONS:-10}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DEVICE="${DEVICE:-'cuda'}"
LOG_FILENAME="${LOG_FILENAME:-"$LOGS_FOLDER/benchmark_burn_$(date +'%Y%m%d%H%M%S').log"}"

run_benchmarks "$PROMPT" "$REPETITIONS" "$MAX_TOKENS" "$DEVICE" "$LOG_FILENAME" "$MODELS_DIR"
