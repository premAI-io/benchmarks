#!/bin/bash

##############################################################################################
# Script: run_benchmarks.sh
# Description: This script runs benchmarks for a transformer model using both 
# Rust and Python implementations. It provides options to customize the 
# benchmarks, such as the prompt, repetitions, maximum tokens, device, and NVIDIA flag.
#
# Usage: ./run_benchmarks.sh [OPTIONS]
# OPTIONS:
#   -p, --prompt      Prompt for benchmarks (default: 'Explain what is a transformer')
#   -r, --repetitions Number of repetitions for benchmarks (default: 2)
#   -m, --max_tokens  Maximum number of tokens for benchmarks (default: 100)
#   -d, --device      Device for benchmarks (possible values: 'gpu' or 'cpu', default: 'cpu')
#   --nvidia          Use NVIDIA for benchmarks (default: false)
#   -h, --help        Show this help message
##############################################################################################

set -euo pipefail

# Function to print script usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  -p, --prompt      Prompt for benchmarks (default: 'Explain what is a transformer')"
    echo "  -r, --repetitions Number of repetitions for benchmarks (default: 2)"
    echo "  -m, --max_tokens  Maximum number of tokens for benchmarks (default: 100)"
    echo "  -d, --device      Device for benchmarks (possible values: 'gpu' or 'cpu', default: 'cpu')"
    echo "  --nvidia          Use NVIDIA for benchmarks (default: false)"
    echo "  -h, --help        Show this help message"
    exit 1
}

# Function to check the platform
check_platform() {
    local platform=$(uname -s)
    if [[ "$platform" == "Linux" ]]; then
        echo "Running on Linux."
        check_cuda
    elif [[ "$platform" == "Darwin" ]]; then
        echo "Running on Mac OS."
    else
        echo "Unknown platform."
        exit 1
    fi
}

# Function to check if CUDA is available
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

# Function to check if Python exists
check_python() {
    if command -v python &> /dev/null
    then
        echo -e "\nUsing $(python --version)."
    else
        echo -e "\nPython does not exist."
        exit 1
    fi
}

# Function to check if rust is installed
check_rust() {
    if which cargo &>/dev/null ; then
        echo -e "\nRust is installed. Using $(which cargo)"
    else
        echo -e "\nRust is not installed. Please install Rust before proceeding."
        exit 1  # Error exit code
    fi
}

# Function to check if jq is installed
check_jq() {
    if ! command -v jq &> /dev/null
    then
        echo -e "\njq is not installed."
        exit 1
    fi
}

# Function to download models
download_models() {
    echo -e "\nDownloading models..."
    bash ./download.sh --models models.json --cache cache.log
}

# Function to set up
setup() {
    echo -e "\nSetting up..."
    bash ./setup.sh
}

# Function to run python benchmarks
run_benchmarks() {
    local PROMPT="$1"
    local REPETITIONS="$2"
    local MAX_TOKENS="$3"
    local DEVICE="$4"
    local USE_NVIDIA="$5"
    local LOG_FILENAME="$6"
    local DIR=$(pwd)
    local CARGO_CANDLE_FEATURES=""
    local PYTHON_DEVICE=""
    local PYTHON_NVIDIA=""
    local PLATFORM=$(uname -s)

    echo "Running benchmarks with the following parameters:"
    echo "  Prompt: $PROMPT"
    echo "  Repetitions: $REPETITIONS"
    echo "  Max Tokens: $MAX_TOKENS"
    echo "  Device: $DEVICE"
    echo "  NVIDIA: $USE_NVIDIA"

    echo "Running rust benchmarks..."
    source ./venv/bin/activate

    # Run Rust benchmarks
    if [ "$DEVICE" == "gpu" ] && [ "$PLATFORM" != "Darwin" ]; then
        TORCH_CUDA_VERSION=cu117
    fi
    cargo run --release --bin sample \
        --manifest-path="$DIR/rust_bench/llama2-burn/Cargo.toml" \
        "$DIR/models/llama-2-7b-burn/llama-2-7b-burn" \
        "$DIR/models/llama-2-7b-burn/tokenizer.model" \
        "$PROMPT" \
        $MAX_TOKENS \
        $DEVICE \
        $REPETITIONS \
        "$LOG_FILENAME"

    if [ "$DEVICE" == "cpu" ] || [ "$USE_NVIDIA" == true ]; then
        # Set features option based on $DEVICE
        [ "$DEVICE" == "gpu" ] && CARGO_CANDLE_FEATURES="--features cuda"

        cargo run --release $CARGO_CANDLE_FEATURES \
            --manifest-path="$DIR/rust_bench/llama2-candle/Cargo.toml" \
            -- --local-weights "$DIR/models/llama-2-7b-st/" \
            --repetitions "$REPETITIONS" \
            --prompt "$PROMPT" \
            --sample-len $MAX_TOKENS \
            --log-file $LOG_FILENAME
    fi 
    
    # Set options based on $DEVICE and $USE_NVIDIA
    [ "$DEVICE" == "gpu" ] && PYTHON_DEVICE="--gpu"
    [ "$USE_NVIDIA" == true ] && PYTHON_NVIDIA="--nvidia"

    cd $DIR
    echo "Running python benchmarks..."
    python ./bench.py \
        --prompt "$PROMPT" \
        --repetitions "$REPETITIONS" \
        --max_tokens $MAX_TOKENS \
        --log_file "$LOG_FILENAME" \
        $PYTHON_DEVICE \
        $PYTHON_NVIDIA
    deactivate
}

# Default values
DEFAULT_PROMPT="Explain what is a transformer"
DEFAULT_REPETITIONS=10
DEFAULT_MAX_TOKENS=100
DEFAULT_DEVICE="gpu"
USE_NVIDIA=false

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
                "gpu" | "cpu")
                    ;;
                *)
                    echo "Invalid value for --device. Please use 'gpu' or 'cpu'."
                    print_usage
                    ;;
            esac
            shift 2
            ;;
        --nvidia)
            USE_NVIDIA=true
            if [ "$DEVICE" != "gpu" ]; then
                echo "Error: The '--nvidia' flag can only be used with 'gpu' as the device."
                print_usage
            fi
            shift 1
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

# Set default values if not provided
PROMPT="${PROMPT:-$DEFAULT_PROMPT}"
REPETITIONS="${REPETITIONS:-$DEFAULT_REPETITIONS}"
MAX_TOKENS="${MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"

timestamp=$(date +"%Y%m%d%H%M%S")
log_filename="benchmark_${timestamp}.log"

check_platform
check_python
check_rust
check_jq
download_models
setup
run_benchmarks "$PROMPT" "$REPETITIONS" "$MAX_TOKENS" "$DEVICE" $USE_NVIDIA "$log_filename"