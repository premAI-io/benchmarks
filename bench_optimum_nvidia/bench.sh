#!/bin/bash

########################################################################################################
# Script: bench.sh
# Description: This script runs benchmarks HF Optimum Nvidia benchmark.
#
# Usage: ./bench.sh [OPTIONS]
# OPTIONS:
#   -p, --prompt        Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')
#   -r, --repetitions   Number of repetitions for benchmarks (default: 10)
#   -m, --max_tokens    Maximum number of tokens for benchmarks (default: 512)
#   -d, --device        Device for benchmarks (possible values: 'metal', 'cuda', and 'cpu', default: 'cuda')
#   -n, --model_name    The name of the model to benchmark (possible values: 'llama' for using Llama2, 'mistral' for using Mistral 7B v0.1)
#   -lf, --log_file     Logging file name.
#   -h, --help          Show this help message
########################################################################################################

set -euo pipefail

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  -p, --prompt        Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')"
    echo "  -r, --repetitions   Number of repetitions for benchmarks (default: 10)"
    echo "  -m, --max_tokens    Maximum number of tokens for benchmarks (default: 512)"
    echo "  -d, --device        Device for benchmarks (possible values: 'metal', 'cuda', and 'cpu', default: 'cuda')"
    echo "  -n, --model_name    The name of the model to benchmark (possible values: 'llama' for using Llama2, 'mistral' for using Mistral 7B v0.1)"
    echo "  -lf, --log_file     Logging file name."
    echo "  -h, --help          Show this help message"
    exit 1
}

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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

setup() {
    local MODEL_NAME="${1:-llama}"
    echo -e "\nSetting up with $SCRIPT_DIR/setup.sh..."
    bash "$SCRIPT_DIR/setup.sh" "$MODEL_NAME"
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
                    echo "Invalid value for --device. Please use 'cuda', 'cpu' or 'metal'."
                    print_usage
                    ;;
            esac
            if [ "$DEVICE" == "cuda" ]; then
                check_cuda
            else
                echo "Not supported for $DEVICE"
                exit 1
            fi
            shift 2
            ;;
        -n|--model_name)
            MODEL_NAME="$2"
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

check_platform
setup "$MODEL_NAME"

# Set default values if not provided
PROMPT="${PROMPT:-"Write an essay about the transformer model architecture"}"
REPETITIONS="${REPETITIONS:-10}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DEVICE="${DEVICE:-'cuda'}"
MODEL_NAME="${MODEL_NAME:-"llama"}"


docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -v "$CURRENT_DIR:/mnt/benchmarks" \
    -it huggingface/optimum-nvidia:latest \
    python3 -u "/mnt/benchmarks/bench_optimum_nvidia/bench.py" \
        --prompt "$PROMPT" \
        --repetitions "$REPETITIONS" \
        --max_tokens "$MAX_TOKENS" \
        --model_name "$MODEL_NAME" \
        --device "$DEVICE"
