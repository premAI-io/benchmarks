#!/bin/bash

########################################################################################################
# Script: bench.sh
# Description: This script runs benchmarks AutoAWQ llama benchmark.
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
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Python is not installed."
        exit 1
    fi
}

setup() {
    echo -e "\nSetting up with $SCRIPT_DIR/setup.sh..."
    bash "$SCRIPT_DIR"/setup.sh
}

run_benchmarks() {
    local PROMPT="$1"
    local REPETITIONS="$2"
    local MAX_TOKENS="$3"
    local DEVICE="$4"
    local LOG_FILENAME="$5"
    local MODELS_DIR="$6"

    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
    "$PYTHON_CMD" "$SCRIPT_DIR"/bench.py \
        --prompt "$PROMPT" \
        --repetitions "$REPETITIONS" \
        --max_tokens "$MAX_TOKENS" \
        --log_file "$LOG_FILENAME" \
        --models_dir "$MODELS_DIR" \
        --device "$DEVICE"
}


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

# Set default values if not provided
PROMPT="${PROMPT:-"Explain what is a transformer"}"
REPETITIONS="${REPETITIONS:-10}"
MAX_TOKENS="${MAX_TOKENS:-100}"
DEVICE="${DEVICE:-'cuda'}"
LOG_FILENAME="${LOG_FILENAME:-"benchmark_$(date +'%Y%m%d%H%M%S').log"}"
MODELS_DIR="${MODELS_DIR:-"./models"}"

check_platform
check_python
setup
run_benchmarks "$PROMPT" "$REPETITIONS" "$MAX_TOKENS" "$DEVICE" "$LOG_FILENAME" "$MODELS_DIR"
