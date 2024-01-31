#!/bin/bash

########################################################################################################
# Script: bench.sh
# Description: This script runs benchmarks TinyGrad llama benchmark.
#
# Usage: ./bench.sh [OPTIONS]
# OPTIONS:
#   -p, --prompt      Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')
#   -r, --repetitions Number of repetitions for benchmarks (default: 10)
#   -m, --max_tokens  Maximum number of tokens for benchmarks (default: 512)
#   -d, --device      Device for benchmarks (possible values: 'metal', 'cuda', and 'cpu', default: 'cuda')
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
    echo "  -p, --prompt        Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')"
    echo "  -r, --repetitions   Number of repetitions for benchmarks (default: 10)"
    echo "  -m, --max_tokens    Maximum number of tokens for benchmarks (default: 512)"
    echo "  -d, --device        Device for benchmarks (possible values: 'metal', 'cuda', and 'cpu', default: 'cuda')"
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
    bash "$SCRIPT_DIR"/setup.sh "$1"
}

run_llama_experiment() {
    models_dir=$1
    script_dir=$2
    prompt=$3
    max_tokens=$4
    repetitions=$5
    device=$6

    if [ "$device" != "cuda" ]; then
        export CUDA_VISIBLE_DEVICES=""
    fi

    declare -a tokens_per_second_array=()

    for ((i=1; i<=repetitions; i++)); do
        tokens_per_second=$("$PYTHON_CMD" "$script_dir/tinygrad/examples/tiny.py" \
            --model "$models_dir/llama-2-7b-raw" \
            --prompt "$prompt" \
            --count "$max_tokens" \
            --timing \
            | grep -E 'total [0-9]+[.][0-9]+ ms, [0-9]+[.][0-9]+ tok/sec' \
            | awk -F '[:, ]' '{ sum += $(NF-1); count++ } END { if (count > 0) print sum/count }'
        )
        tokens_per_second_array+=("$tokens_per_second")
    done

    # Return the array of values
    echo "${tokens_per_second_array[@]}"
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

    # Assign the result to an array variable
    # shellcheck disable=SC2207
    result_array=($(run_llama_experiment "$MODELS_DIR" "$SCRIPT_DIR" "$PROMPT" "$MAX_TOKENS" "$REPETITIONS" "$DEVICE"))

    total=0
    for value in "${result_array[@]}"; do
        total=$(echo "$total + $value" | bc -l)
    done
    mean=$(echo "$total / ${#result_array[@]}" | bc -l)

    sum_squared_diff=0
    for value in "${result_array[@]}"; do
        diff=$(echo "$value - $mean" | bc -l)
        sum_squared_diff=$(echo "$sum_squared_diff + ($diff * $diff)" | bc -l)
    done
    variance=$(echo "$sum_squared_diff / ${#result_array[@]}" | bc -l)
    std=$(echo "sqrt($variance)" | bc -l)
    echo "tinygrad, float16 : $(printf "%.2f" "$mean") ± $(printf "%.2f" "$std")" >> "$LOG_FILENAME"
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

check_platform
check_python
setup "$DEVICE"

# Set default values if not provided
PROMPT="${PROMPT:-"Write an essay about the transformer model architecture"}"
REPETITIONS="${REPETITIONS:-10}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DEVICE="${DEVICE:-'cuda'}"
LOG_FILENAME="${LOG_FILENAME:-"$LOGS_FOLDER/benchmark_pytorch_$(date +'%Y%m%d%H%M%S').log"}"
MODELS_DIR="${MODELS_DIR:-"./models"}"

run_benchmarks "$PROMPT" "$REPETITIONS" "$MAX_TOKENS" "$DEVICE" "$LOG_FILENAME" "$MODELS_DIR"
