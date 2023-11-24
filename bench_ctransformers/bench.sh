#!/bin/bash

########################################################################################################
# Script: bench.sh
# Description: This script runs benchmarks llama.cpp llama benchmark.
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
    if command -v python &> /dev/null
    then
        echo -e "\nUsing $(python --version)."
    else
        echo -e "\nPython does not exist."
        exit 1
    fi
}

setup() {
    echo -e "\nSetting up with $SCRIPT_DIR/setup.sh..."
    bash "$SCRIPT_DIR"/setup.sh "$1"
}

