#!/bin/bash
set -euo pipefail

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
        echo -e "Using CUDA\n"
        nvcc --version
    else
        echo "CUDA is not available."
        exit 1
    fi
}

# Function to check if Python exists
check_python() {
    if command -v python &> /dev/null
    then
        echo "Using $(python --version)."
    else
        echo "Python does not exist."
        exit 1
    fi
}

# Function to check if jq is installed
check_jq() {
    if ! command -v jq &> /dev/null
    then
        echo "jq is not installed."
        exit 1
    fi
}

# Function to download models
download_models() {
    echo -e "Downloading models...\n"
    bash ./download.sh ./models.json
}

# Function to set up
setup() {
    echo -e "Setting up...\n"
    bash ./setup.sh
}

# Function to run python benchmarks
run_benchmarks() {
    echo -e "Running python benchmarks...\n"
    source ./venv/bin/activate
    python ./bench.py
    deactivate
}

check_platform
check_python
check_jq
download_models
setup
run_benchmarks