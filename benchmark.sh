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

# Function to check if rust is installed
check_rust() {
    if which cargo &>/dev/null ; then
        echo "Rust is installed. Using $(which cargo)"
    else
        echo "Rust is not installed. Please install Rust before proceeding."
        exit 1  # Error exit code
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

    PROMPT="Explain what is a transformer"
    REPETITIONS=2
    MAX_TOKENS=100
    DIR=$(pwd)
    
    echo -e "Running rust benchmarks...\n"

    cargo run --release --bin sample \
        --manifest-path="$DIR/rust_bench/llama2-burn/Cargo.toml" \
        "$DIR/models/llama-2-7b-burn/llama-2-7b-burn" \
        "$DIR/models/llama-2-7b-burn/tokenizer.model" \
        "$PROMPT" \
        $MAX_TOKENS \
        gpu \
        $REPETITIONS

    cargo run --release --features cuda \
        --manifest-path="$DIR/rust_bench/llama_candle/Cargo.toml" \
        -- --local-weights "$DIR/models/llama-2-7b-st/" \
        --repetitions "$REPETITIONS" \
        --prompt "$PROMPT" \
        --sample-len $MAX_TOKENS

    cd $DIR
    echo -e "Running python benchmarks...\n"
    source ./venv/bin/activate
    python ./bench.py \
        --prompt "$PROMPT" \
        --repetitions "$REPETITIONS" \
        --max_tokens $MAX_TOKENS
    deactivate
}

check_platform
check_python
check_rust
check_jq
download_models
setup
run_benchmarks