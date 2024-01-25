#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

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

clone_and_build_llama() {
    local DEVICE="$1"
    local VENV_DIR="$2"
    local SCRIPT_DIR="$3"

    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 <DEVICE> <ENV> <SCRIPT_DIR>"
        exit 1
    fi

    case "$DEVICE" in
        cuda)
            export LLAMA_CUBLAS=on
            ;;
        metal)
            export LLAMA_METAL=on
            ;;
        cpu)
            return 0
            ;;
        *)
            echo "Unsupported DEVICE: $DEVICE"
            return 1
            ;;
    esac

    local LIBLLAMA_FILE="$VENV_DIR/libllama_$DEVICE.so"

    if [ -e "$LIBLLAMA_FILE" ]; then
        echo "File $LIBLLAMA_FILE exists."
        exit 0
    fi

    # Remove existing llama.cpp directory if it exists
    if [ -d "$SCRIPT_DIR/llama.cpp" ]; then
        echo "Removing existing llama.cpp directory..."
        rm -rf "$SCRIPT_DIR"/llama.cpp
    fi

    git clone --depth=1 https://github.com/ggerganov/llama.cpp "$SCRIPT_DIR"/llama.cpp
    cd "$SCRIPT_DIR"/llama.cpp

    # Build llama.cpp
    make clean > /dev/null
    echo "Building llama.cpp..."
    make libllama.so > /dev/null
    cp libllama.so "$LIBLLAMA_FILE"
    cd "$SCRIPT_DIR"

    rm -rf "$SCRIPT_DIR"/llama.cpp
}

# Main script starts here

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DEVICE>"
    exit 1
fi

# Define directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="$1"
VENV_DIR="$SCRIPT_DIR/venv"
LIBLLAMA_FILE="$VENV_DIR/libllama_$DEVICE.so"

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

clone_and_build_llama "$DEVICE" "$VENV_DIR" "$SCRIPT_DIR"
