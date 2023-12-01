#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Function to install CTransformers with CUDA version check
install_ctransformers_cuda() {
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \(.*\),.*/\1/p')

    if [ -z "$CUDA_VERSION" ]; then
        echo "CUDA is not installed or not found."
        exit 1
    fi

    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 2 ]; }; then
        echo "Detected CUDA version >= 12.2"
        pip install ctransformers[cuda] > /dev/null
    else
        echo "Detected CUDA version < 12.2"
        CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$(which nvcc)" CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers > /dev/null
    fi
}

install_device_specific_ctransformers() {
    local DEVICE="$1"

    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <DEVICE>"
        exit 1
    fi

    case "$DEVICE" in
        cuda)
            echo "Installing CTransformers for CUDA."
            install_ctransformers_cuda
            ;;
        metal)
            echo "Installing CTransformers for Metal."
            CT_METAL=1 pip install ctransformers --no-binary ctransformers > /dev/null
            ;;
        cpu)
            echo "Installing CTransformers for CPU."
            pip install ctransformers > /dev/null
            ;;
        *)
            echo "Unsupported DEVICE: $DEVICE"
            return 1
            ;;
    esac
}

# Main script starts here.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DEVICE>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="$1"
VENV_DIR="$SCRIPT_DIR/venv"

# Build and activate the virtual environment.

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

install_device_specific_ctransformers "$DEVICE"
