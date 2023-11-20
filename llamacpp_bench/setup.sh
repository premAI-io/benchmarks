#!/bin/bash

################################################################################
# Script: setup.sh <DEVICE>
# Description: This script automates the setup of a virtual environment,
# installs project requirements.
################################################################################

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DEVICE>"
    exit 1
fi


# Define directory paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
DEVICE="$1"
if [ "$DEVICE" == "cuda" ]; then
    export CMAKE_ARGS=-DLLAMA_CUBLAS=on
elif [ "$DEVICE" == "metal" ]; then
    export CMAKE_ARGS=-DLLAMA_METAL=on
else
    export CMAKE_ARGS=-DLLAMA_CUBLAS=off
fi
export FORCE_CMAKE=1

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    source $VENV_DIR/bin/activate
    pip install --upgrade pip > /dev/null
else
    source $VENV_DIR/bin/activate
fi

echo "Installing requirements with CMAKE_ARGS=$CMAKE_ARGS and FORCE_CMAKE=$FORCE_CMAKE"
pip install -r $SCRIPT_DIR/requirements.txt --no-cache-dir --force-reinstall llama-cpp-python > /dev/null
