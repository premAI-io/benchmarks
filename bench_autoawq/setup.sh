#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Main script starts here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
AWQ_WEIGHTS_FOLDER="${AWQ_WEIGHTS_FOLDER:-"./models/llama-2-7b-awq"}"

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

download_awq_weights() {
    # download the sample file if not exists
    if [ ! -d "$AWQ_WEIGHTS_FOLDER" ]; then
        huggingface-cli download TheBloke/Llama-2-7B-AWQ --local-dir ./models/llama-2-7b-autoawq --exclude "*.git*" "*.md" "Notice" "LICENSE"
    else
        echo "Weights already downloaded!"
    fi
}

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    "$PYTHON_CMD" -m pip install --upgrade pip > /dev/null
    "$PYTHON_CMD" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

download_awq_weights
