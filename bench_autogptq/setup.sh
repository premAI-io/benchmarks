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
CURRENT_DIR="$(pwd)"

GPTQ_WEIGHTS_FOLDER="${GPTQ_WEIGHTS_FOLDER:-"$CURRENT_DIR/models/llama-2-7b-autogptq"}"

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

download_gptq_weights() {
    # download the sample file if not exists
    if [ ! -d "$GPTQ_WEIGHTS_FOLDER" ]; then
        echo "Downloading GPT weights..."
        huggingface-cli download TheBloke/Llama-2-7B-GPTQ --local-dir "$GPTQ_WEIGHTS_FOLDER" --exclude "*.git*" "*.md" "Notice" "LICENSE"
    else
        echo "Weights already downloaded!"
    fi
}

check_python

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip and install requirements
"$PYTHON_CMD" -m pip install --upgrade pip > /dev/null
"$PYTHON_CMD" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null

download_gptq_weights
