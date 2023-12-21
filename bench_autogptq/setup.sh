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
GPTQ_WEIGHTS_FOLDER="${GPTQ_WEIGHTS_FOLDER:-"./models/llama-2-7b-gptq"}"

download_gptq_weights() {
    # download the sample file if not exists
    if [ ! -d "$GPTQ_WEIGHTS_FOLDER" ]; then
    huggingface-cli download TheBloke/Llama-2-7B-GPTQ --local-dir ./models/llama-2-7b-autogptq --exclude "*.git*" "*.md" "Notice" "LICENSE"
    else
        echo "Weights already downloaded!"
    fi
}


if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir > /dev/null
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

download_gptq_weights
