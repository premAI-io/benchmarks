#!/bin/bash

####################################################################################
# Script: setup.sh <MODELS_FOLDER>
# Description: Automates the setup of a virtual environment, clone llama burn repo,
# installs project requirements and handles model conversion.
####################################################################################

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <models_folder>"
    exit 1
fi

# Define directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_FOLDER="$1"
BURN_MODEL_INPUT_DIR=$MODELS_FOLDER/llama-2-7b-raw
BURN_FOLDER=$SCRIPT_DIR/llama2-burn
BURN_MODEL_FOLDER=$MODELS_FOLDER/llama-2-7b-burn
BURN_MODEL_NAME="llama-2-7b-burn"

check_and_create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    if [ -d "$BURN_FOLDER" ]; then
        rm -rf "$BURN_FOLDER"
    fi
    git clone --depth=1 https://github.com/premAI-io/llama2-burn.git "$BURN_FOLDER"
    pip install -r "$BURN_FOLDER"/llama-py/requirements.txt > /dev/null
fi

# Check and create llama-2-7b-burn model
if [ ! -e "$BURN_MODEL_FOLDER/$BURN_MODEL_NAME.cfg" ]; then
    check_and_create_directory "$BURN_MODEL_FOLDER"

    if [ ! -d "$BURN_MODEL_FOLDER/params" ]; then
        echo "Dumping model from $BURN_MODEL_INPUT_DIR to $BURN_MODEL_FOLDER"
        python "$BURN_FOLDER/llama-py/dump_model.py" "$BURN_MODEL_INPUT_DIR" "$BURN_MODEL_INPUT_DIR/tokenizer.model"
        mv "$(pwd)/params" "$BURN_MODEL_FOLDER"
        cp "$BURN_MODEL_INPUT_DIR/tokenizer.model" "$BURN_MODEL_FOLDER"
    else
        echo "Model already dumped at $BURN_MODEL_FOLDER/params."
    fi

    echo "Converting dumped model to burn"
    cargo run --manifest-path="$BURN_FOLDER/Cargo.toml" --bin convert -- "$BURN_MODEL_FOLDER/params" "$BURN_MODEL_NAME"
    mv "$BURN_MODEL_NAME.bin" "$BURN_MODEL_FOLDER"
    mv "$BURN_MODEL_NAME.cfg" "$BURN_MODEL_FOLDER"
    rm -r "$BURN_MODEL_FOLDER/params"
else
    echo "Model llama-2-7b-burn already exists!"
fi
