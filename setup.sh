#!/bin/bash

################################################################################
# Script: setup_and_convert.sh
# Description: This script automates the setup of a virtual environment, 
# installs project requirements, converts and stores models.
################################################################################

set -euo pipefail

# Define directory paths
VENV_DIR="venv"
LLAMA_HF_MODEL_DIR="./models/llama-2-7b-hf"
LLAMA_ST_MODEL_DIR="./models/llama-2-7b-st"
BURN_MODEL_INPUT_DIR=$(pwd)/models/llama-2-7b-raw
BURN_FOLDER=$(pwd)/rust_bench/llama2-burn
BURN_MODEL_FOLDER=$(pwd)/models/llama-2-7b-burn
BURN_MODEL_NAME="llama-2-7b-burn"

create_and_activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        python -m venv "$VENV_DIR"
        echo "Virtual environment '$VENV_DIR' created."
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
}

install_requirements() {
    pip install -r "$1"
}

check_and_create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Check and create virtual environment
create_and_activate_venv

# Install requirements for the project
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt > /dev/null

# Check and create llama-2-7b-hf-float16 model
if [ ! -d "$LLAMA_HF_MODEL_DIR-float16" ]; then
    echo "Creating llama-2-7b-hf-float16 model..."
    ct2-transformers-converter --model "$LLAMA_HF_MODEL_DIR/" --quantization float16 --output_dir "$LLAMA_HF_MODEL_DIR-float16" --copy_files tokenizer.model
else
    echo "Model llama-2-7b-hf-float16 already exists!"
fi

# Check and create llama-2-7b-hf-float16 model
if [ ! -d "$LLAMA_HF_MODEL_DIR-int8" ]; then
    echo "Creating llama-2-7b-hf-int8 model..."
    ct2-transformers-converter --model "$LLAMA_HF_MODEL_DIR/" --quantization int8 --output_dir "$LLAMA_HF_MODEL_DIR-int8" --copy_files tokenizer.model
else
    echo "Model llama-2-7b-hf-int8 already exists!"
fi

# Check and create llama-2-7b-st model
if [ ! -d "$LLAMA_ST_MODEL_DIR" ]; then
    echo "Storing llama-2-7b-hf in safetensors format..."
    python convert_to_safetensors.py --input_dir "$LLAMA_HF_MODEL_DIR" --output_dir "$LLAMA_ST_MODEL_DIR"
else
    echo "Model llama-2-7b-hf in safetensors format already exists!"
fi

# Check and create llama-2-7b-burn model
if [ ! -e "$BURN_MODEL_FOLDER/$BURN_MODEL_NAME.cfg" ]; then
    check_and_create_directory "$BURN_MODEL_FOLDER"
    
    if [ ! -d "$BURN_MODEL_FOLDER/params" ]; then
        create_and_activate_venv
        echo "Installing requirements for dumping"
        install_requirements "$BURN_FOLDER/llama-py/requirements.txt" > /dev/null
        echo "Dumping model from $BURN_MODEL_INPUT_DIR to $BURN_MODEL_FOLDER"
        python "$BURN_FOLDER/llama-py/dump_model.py" --model-dir "$BURN_MODEL_INPUT_DIR" --output-dir "$BURN_MODEL_FOLDER"
        deactivate
    else
        echo "Model already dumped at $BURN_MODEL_FOLDER/params."
    fi

    echo "Converting dumped model to burn"
    cargo run --manifest-path="$BURN_FOLDER/Cargo.toml" --bin convert -- "$BURN_MODEL_FOLDER/params" "$BURN_MODEL_NAME" "$BURN_MODEL_FOLDER"
    cp "$BURN_MODEL_INPUT_DIR/tokenizer.model" "$BURN_MODEL_FOLDER"
else
    echo "Model llama-2-7b-burn already exists!"
fi
