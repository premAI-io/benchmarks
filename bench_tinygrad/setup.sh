#!/bin/bash

################################################################################
# Script: setup.sh
# Description: Automates the setup of a virtual environment and installs project
# requirements.
################################################################################

set -euo pipefail

# Define directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    git clone --depth=1 https://github.com/tinygrad/tinygrad.git "$SCRIPT_DIR"/tinygrad
    cd "$SCRIPT_DIR"/tinygrad
    pip install -e . > /dev/null
    pip install sentencepiece > /dev/null
    cd ..
fi
