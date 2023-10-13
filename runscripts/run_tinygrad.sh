#!/bin/bash
source ./runscripts/tinygrad_setup_source.sh
ARGS=$@
## setup models
./setup_llama7b_model.sh
## run model
PYTHONPATH="/tmp/tinygrad" python3 /tmp/tinygrad/examples/${ARGS}
# get out of venv
deactivate
