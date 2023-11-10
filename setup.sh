#!/bin/bash

# Check if the venv directory exists
if [ ! -d "venv" ]; then
 # Create a new venv environment
 python -m venv venv
 echo "Virtual environment 'venv' created."
 source venv/bin/activate
 CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
    pip install -r requirements.txt
else
 echo "Using already existing environment 'venv'."
 source venv/bin/activate
fi
if [ ! -d "./models/llama-2-7b-hf-float16" ]; then
    echo "Creating llama-2-7b-hf-float16..."
    ct2-transformers-converter --model ./models/llama-2-7b-hf/ --quantization float16 --output_dir ./models/llama-2-7b-hf-float16 --copy_files tokenizer.model
else
 echo "Model llama-2-7b-hf-bfloat16 already exists!."
fi

ST_FOLDER=./models/llama-2-7b-st
if [ ! -d "$ST_FOLDER" ]; then
  source venv/bin/activate
  echo "Storing llama-2-7b-hf in safetensors format!."
  python convert_to_safetensors.py \
  --input_dir ./models/llama-2-7b-hf \
  --output_dir $ST_FOLDER
else
 echo "Model llama-2-7b-hf in safetensors format already exists!."
fi

BURN_MODEL_INPUT_DIR=$(pwd)/models/llama-2-7b-raw
BURN_FOLDER=$(pwd)/rust_bench/llama2-burn
BURN_MODEL_FOLDER=$(pwd)/models/llama-2-7b-burn
BURN_MODEL_NAME=llama-2-7b-burn

if [ ! -e "$BURN_MODEL_FOLDER/$BURN_MODEL_NAME.cfg" ]; then
  mkdir -p "$BURN_MODEL_FOLDER"
  if [ ! -d "$BURN_MODEL_FOLDER/params" ]; then
    source venv/bin/activate 
    echo "Installing requirements for dumping"
    python -m pip install \
      -r "$BURN_FOLDER/llama-py/requirements.txt" > /dev/null
    echo "Dumping model from $BURN_MODEL_INPUT_DIR to $BURN_MODEL_FOLDER"
    python "$BURN_FOLDER/llama-py/dump_model.py" \
      --model-dir "$BURN_MODEL_INPUT_DIR" \
      --output-dir "$BURN_MODEL_FOLDER"
    deactivate
  else
    echo "Model already dumped at $BURN_MODEL_FOLDER/params."
  fi
  echo "Converting dumped model to burn"
  cargo run \
    --manifest-path="$BURN_FOLDER/Cargo.toml" \
    --bin convert \
    -- \
    "$BURN_MODEL_FOLDER/params" \
    "$BURN_MODEL_NAME"  \
    "$BURN_MODEL_FOLDER"
  cp "$BURN_MODEL_INPUT_DIR/tokenizer.model" "$BURN_MODEL_FOLDER"
else
  echo "Model llama-2-7b-burn already exists!."
fi