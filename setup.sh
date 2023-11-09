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