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