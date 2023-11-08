#!/bin/bash

echo "Downloading models..."
bash ./download.sh ./models.json

echo "Setting up..."
bash ./setup.sh

echo "Running python benchmarks..."
source ./venv/bin/activate
python ./bench.py
deactivate