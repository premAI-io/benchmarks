git clone --depth=1 https://github.com/Gadersd/llama2-burn.git /tmp/llama2-burn

python -m venv /tmp/llama2-burn/.venv
source /tmp/llama2-burn/.venv/bin/activate
python3 -m pip install -r /tmp/llama2-burn/llama-py/requirements.txt

cd /tmp/llama2-burn/llama-py
# test the model
# python3 test.py /tmp/llama-2-7b /tmp/llama-2-7b/tokenizer.model
python3 dump_model.py /tmp/llama-2-7b /tmp/llama-2-7b/tokenizer.model
python3 test_tokenizer.py

cargo run --bin convert params llama2-7b-burn
# test the model weights in rust
# cargo run --bin test tokenizer.model params

## todo: move to separate run script

# provide correct CUDA VERSION for GPU inference
#export TORCH_CUDA_VERSION=cu113 # if running on gpu
cargo run --bin sample llama2-7b-burn tokenizer.model "Hello, I am " 10
