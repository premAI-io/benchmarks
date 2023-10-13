python -m venv /tmp/llama2-burn/.venv
source /tmp/llama2-burn/.venv/bin/activate
python3 -m pip install -r /tmp/llama2-burn/llama-py/requirements.txt

cd /tmp/llama2-burn

# test the model
# python3 llama-py/test.py $BASE_MODEL_DIR $BASE_MODEL_TOKENIZER
# python3 llama-py/test_tokenizer.py

# dumps to params folder in repo root
python3 llama-py/dump_model.py $BASE_MODEL_DIR $BASE_MODEL_TOKENIZER

if ! which carg &>/dev/null ; then
  echo "Burn requires rust toolchain to be installed"
  echo "Download from ==> https://rustup.rs"
  exit -1 # 255
fi

# test the model weights in rust
# cargo run --bin test $BASE_MODEL_TOKENIZER params

# convert dumped text into burn compatible model
cargo run --bin convert params $MODEL_NAME

