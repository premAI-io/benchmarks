cd ./src/custom/llama_candle

MODEL_SPEC="7b${QUANTIZE:-q8}"
echo "Model Used: $MODEL_SPEC"

# n tokens and etc can be passed as flags as well,
# use --help to know more
cargo run --features accelerate --release -- --which $MODEL_SPEC --prompt $PROMPT
