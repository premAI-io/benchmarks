# provide correct CUDA VERSION for GPU inference
#export TORCH_CUDA_VERSION=cu113 # if running on gpu

cargo run --bin sample $MODEL_NAME $MODEL_TOKENIZER $PROMPT $N_TOKENS $DEVICE_TYPE
