# provide correct CUDA VERSION for GPU inference
#export TORCH_CUDA_VERSION=cu113 # if running on gpu
cd /tmp/llama2-burn

N_TOKENS:=100
DEVICE_TYPE:="cpu"

cargo run --bin sample $MODEL_NAME $MODEL_TOKENIZER $PROMPT $N_TOKENS $DEVICE_TYPE
