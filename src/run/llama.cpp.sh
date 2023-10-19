cd /tmp/llama.cpp

QUANTIZE=${QUANTIZE:-1}
if [ $QUANTIZE -eq 1 ]; then
  ./main -m /tmp/llama.cpp/models/llama-2-7b/ggml-model-q8_0.gguf -n 100 --prompt "$1" ${@:2}
else
  ./main -m /tmp/llama.cpp/models/llama-2-7b/ggml-model-f16.gguf -n 100 --prompt "$1" ${@:2}
fi
