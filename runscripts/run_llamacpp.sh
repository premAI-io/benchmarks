source ./llamacpp_setup.sh
# assumes the quantized model is present as well
# and also the non-quantized but converted 7b model is present

# provide consistent flags and parameters
./main -m models/llama-2-7b/ggml-model-f16.gguf $@
