# source this!
git clone --depth=1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp

python -m venv /tmp/llama.cpp/.venv
source /tmp/llama.cpp/.venv/bin/activate
python3 -m pip install -r requirements.txt

# copy model from /tmp/llama-2-7b
if [ (ls /tmp/llama.cpp/models/llama-2-7b >/dev/null) ]; then
  echo "assuming models/llama-2-7b is correct!"
else
  echo "copy to models/llama-2-7b please!"
fi

python3 convert.py models/llama-2-7b/
./quantize ./models/llama-2-7b/ggml-model-f16.gguf ./models/llama-2-7b/ggml-model-q8_0.gguf q8_0

make -j