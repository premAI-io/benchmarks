git clone --depth=1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp

pushd /tmp/llama.cpp &>/dev/null

MODEL_DIR=${MODEL_DIR:-/tmp/llama-2-7b-gguf}

LS_CHECK=`ls $MODEL_DIR &>/dev/null`
LS_RET=$?

if [ $LS_RET -ne 0 ]; then
	echo "Models not provided at $MODEL_DIR"
	echo "Consider providing a different \`MODEL_DIR\`"
	popd &>/dev/null
	return
fi

# copy model folder
ln -s /tmp/llama-2-7b-gguf /tmp/llama.cpp/models/llama-2-7b

LS_CHECK=`ls /tmp/llama.cpp/models/llama-2-7b &>/dev/null`
LS_RET=$?

# copy model from /tmp/llama-2-7b
if [ $LS_RET -eq 0 ]; then
  echo "Assuming /tmp/llama.cpp/models/llama-2-7b is correct!"
	LS_CHECK=`ls /tmp/llama.cpp/models/llama-2-7b/ggml-model-q8_0.gguf &>/dev/null`
	LS_RET=$?
	if [ $LS_RET -eq 0 ]; then
		echo "Found prequantized model at, \`./models/llama-2-7b/ggml-model-q8_0.gguf\`"
	else
		python -m venv /tmp/llama.cpp/.venv
		source /tmp/llama.cpp/.venv/bin/activate
		python3 -m pip install -r requirements.txt
		python3 convert.py models/llama-2-7b/
		./quantize ./models/llama-2-7b/ggml-model-f16.gguf ./models/llama-2-7b/ggml-model-q8_0.gguf q8_0
		deactivate
	fi
else
	echo "Failed to create a link at \`/tmp/llama.cpp/models/llama-2-7b\`"
	echo "Consider moving or copying files manually!"
	popd &>/dev/null
	return
fi

make -j

popd &>/dev/null
