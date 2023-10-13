if ${NOT_QUANTIZE:-0} ; then
	Q="--quantize"
else
	Q=""
fi

# required env flags, MODEL_DIR where 7B llama 2 model and tokenizer is present
# we assume $MODEL_DIR is a directory
# default to "/tmp/tinygrad/weights/LLaMA-2/7B"
ln -s $MODEL_DIR/tokenizer.model $MODEL_DIR/../tokenizer.model

# --profile outputs out.prof profile for the run
# the custom script will generate average timing results for usage in data table
PYTHONPATH="/tmp/tinygrad" python3 src/custom/llama_tinygrad.py --gen 2 $Q --prompt $@ --count 100 --model $MODEL_DIR
