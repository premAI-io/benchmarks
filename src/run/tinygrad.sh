NOT_QUANTIZE=${NOT_QUANTIZE:-0}

if [ $NOT_QUANTIZE ] ; then
	Q="--quantize"
else
	Q=""
fi

# required env flags, MODEL_DIR where 7B llama 2 model and tokenizer is present
if ! [ $MODEL_DIR ]  ; then
	echo "Env variable \$MODEL_DIR not provided!"
	echo "MODEL_DIR structure
	dir
	| tokenizer.model
	| xxx.000.pth
	"
	exit -1
fi

mkdir -p /tmp/llama-2

if [ `ls $MODEL_DIR` ]; then
	ln -s $MODEL_DIR /tmp/llama-2/7b
else
	echo "$MODEL_DIR doesn't point to a valid path"
	exit -1
fi

if [ `ls $MODEL_DIR/../tokenizer.model` ]; then
	cp $MODEL_DIR/../tokenizer.model /tmp/llama-2/tokenizer.model
else
	cp $MODEL_DIR/tokenizer.model /tmp/llama-2/tokenizer.model
fi

# --profile outputs out.prof profile for the run
# the custom script will generate average timing results for usage in data table
PYTHONPATH="/tmp/tinygrad" python3 src/custom/llama_tinygrad.py --gen 2 $Q --prompt "$1" --count 100 --model "/tmp/llama-2/7b" ${@:2}
