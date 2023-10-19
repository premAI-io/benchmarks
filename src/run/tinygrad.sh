QUANTIZE=${QUANTIZE:-1}

if [ ${QUANTIZE} -eq 1 ] ; then
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

LS_MODEL_DIR=`ls $MODEL_DIR &>/dev/null`
LS_MODEL_RET=$?

echo "LS_MODEL_RET = $LS_MODEL_RET"

if [ $LS_MODEL_RET -eq 0 ]; then
	rm -rf /tmp/llama-2/7b
	ln -s $MODEL_DIR /tmp/llama-2/7b
else
	echo "$MODEL_DIR doesn't point to a valid path"
	exit -1
fi

LS_TOK=`ls $MODEL_DIR/../tokenizer.model`
LS_TOK_RET=$?

echo "LS_TOK_RET = $LS_TOK_RET"

if [ $LS_TOK_RET -eq 0 ]; then
	cp $MODEL_DIR/../tokenizer.model /tmp/llama-2/tokenizer.model
else
	cp $MODEL_DIR/tokenizer.model /tmp/llama-2/tokenizer.model
fi

# --profile outputs out.prof profile for the run
# the custom script will generate average timing results for usage in data table
PYTHONPATH="/tmp/tinygrad" python3 src/custom/llama_tinygrad.py --gen 2 $Q --prompt "$1" --count 100 --model "/tmp/llama-2/7b" ${@:2}
