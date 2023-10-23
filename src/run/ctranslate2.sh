if ! $MODEL_DIR; then
  echo "Provide \$MODEL_DIR with dir format 
  dir
  | model.bin
  | tokenizer.model
  "
fi

LS_CHK=`ls $MODEL_DIR &>/dev/null`
LS_RET=$?

if [ $LS_RET -eq 0 ]; then
  python3 src/custom/ctranslate2/exec.py $MODEL_DIR $@
else
  echo "Invalid \`$MODEL_DIR\` path"
fi
