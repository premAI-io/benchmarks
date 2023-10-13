# benchmarks
MLOps Engines, Frameworks, and Languages benchmarks over main stream AI Models.

## Usage for running scripts to get performance info

the runscripts are now going to be outdated I am still keeping them around as they are useful for reference.

### tinygrad

We provide custom script to run inference, and require providing path to MODEL_DIR to ensure correct handling of model setup.
The script should report average time in tokens/sec.

Note: doesn't require a quantized model it just performs dynamic quantization to when loading weights.

Also: All scripts may require a `chmod +x` if they are being executed, after cloning.

```sh
source ./src/setup/tinygrad.sh
# set QUANTIZE=0 to not use int8 quantization
# mac m1 and beyond defaults to "METAL" backend
# use CPU=1 to use CPU backend
QUANTIZE=1 MODEL_DIR="<path/to/llama2-7b-model>" ./src/run/tinygrad.sh "prompt"
```

### burn

Doesn't support quantization, and can be a bit buggy with backends other than torch.
This is the least properly tested pipeline atm. (lmk of any bugs)

```sh
./src/setup/burn.sh # clone llama2-burn into /tmp/llama2-burn
# converts the model to burn model
# provide the model dir and model tokenizer to be converted
# eg: https://huggingface.co/meta-llama/Llama-2-7b
# and also a model name for specifying what converted model binary name
BASE_MODEL_DIR="<model-dir>" BASE_MODEL_TOKENIZER="<model-dir>/tokenizer.model" MODEL_NAME="llama-2-7b-burn" ./src/convert/burn.sh
# run the actual model
# n_toks is 100 by default
MODEL_NAME="llama-2-7b-burn" MODEL_TOKENIZER="<model-dir>/tokenizer.model" PROMPT="prompt" DEVICE_TYPE="cpu" ./src/run/burn.sh
```

### llama.cpp

```sh
```

### candle

```sh
```

## ML Engines: Feature Table

| Features                    | pytorch | burn | llama.cpp | candle | tinygrad | onnxruntime | CTranslate2 |
| --------------------------- | ------- | ---- | --------- | ------ | -------- | ----------- | ----------- |
| Inference support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 16-bit quantization support | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 8-bit quantization support  | ✅      | ❌   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 4-bit quantization support  | ✅      | ❌   | ✅        | ✅     | ❌       | ❌          | ❌          |
| 2/3bit quantization support | ✅      | ❌   | ✅        | ✅     | ❌       | ❌          | ❌          |
| CUDA support                | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| ROCM support                | ✅      | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Intel OneAPI/SYCL support   | ✅**    | ❌   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅        | ✅***  | ✅       | ✅          | ✅          |
| BLAS support(CPU)           | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Training support            | ✅      | ✅   | ❌*       | ✅     | ❌       | ❌          | ✅          |

## Benchmarking ML Engines

### Consumer Hardware Inference:
#### M1 Pro Mac 16GB Variant
#### LLAMA2-7B
#### mean of runs: 24 (with outliers removed)

| engines             | (cpu) tokens/sec                | (metal/gpu) tokens/sec     |
| -------             | ----------------                | ----------------------     |
| pytorch(8bit)       |                                 |                            |
| pytorch(4bit)       |                                 |                            |
| burn(torch)(16bit)  | quantization not-supported      | quantization not-supported |
| llama.cpp(8bit)     | 13.2                            | 21.5                       |
| llama.cpp(4bit)     | 13.2                            | 21.5                       |
| candle(8bit)        | 9.2                             | metal not supported yet!   |
| candle(4bit)        | 9.2                             | metal not supported yet!   |
| CTranslate2(8bit)   | 12.3                            | metal not supported yet!   |
| tinygrad(8bit)      | 0.75                            | 7.8                        |

*(data updated: 12th October 2023)

<!-- TODO(swarnim)
### A100 Inference:
#### LLAMA-B

| engines                    | performance |
| -------------------------- | ----------- |
| pytorch                    |             |
| fastertransformer          |             |
| pytorch(tensor-rt)         |             |
| pytorch(LLM.int8 CUDA only)|             |
| burn(wgpu)                 |             |
| burn(torch)                |             |
| ggml(cuda)                 |             |
| candle                     |             |
| tinygrad                   |             |
| CTranslate2                |             |

*(data updated: )
-->

### TODO: Operator-based performance benchmarking

This is a much rougher but arguably more represenatative example of inference engine performance,
using just a single operator across different engines.

