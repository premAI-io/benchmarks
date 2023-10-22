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
./src/run/llama.cpp.sh --prompt "prompt" -n 100
```

### candle

The code for candle is largely inspired from huggingface libraries itself, it should automatically download and run the correct model.
Only CPU is supported on M1 atm.
(this is wip, I might break some stuff behind the scenes am working it still, should be stable in a day or two)

```sh
QUANTIZE="q8" PROMPT="prompt" ./src/run/candle.sh
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
| Intel OneAPI/SYCL support   | ✅**    | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅        | ⭐     | ✅       | ✅          | ⭐          |
| BLAS support(CPU)           | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ❌          |
| Training support            | ✅      | 🌟   | ❌        | 🌟     | ❌       | ❌          | ❌          |

⭐ = No Metal Support
🌟 = Partial Support for Training (Finetuning already works, but training from scratch may not work)

## Benchmarking ML Engines

### Consumer Hardware Inference:
#### M1 Pro Mac 16GB Variant
#### LLAMA2-7B
#### mean of runs: 24 (with outliers removed)

| engines     | (cpu) (16bit) tokens/sec | (cpu) (8bit) tokens/sec    | (cpu) (4bit) tokens/sec | (metal) (16bit) tokens/sec | (metal) (8bit) tokens/sec  | (metal/gpu) tokens/sec (4bit) | (metal/gpu) tokens/sec (2bit) |
| ----------- | ------------------------ | -------------------------- | ----------------------- | -------------------------- | -------------------------- | ----------------------------- | ----------------------------- |
| pytorch     |                          |                            |                         |                            |                            |                               |                               |
| burn(torch) |                          | quantization not-supported |                         |                            | quantization not-supported |                               |                               |
| llama.cpp   |                          | 13.2                       |                         |                            | 21.5                       |                               |                               |
| candle      |                          | 9.2                        |                         |                            | metal not supported yet!   |                               |                               |
| CTranslate2 |                          | 12.3                       |                         |                            | metal not supported yet!   |                               |                               |
| tinygrad    |                          | 0.75                       |                         |                            | 7.8                        |                               |                               |


*(data updated: 12th October 2023)

### A100 80GB Inference Bench #1:

#### Model: LLAMA-2-7B
#### CUDA Version: 12.0

| engines                       | version | tokens/sec  |
| --------------------------    | ------- | ----------- |
| pytorch(better-transformer)   | v2.1.0  | 50.8        |
| pytorch(fa2 + bf16)           | v2.1.0  | 47.4        |
| pytorch(bf16)                 | v2.1.0  | 45.1        |
| pytorch(f16)                  | v2.1.0  | 43.2        |
| pytorch(8bit)                 | v2.1.0  | 38.6        |
| pytorch(4bit)                 | v2.1.0  | 29.8        |
| candle(bf16)                  | main    | 32.2        |
| candle(f16)                   | main    | 31.4        |
| candle(f32)                   | main    | 28.1        |
| llama.cpp(4bit)               | master  | 140.1       |
| llama.cpp(8bit)               | master  | 97.8        |
| llama.cpp(f16)                | master  | 77.2        |
| tinygrad(8bit)                | master  | 3.8         |
| tinygrad(f16)[no bf16]        | master  | 21.2        |

fa2 = Flash Attention2
bf16 = bfloat16

No usage of custom kernels or serving strategies/batching.
Stuff like triton, tensor-rt should provide order of magnitude better performance, especially with batching. 
No usage of torchscript to tinkering with default model weights of hf models. (this could provide a 10-20% perf bump)

Candle was benched without flash attention. And quantization as qmatmul on it doesn't support GPU (yet!). 

Note: the perf degradation of quantized models on pytorch for bitsandbytes is expected especially for smaller models like 7B.
Even larger models generally have 15-20% perf degradation, assuming the model could be loaded a single gpu/cluster.

With llama.cpp it is much faster to run quantized models as it isn't as strict about loss of performance(evaluation/quality of output).

*(data updated: 22nd October 2023)

<!--
### A100 80GB Inference Bench #2:

#### Model: LLAMA-2-7B
#### CUDA Version: 12.0

| engines                       | version | tokens/sec  |
| --------------------------    | ------- | ----------- |
| pytorch(f16)                  | v2.1.0  | 43.2        |
| pytorch(bf16)                 | v2.1.0  | 45.1        |
| pytorch(fa2 + bf16)           | v2.1.0  | 47.4        |
| pytorch(better-transformer)   | v2.1.0  | 50.8        |
| pytorch(8bit)                 | v2.1.0  | 38.6        |
| pytorch(4bit)                 | v2.1.0  | 29.8        |

fa2 = Flash Attention2
bf16 = bfloat16

No usage of custom kernels or serving strategies/batching.
No usage of torchscript to tinkering with default model weights of hf models.

Note: the perf degradation of quantized models via bitsandbytes is expected especially for smaller models like 7B.
Even larger models generally have 15-20% perf degradation, assuming the model could be loaded a single gpu/cluster.

*(data updated: 22nd October 2023)
-->

### TODO: Operator-based performance benchmarking

This is a much rougher but arguably more represenatative example of inference engine performance,
using just a single operator across different engines.

