# benchmarks
MLOps Engines, Frameworks, and Languages benchmarks over main stream AI Models.

## Usage for running scripts to get performance info

the runscripts are now going to be outdated I am still keeping them around as they are useful for reference.

### tinygrad

We provide custom script to run inference, and require providing path to MODEL_DIR to ensure correct handling of model setup.
The script should report average time in tokens/sec.

```sh
source ./src/setup/tinygrad.sh
# set QUANTIZE=0 to not use int8 quantization
# mac m1 and beyond defaults to "METAL" backend
# use CPU=1 to use CPU backend
QUANTIZE=1 MODEL_DIR="<path/to/llama2-7b-model>" ./src/run/tinygrad.sh "prompt"
```

### burn

### llama.cpp

## ML Engines: Feature Table

| Features                    | pytorch | burn | llama.cpp | candle | tinygrad | onnxruntime | CTranslate2 |
| --------------------------- | ------- | ---- | --------- | ------ | -------- | ----------- | ----------- |
| Inference support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 16-bit quantization support | ✅      | x   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 8-bit quantization support  | ✅      | x   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 4-bit quantization support  | ✅      | x   | ✅        | ✅     | ❌       | ❌          | ❌          |
| CUDA support                | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| ROCM support                | ✅      | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Intel OneAPI/SYCL support   | ✅**    | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| BLAS support(CPU)           | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ✅   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ✅   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| training support            | ✅      | ✅   | ❌*       | ✅     | ❌       | ✅          | ✅          |

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

