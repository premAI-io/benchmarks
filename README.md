# benchmarks
MLOps Engines, Frameworks, and Languages benchmarks over main stream AI Models.

## ML Engines: Feature Table

| Features                    | pytorch | burn | llama.cpp | candle | tinygrad | onnxruntime | CTranslate2 |
| --------------------------- | ------- | ---- | --------- | ------ | -------- | ----------- | ----------- |
| Inference support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 16-bit quantization support | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 8-bit quantization support  | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 4-bit quantization support  | ✅      | ✅   | ✅        | ✅     | ❌       | ❌          | ❌          |
| CUDA support                | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| ROCM support                | ✅      | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Intel OneAPI/SYCL support   | ✅**    | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| BLAS support(CPU)           | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ✅   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ✅   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| backwards-op support        | ✅      | ✅   | ❌*       | ✅     | ❌       | ✅          | ✅          |

## Benchmarking ML Engines

### Consumer Hardware Inference:
#### M1 Pro Mac 16GB Variant
#### LLAMA2-7B q8 (quantized to 8-bit)
#### mean of runs: 24 (with outliers removed)

| engines      | (cpu)tokens/sec                 | (metal/gpu) tokens/sec     |
| ------------ | ----------                      | ----------------------     |
| pytorch      | -                               | -                          |
| burn(torch)  | quantization not-supported      | quantization not-supported |
| llama.cpp    | 13.2                            | 21.5                       |
| candle       | 9.2                             | metal not supported yet!   |
| onnxruntime  |                                 |                            |
| CTranslate2  |                                 | metal not supported yet!   |
| tinygrad     | quantization not-supported      | quantization not-supported |

*(data updated: 12th October 2023)

<!-- TODO(swarnim)
### A100 Inference:
#### LLAMA-B

| engines                    | performance |
| -------------------------- | ----------- |
| pytorch                    |             |
| pytorch(tensor-rt)         |             |
| pytorch(LLM.int8)          |             |
| burn(wgpu)                 |             |
| burn(torch)                |             |
| ggml(cuda)                 |             |
| candle                     |             |
| tinygrad                   |             |
| onnxruntime                |             |
| CTranslate2                |             |

*(data updated: )
-->

### TODO: Operator-based performance benchmarking

This is a much rougher but arguably more represenatative example of inference engine performance,
using just a single operator across different engines.

