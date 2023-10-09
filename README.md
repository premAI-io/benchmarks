# benchmarks
MLOps Engines, Frameworks, and Languages benchmarks over main stream AI Models.

## ML Engines: Feature Table

| Features                    | pytorch | burn | ggml | candle | tinygrad | onnxruntime | CTranslate2 |
| --------------------------- | ------- | ---- | ---- | ------ | -------- | ----------- | ----------- |
| Inference support           | ✅      | ✅   | ✅   | ✅     | ✅       | ✅          | ✅          |
| 16-bit quantization support | ✅      | ✅   | ✅   | ✅     | ✅       | ✅          | ✅          |
| 8-bit quantization support  | ✅      | ✅   | ✅   | ✅     | ✅       | ✅          | ✅          |
| 4-bit quantization support  | ✅      | ✅   | ✅   | ❌     | ❌       | ❌          | ❌          |
| CUDA support                | ✅      | ✅   | ✅   | ✅     | ✅       | ✅          | ✅          |
| ROCM support                | ✅      | ✅   | ✅   | ✅     | ✅       | ❌          | ❌          |
| Intel OneAPI/SYCL support   | ✅**    | ✅   | ✅   | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅   | ✅     | ✅       | ✅          | ✅          |
| BLAS support(CPU)           | ✅      | ✅   | ✅   | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ✅   | ❌   | ❌     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ✅   | ❌   | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅   | ✅     | ❌       | ✅          | ✅          |
| Training support            | ✅      | ✅   | ❌*  | ✅     | ❌       | ✅          | ✅          |

## Benchmarking ML Engines

### Consumer Hardware Inference:
#### M1 Pro Mac(assuming 16GB of memory as a better baseline)
#### LLAMA-7B (16GB memory would require quantization to atleast 8-bit)

| engines      | cpu(tokens/sec) | gpu(tokens/sec) |
| ------------ | --------------- | --------------- |
| pytorch      |                 |                 |
| burn(ndarray)|                 | N/A             |
| burn(wgpu)   | N/A             |                 |
| burn(torch)  |                 |                 |
| ggml(c++)    |                 | N/A             |
| ggml(coreml) |                 |                 |
| candle       |                 |                 |
| tinygrad     |                 |                 |
| onnxruntime  |                 |                 |
| CTranslate2  |                 |                 |

*(data updated: )

### A100 Inference:
#### LLAMA-34B

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

### TODO: Operator-based performance benchmarking

This is a much rougher but arguably more represenatative example of inference engine performance,
using just a single operator across different engines.

