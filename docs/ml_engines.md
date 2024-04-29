# 🔧 ML Engines

### Model Framework Support Matrix

| Engine                                     | Float32 | Float16 | Int8  | Int4  | CUDA  | ROCM  | Mac M1/M2 | Training |
| ------------------------------------------ | :-----: | :-----: | :---: | :---: | :---: | :---: | :-------: | :------: |
| [candle](/bench_candle/)                   |    ⚠️    |    ✅    |   ⚠️   |   ⚠️   |   ✅   |   ❌   |     🚧     |    ❌     |
| [llama.cpp](/bench_llamacpp/)              |    ❌    |    ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     🚧     |    ❌     |
| [ctranslate](/bench_ctranslate/)           |    ✅    |    ✅    |   ✅   |   ❌   |   ✅   |   ❌   |     🚧     |    ❌     |
| [onnx](/bench_onnxruntime/)                |    ✅    |    ✅    |   ❌   |   ❌   |   ✅   |   ⚠️   |     ❌     |    ❌     |
| [transformers (pytorch)](/bench_pytorch/)  |    ✅    |    ✅    |   ✅   |   ✅   |   ✅   |   🚧   |     ✅     |    ✅     |
| [vllm](/bench_vllm/)                       |    ✅    |    ✅    |   ❌   |   ✅   |   ✅   |   🚧   |     ❌     |    ❌     |
| [exllamav2](/bench_exllamav2/)             |    ❌    |    ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     ❌     |    ❌     |
| [ctransformers](/bench_ctransformers/)     |    ❌    |    ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     🚧     |    ❌     |
| [AutoGPTQ](/bench_autogptq/)               |    ✅    |    ✅    |   ⚠️   |   ⚠️   |   ✅   |   ❌   |     ❌     |    ❌     |
| [AutoAWQ](/bench_autoawq/)                 |    ❌    |    ❌    |   ❌   |   ✅   |   ✅   |   ❌   |     ❌     |    ❌     |
| [DeepSpeed-MII](/bench_deepspeed/)         |    ❌    |    ✅    |   ❌   |   ❌   |   ✅   |   ❌   |     ❌     |    ⚠️     |
| [PyTorch Lightning](/bench_lightning/)     |    ✅    |    ✅    |   ✅   |   ✅   |   ✅   |   ⚠️   |     ⚠️     |    ✅     |
| [Optimum Nvidia](/bench_optimum_nvidia/)   |    ✅    |    ✅    |   ❌   |   ❌   |   ✅   |   ❌   |     ❌     |    ❌     |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) |    ✅    |    ✅    |   ✅   |   ✅   |   ✅   |   ❌   |     ❌     |    ❌     |


### Legend:
- ✅ Supported
- ❌ Not Supported
- ⚠️ There is a catch related to this
- 🚧 It is supported but not implemented in this current version


### Some pointers to note:
The names are by the name of engines. Except when the name is `Generic` then it means that the nuance applies to all the engines.


| Name              | Type | Description                                                                                                                                                                                                                            |
| ----------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| candle            | ⚠️    | Metal backend is supported but it gives terrible performance even in small models like Phi2. For AMD ROCM there is no support as per this [issue](https://github.com/huggingface/candle/issues/346).                                   |
| candle            | 🚧    | Latest performance for Candle is not implemented. If you want to see the numbers, please check out [archive.md](/docs/archive.md) which contains the benchmark numbers for [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b). |
| ctranslate2       | ⚠️    | ROCM is not supported; however, works are in progress to have this feature on CTranslate2. No support for Mac M1/M2.                                                                                                                   |
| onnxruntime       | ⚠️    | ONNXRuntime in general supports ROCM, but specific to LLMs and ONNXRuntime with HuggingFace Optimum only supports CUDAExecution provider right now. For CPU, it is available but super slow.                                           |
| pytorch lightning | ⚠️    | ROCM is supported but not tested for PyTorch Lightning. See this [issue](https://github.com/Lightning-AI/litgpt/issues/1220).                                                                                                          |
| pytorch lightning | ⚠️    | Metal is supported in PyTorch Lightning, but for Llama 2 7B Chat or Mistral 7B, it is super slow.                                                                                                                                      |
| AutoGPTQ          | ⚠️    | AutoGPTQ is a weight-only quantization algorithm. Activation still remains in either float32 or float16. We used a 4-bit weight quantized model for our benchmarks experiment.                                                         |
| Generic           | 🚧    | For all the engines which support metal, please check out [archive.md](/docs/archive.md) which contains the benchmark numbers for [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b).                                          |
| Deepspeed         | ⚠️    | [DeepSpeed](https://github.com/microsoft/DeepSpeed) supports training; however, for inference, we have used [DeepSpeed MII](https://github.com/microsoft/DeepSpeed-MII).                                                               |
