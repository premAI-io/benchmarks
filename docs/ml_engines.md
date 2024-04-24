# 🔧 ML Engines

### Model Framework Support Matrix

| Engine                                     | Float32 | Float16 | Float8 | Int8  | Int4  | CUDA  | ROCM  | Mac M1/M2 | Training |
| ------------------------------------------ | :-----: | :-----: | :----: | :---: | :---: | :---: | :---: | :-------: | :------: |
| [candle](/bench_candle/)                   |    ⚠️    |    ✅    |   ❌    |   ⚠️   |   ⚠️   |   ✅   |   ❌   |     ✅     |    ❌     |
| [llama.cpp](/bench_llamacpp/)              |    ❌    |    ❌    |   ❌    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ❌     |
| [ctranslate](/bench_ctranslate/)           |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [onnx](/bench_onnxruntime/)                |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [transformers (pytorch)](/bench_pytorch/)  |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [vllm](/bench_vllm/)                       |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [exllamav2](/bench_exllamav2/)             |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [ctransformers](/bench_ctransformers/)     |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [AutoGPTQ](/bench_autogptq/)               |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [AutoAWQ](/bench_autoawq/)                 |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [DeepSpeed](/bench_deepspeed/)             |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [PyTorch Lightning](/bench_lightning/)     |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [Optimum Nvidia](/bench_optimum_nvidia/)   |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) |    ✅    |    ✅    |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |     ✅     |    ✅     |


### Legend:
- ✅ Supported
- ❌ Not Supported
- ⚠️ Supported but not implemented


### Some pointers to note:

1. For candle, Metal backend is supported but it gives terrible performance [even in small models like Phi2](https://github.com/huggingface/candle/issues/1568). For AMD ROCM there is no support as per this [issue](https://github.com/huggingface/candle/issues/346).
