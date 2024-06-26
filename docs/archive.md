# ⚙️ Benchmarking ML Engines

This file contains numbers for different engines and precision. Since a lot of upgrades in models and engines were made. So these
results are now archived. However latest implementation does not have benchmarks for Metal or Mac CPU. So if you want to see that, feel free to check those out here.

## A100 80GB Inference Bench:

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: 11.7
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cuda --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)

| Engine                                     | float32       | float16       | int8          | int4           |
| ------------------------------------------ | ------------- | ------------- | ------------- | -------------- |
| [candle](/bench_candle/)                   | -             | 36.78 ± 2.17  | -             | -              |
| [llama.cpp](/bench_llamacpp/)              | -             | -             | 79.15 ± 1.20  | 100.90 ± 1.46  |
| [ctranslate](/bench_ctranslate/)           | 35.23 ± 4.01  | 55.72 ± 16.66 | 35.73 ± 10.87 | -              |
| [onnx](/bench_onnxruntime/)                | -             | 54.16 ± 3.15  | -             | -              |
| [transformers (pytorch)](/bench_pytorch/)  | 43.79 ± 0.61  | 46.39 ± 0.28  | 6.98 ± 0.05   | 21.72 ± 0.11   |
| [vllm](/bench_vllm/)                       | 90.78 ± 1.60  | 90.54 ± 2.22  | -             | 114.69 ± 11.20 |
| [exllamav2](/bench_exllamav2/)             | -             | -             | 121.63 ± 0.74 | 130.16 ± 0.35  |
| [ctransformers](/bench_ctransformers/)     | -             | -             | 76.75 ± 10.36 | 84.26 ± 5.79   |
| [AutoGPTQ](/bench_autogptq/)               | 42.01 ± 1.03  | 30.24 ± 0.41  | -             | -              |
| [AutoAWQ](/bench_autoawq/)                 | -             | -             | -             | 109.20 ± 3.28  |
| [DeepSpeed](/bench_deepspeed/)             | -             | 81.44 ± 8.13  | -             |                |
| [PyTorch Lightning](/bench_lightning/)     | 24.85 ± 0.07  | 44.56 ± 2.89  | 10.50 ± 0.12  | 24.83 ± 0.05   |
| [Optimum Nvidia](/bench_optimum_nvidia/)   | 110.36 ± 0.52 | 109.09 ± 4.26 | -             | -              |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) | 55.19 ± 1.03  | 85.03 ± 0.62  | 167.66 ± 2.05 | 235.18 ± 3.20  |

*(Data updated: `05th April 2024`)


## M2 MAX 32GB Inference Bench:

### CPU

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: NA
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cpu --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                                 | float32 | float16     | int8         | int4         |
| -------------------------------------- | ------- | ----------- | ------------ | ------------ |
| [candle](/bench_candle/)               | -       | 3.43 ± 0.02 | -            | -            |
| [llama.cpp](/bench_llamacpp/)          | -       | -           | 13.24 ± 0.62 | 21.43 ± 0.47 |
| [ctranslate](/bench_ctranslate/)       | -       | -           | 1.87 ± 0.14  | -            |
| [ctransformers](/bench_ctransformers/) | -       | -           | 13.50 ± 0.48 | 20.57 ± 2.50 |


### GPU (Metal)

**Command:** `./benchmark.sh --repetitions 10 --max_tokens 512 --device metal --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                                 | float32 | float16 | int8         | int4         |
| -------------------------------------- | ------- | ------- | ------------ | ------------ |
| [llama.cpp](/bench_llamacpp/)          | -       | -       | 30.11 ± 0.45 | 44.27 ± 0.12 |
| [ctransformers](/bench_ctransformers/) | -       | -       | 20.75 ± 0.36 | 34.04 ± 2.11 |

*(Data updated: `05th April 2024`)
