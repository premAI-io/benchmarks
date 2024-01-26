# ⚙️ Benchmarking ML Engines

## A100 80GB Inference Bench:

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: 11.7
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cuda --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                       | float32      | float16        | int8          | int4          |
|------------------------------|--------------|----------------|---------------|---------------|
| burn                         | 13.12 ± 0.85 |      -         |      -        |      -        |
| candle                       |      -       | 36.78 ± 2.17   |      -        |      -        |
| llama.cpp                    |      -       |      -         | 79.15 ± 1.20  | 100.90 ± 1.46 |
| ctranslate                   |      -       | 51.38 ± 16.01  | 36.12 ± 11.93 |      -        |
| tinygrad                     |      -       | 20.32 ± 0.06   |      -        |      -        |
| onnx                         |      -       | 54.16 ± 3.15   |      -        |      -        |
| transformers (pytorch)       | 43.79 ± 0.61 | 46.39 ± 0.28   | 6.98 ± 0.05   | 21.72 ± 0.11  |
| vllm                         | 90.78 ± 1.60 | 90.54 ± 2.22   |      -        |      -        |
| exllamav2                    |      -       |      -         | 116.91 ± 1.73 | 164.28 ± 4.07 |
| ctransformers                |      -       |      -         | 76.75 ± 10.36 | 84.26 ± 5.79  |
| AutoGPTQ                     | 42.01 ± 1.03 | 30.24 ± 0.41  |      -        |      -        |
| AutoAWQ                      |      -       |      -         |      -        | 116.94 ± 13.14|
| DeepSpeed                    |      -       | 81.44 ± 8.13   |      -        |               |
| PyTorch Lightning            | 24.85 ± 0.07 | 44.56 ± 2.89   | 10.50 ± 0.12  | 24.83 ± 0.05  |
| Optimum Nvidia               | 110.36 ± 0.52| 109.09 ± 4.26  |      -        |      -        |
| Nvidia TensorRT-LLM          | 60.39 ± 0.62 | 101.94 ± 8.34  |      -        |      -        |

*(Data updated: `26th January 2024`)


## M2 MAX 32GB Inference Bench:

### CPU

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: NA
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cpu --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                | float32      | float16      | int8         | int4         |
|-----------------------|--------------|--------------|--------------|--------------|
| burn                  | 0.30 ± 0.09  |      -       |      -       |      -       |
| candle                |      -       | 3.43 ± 0.02  |      -       |      -       |
| llama.cpp             |      -       |      -       | 13.24 ± 0.62 | 21.43 ± 0.47 |
| ctranslate            |      -       |      -       | 2.11 ± 0.73  |      -       |
| tinygrad              |      -       | 4.21 ± 0.38  |      -       |      -       |
| onnx                  |      -       |      -       |      -       |      -       |
| ctransformers         |      -       |      -       | 13.50 ± 0.48 | 20.57 ± 2.50 |
| transformers (pytorch)|      -       |      -       |      -       |      -       |
| exllamav2             |      -       |      -       |      -       |      -       |
| vllm                  |      -       |      -       |      -       |      -       |

### GPU (Metal)

**Command:** `./benchmark.sh --repetitions 10 --max_tokens 512 --device metal --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                | float32      | float16       | int8         | int4         |
|-----------------------|--------------|---------------|--------------|--------------|
| burn                  |      -       |      -        |      -       |      -       |
| candle                |      -       |      -        |      -       |      -       |
| llama.cpp             |      -       |      -        | 30.11 ± 0.45 | 44.27 ± 0.12 |
| ctranslate            |      -       |      -        |      -       |      -       |
| tinygrad              |      -       | 29.78 ± 1.18  |      -       |      -       |
| onnx                  |      -       |      -        |      -       |      -       |
| ctransformers         |      -       |      -        | 20.75 ± 0.36 | 34.04 ± 2.11 |
| transformers (pytorch)|      -       |      -        |      -       |      -       |
| exllamav2             |      -       |      -        |      -       |      -       |
| vllm                  |      -       |      -        |      -       |      -       |

*(Data updated: `26th January 2024`)
