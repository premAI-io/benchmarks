# ⚙️ Benchmarking ML Engines

## A100 80GB Inference Bench:

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: 11.7
- Command: `./benchmark.sh --repetitions 10 --max_tokens 100 --device cuda --prompt 'Explain what is a transformer'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                       | float32      | float16       | int8          | int4          |
|------------------------------|--------------|---------------|---------------|---------------|
| burn                         | 13.12 ± 0.85 |      -        |      -        |      -        |
| candle                       |      -       | 36.78 ± 2.17  |      -        |      -        |
| llama.cpp                    |      -       |      -        | 38.48 ± 1.02  | 41.99 ± 2.70  |
| ctranslate                   |      -       | 51.38 ± 16.01 | 36.12 ± 11.93 |      -        |
| tinygrad                     |      -       | 20.32 ± 0.06  |      -        |      -        |
| onnx                         |      -       | 54.16 ± 3.15  |      -        |      -        |
| transformers (pytorch)       | 46.44 ± 46.44| 42.56 ± 42.56 |      -        |      -        |
| vllm                         | 90.78 ± 1.60 | 90.54 ± 2.22  |      -        |      -        |
| exllamav2                    |      -       |      -        | 116.91 ± 1.73 | 164.28 ± 4.07 |
| ctransformers                |      -       |      -        | 80.67 ± 3.89  | 84.42 ± 4.57  |

*(Data updated: `29th December 2023`)


## M2 MAX 32GB Inference Bench:

### CPU

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: NA
- Command: `./benchmark.sh --repetitions 10 --max_tokens 100 --device cpu --prompt 'Explain what is a transformer'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                | float32      | float16      | int8         | int4         |
|-----------------------|--------------|--------------|--------------|--------------|
| burn                  | 0.30 ± 0.09  |      -       |      -       |      -       |
| candle                |      -       | 3.43 ± 0.02  |      -       |      -       |
| llama.cpp             |      -       |      -       | 14.41 ± 1.59 | 20.96 ± 1.94 |
| ctranslate            |      -       |      -       | 2.11 ± 0.73  |      -       |
| tinygrad              |      -       | 4.21 ± 0.38  |      -       |      -       |
| onnx                  |      -       |      -       |      -       |      -       |
| ctransformers         |      -       |      -       | 13.79 ± 0.50 | 22.93 ± 0.86 |
| transformers (pytorch)|      -       |      -       |      -       |      -       |
| exllamav2             |      -       |      -       |      -       |      -       |
| vllm                  |      -       |      -       |      -       |      -       |

### GPU (Metal)

**Command:** `./benchmark.sh --repetitions 10 --max_tokens 100 --device metal --prompt 'Explain what is a transformer'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                | float32      | float16       | int8         | int4         |
|-----------------------|--------------|---------------|--------------|--------------|
| burn                  |      -       |      -        |      -       |      -       |
| candle                |      -       |      -        |      -       |      -       |
| llama.cpp             |      -       |      -        | 31.24 ± 7.82 | 46.75 ± 9.55 |
| ctranslate            |      -       |      -        |      -       |      -       |
| tinygrad              |      -       | 29.78 ± 1.18  |      -       |      -       |
| onnx                  |      -       |      -        |      -       |      -       |
| ctransformers         |      -       |      -        | 21.24 ± 0.81 | 34.08 ± 4.78 |
| transformers (pytorch)|      -       |      -        |      -       |      -       |
| exllamav2             |      -       |      -        |      -       |      -       |
| vllm                  |      -       |      -        |      -       |      -       |

*(Data updated: `29th December 2023`)

*Note: Although benchmarking for pytorch transformers on mac is possible. But, we are not doing it, since it is very much time taking, and so makes it very less significant.
*Note: ExllamaV2 does not run in only CPUs or Apple GPU. It requires CUDA.
*Note: CPU/Metal is not supported right now. Support for CPU is on [developement](https://github.com/vllm-project/vllm/pull/1028). No developement for metal so far.
