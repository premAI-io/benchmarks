# ⚙️ Benchmarking ML Engines

## A100 80GB Inference Bench:

**Environment:**
- Model: Llama 2 7B Chat
- CUDA Version: 12.1
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cuda --model llama --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)

| Engine                                      | float32      | float16        | int8          | int4          |
|---------------------------------------------|--------------|----------------|---------------|---------------|
| [transformers (pytorch)](/bench_pytorch/)   | 37.37 ± 0.45 | 34.42 ± 0.45   | 7.07 ± 0.08   | 18.88 ± 0.08  |

**Performance Metrics:** GPU Memory Consumption (unit: MB)

| Engine                                      | float32  | float16  | int8     | int4     |
|---------------------------------------------|----------|----------|----------|----------|
| [transformers (pytorch)](/bench_pytorch/)   | 29114.76 | 41324.38 | 21384.66 | 12830.38 |


*(Data updated: `15th April 2024`)


## M2 MAX 32GB Inference Bench:

### CPU

**Environment:**
- Model: LLAMA-2-7B
- CUDA Version: NA
- Command: `./benchmark.sh --repetitions 10 --max_tokens 512 --device cpu --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                                 | float32      | float16      | int8         | int4         |
|----------------------------------------|--------------|--------------|--------------|--------------|
| [candle](/bench_candle/)               |      -       | 3.43 ± 0.02  |      -       |      -       |
| [llama.cpp](/bench_llamacpp/)          |      -       |      -       | 13.24 ± 0.62 | 21.43 ± 0.47 |
| [ctranslate](/bench_ctranslate/)       |      -       |      -       | 1.87 ± 0.14  |      -       |
| [ctransformers](/bench_ctransformers/) |      -       |      -       | 13.50 ± 0.48 | 20.57 ± 2.50 |


### GPU (Metal)

**Command:** `./benchmark.sh --repetitions 10 --max_tokens 512 --device metal --prompt 'Write an essay about the transformer model architecture'`

**Performance Metrics:** (unit: Tokens / second)
| Engine                                  | float32      | float16       | int8         | int4         |
|-----------------------------------------|--------------|---------------|--------------|--------------|
| [llama.cpp](/bench_llamacpp/)           |      -       |      -        | 30.11 ± 0.45 | 44.27 ± 0.12 |
| [ctransformers](/bench_ctransformers/)  |      -       |      -        | 20.75 ± 0.36 | 34.04 ± 2.11 |

*(Data updated: `15th April 2024`)
