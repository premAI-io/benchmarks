# benchmarks
MLOps Engines, Frameworks, and Languages benchmarks over main stream AI Models.

## Tool

The benchmarking tool comprises three main scripts:
- `benchmark.sh` for running the end-to-end benchmarking
- `download.sh` which is internally used by the benchmark script to download the needed model files based on a configuration
- `setup.sh` script for setup of dependencies and needed formats conversion

### benchmark

This script runs benchmarks for a transformer model using both Rust and Python implementations. It provides options to customize the benchmarks, such as the prompt, repetitions, maximum tokens, device, and NVIDIA flag.

```bash
./benchmark.sh [OPTIONS]
```
where `OPTIONS`:
- `-p, --prompt`: Prompt for benchmarks (default: 'Explain what is a transformer')
- `-r, --repetitions`: Number of repetitions for benchmarks (default: 2)
- `-m, --max_tokens`: Maximum number of tokens for benchmarks (default: 100)
- `-d, --device`: Device for benchmarks (possible values: 'gpu' or 'cpu', default: 'cpu')
- `--nvidia`: Use NVIDIA for benchmarks (default: false)

### download

Downloads files from a list of URLs specified in a JSON file. The JSON file should contain an array of objects, each with a 'url', 'file', and 'folder' property. The script checks if the file already exists before downloading it.

```bash
./download.sh --models <json_file> --cache <cache_file> --force-download
```
Options
- `--models`: JSON file specifying the models to download (default: models.json)
- `--cache`: Cache file to keep track of downloaded files (default: cache.log)
- `--force-download`: Force download of all files, removing existing files and cache

### setup
1. Creates a python virtual environment `venv` and installs project requirements.
3. Converts and stores models in different formats.

```bash
./setup.sh
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

### A100 80GB Inference Bench:

Model: LLAMA-2-7B

CUDA Version: 11.7

Command: `./benchmark.sh --repetitions 10 --max_tokens 100 --device gpu --nvidia --prompt 'Explain what is a transformer'`

| Engine      | float32      | float16      | int8         | int4         |
|-------------|--------------|--------------|--------------|--------------|
| burn        | 13.28 ± 0.79 |      -       |      -       |      -       |
| candle      |      -       | 26.30 ± 0.29 |      -       |      -       |
| llama.cpp   |      -       |      -       | 67.64 ± 22.57| 106.21 ± 2.21|
| ctranslate  |      -       | 58.54 ± 13.24| 34.22 ± 6.29 |      -       |
| tinygrad    |      -       | 20.13 ± 1.35 |      -       |      -       |

*(data updated: 15th November 2023)


### M2 MAX 32GB Inference Bench:

#### CPU

Model: LLAMA-2-7B

CUDA Version: NA

Command: `./benchmark.sh --repetitions 10 --max_tokens 100 --device cpu --prompt 'Explain what is a transformer'`

| Engine      | float32       | float16       | int8         | int4         |
|-------------|--------------|--------------|--------------|--------------|
| burn        | 0.30 ± 0.09  |      -       |      -       |      -       |
| candle      |      -       | 3.43 ± 0.02  |      -       |      -       |
| llama.cpp   |      -       |      -       | 14.41 ± 1.59 | 20.96 ± 1.94 |
| ctranslate  |      -       |      -       | 2.11 ± 0.73  |      -       |
| tinygrad    |      -       | 4.21 ± 0.38  |      -       |      -       |

#### GPU (Metal)

Command: `./benchmark.sh --repetitions 10 --max_tokens 100 --device gpu --prompt 'Explain what is a transformer'`

| Engine      | float32       | float16       | int8         | int4         |
|-------------|--------------|--------------|--------------|--------------|
| burn        |      -       |      -       |      -       |      -       |
| candle      |      -       |      -       |      -       |      -       |
| llama.cpp   |      -       |      -       | 31.24 ± 7.82 | 46.75 ± 9.55 |
| ctranslate  |      -       |      -       |      -       |      -       |
| tinygrad    |      -       | 29.78 ± 1.18 |      -       |      -       |

*(data updated: 15th November 2023)