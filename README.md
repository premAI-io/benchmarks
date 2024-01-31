<div align="center">

  <h1 align="center">🕹️ Benchmarks</h1>
    <p align="center">Performance Comparison of MLOps Engines, Frameworks, and Languages on Mainstream AI Models.</p>
</div>

[![GitHub contributors](https://img.shields.io/github/contributors/premAI-io/benchmarks.svg)](https://github.com/premAI-io/benchmarks/graphs/contributors)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/premAI-io/benchmarks.svg)](https://github.com/premAI-io/benchmarks/commits/master)
[![GitHub last commit](https://img.shields.io/github/last-commit/premAI-io/benchmarks.svg)](https://github.com/premAI-io/benchmarks/commits/master)
[![GitHub top language](https://img.shields.io/github/languages/top/premAI-io/benchmarks.svg)](https://github.com/premAI-io/benchmarks)
[![GitHub issues](https://img.shields.io/github/issues/premAI-io/benchmarks.svg)](https://github.com/premAI-io/benchmarks/issues)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-quick-glance">Quick glance towards performance metrics for Llama-2-7B</a></li>
    <li><a href="#-getting-started">Getting started</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-contribute">Contribute</a></li>
  </ol>
</details>

</br>

## 📊 Quick glance towards performance metrics for Llama-2-7B

Take a first glance of Llama-2-7B Model Performance Metrics Across Different Precision and Inference Engines


| Engine                       | float32      | float16        | int8          | int4          |
|------------------------------|--------------|----------------|---------------|---------------|
| burn                         | 10.04 ± 0.64 |      -         |      -        |      -        |
| candle                       |      -       | 36.78 ± 2.17   |      -        |      -        |
| llama.cpp                    |      -       |      -         | 79.15 ± 1.20  | 100.90 ± 1.46 |
| ctranslate                   | 35.23 ± 4.01 | 55.72 ± 16.66  | 35.73 ± 10.87 |      -        |
| tinygrad                     |      -       | 20.32 ± 0.06   |      -        |      -        |
| onnx                         |      -       | 54.16 ± 3.15   |      -        |      -        |
| transformers (pytorch)       | 43.79 ± 0.61 | 46.39 ± 0.28   | 6.98 ± 0.05   | 21.72 ± 0.11  |
| vllm                         | 90.78 ± 1.60 | 90.54 ± 2.22   |      -        |      -        |
| exllamav2                    |      -       |      -         | 121.63 ± 0.74 | 130.16 ± 0.35 |
| ctransformers                |      -       |      -         | 76.75 ± 10.36 | 84.26 ± 5.79  |
| AutoGPTQ                     | 42.01 ± 1.03 | 30.24 ± 0.41   |      -        |      -        |
| AutoAWQ                      |      -       |      -         |      -        | 109.20 ± 3.28 |
| DeepSpeed                    |      -       | 81.44 ± 8.13   |      -        |               |
| PyTorch Lightning            | 24.85 ± 0.07 | 44.56 ± 2.89   | 10.50 ± 0.12  | 24.83 ± 0.05  |
| Optimum Nvidia               | 110.36 ± 0.52| 109.09 ± 4.26  |      -        |      -        |
| Nvidia TensorRT-LLM          | 55.19 ± 1.03 | 85.03 ± 0.62   | 167.66 ± 2.05 | 235.18 ± 3.20 |

*(Data updated: `31th January 2024`)



- The above benchmarking is done on A100-80GB GPU. You can find more details for other devices like CPU/Metal under [docs](docs/llama2.md) folder.

- Also if you want to see more detailed information about each of the benchmark, you can find those details the respective benchmark folders.


## 🚀 Getting Started

Welcome to our benchmarking repository! This organized structure is designed to simplify benchmark management and execution. Here's a quick guide to get you started:

- **Benchmark Organization:** Each benchmark is uniquely identified as `bench_name` and resides in its dedicated folder, named `bench_{bench_name}`.

- **Benchmark Script (`bench.sh`):** Within these benchmark folders, you'll find a common script named `bench.sh`. This script takes care of everything from setup and environment configuration to actual execution.

### Benchmark Script Parameters

The `bench.sh` script supports the following key parameters, allowing for customization and flexibility:

- `prompt`: Benchmark-specific prompt.
- `max_tokens`: Maximum tokens for the benchmark.
- `repetitions`: Number of benchmark repetitions.
- `log_file`: File for storing benchmark logs.
- `device`: Specify the device for benchmark execution (CPU, CUDA, Metal).
- `models_dir`: Directory containing necessary model files.

### Streamlined Execution

The overarching [`benchmark.sh`](./benchmark.sh) script further simplifies the benchmark execution process:

- **File Download:** It automatically downloads essential files required for benchmarking.
- **Folder Iteration:** The script iterates through all benchmark folders in the repository, streamlining the process for multiple benchmarks.

This approach empowers users to effortlessly execute benchmarks based on their preferences. To run a specific benchmark, navigate to the corresponding benchmark folder (e.g., `bench_{bench_name}`) and execute the `bench.sh` script with the required parameters.

## 📄 Usage

To utilize the benchmarking capabilities of this repository, follow these usage examples:

### Run a Specific Benchmark

Navigate to the benchmark folder and execute the `bench.sh` script with the desired parameters:

```bash
./bench_{bench_name}/bench.sh --prompt <value> --max_tokens <value> --repetitions <value> --log_file <file_path> --device <cpu/cuda/metal> --models_dir <path_to_models>
```

Replace <value> with the specific values for your benchmark, and <file_path> and <path_to_models> with the appropriate file and directory paths.

### Run All Benchmarks Collectively

For a comprehensive execution of all benchmarks, use the overarching `benchmark.sh` script:

```bash
./bench.sh --prompt <value> --max_tokens <value> --repetitions <value> --log_file <file_path> --device <cpu/cuda/metal> --models_dir <path_to_models>
```

Again, customize the parameters according to your preferences, ensuring that <file_path> and <path_to_models> point to the correct locations.

Feel free to adjust the parameters as needed for your specific benchmarking requirements.

## 🤝 Contribute

We welcome contributions to enhance and expand our benchmarking repository. If you'd like to contribute a new benchmark, follow these steps:

### Creating a New Benchmark

**1. Create a New Folder**

Start by creating a new folder for your benchmark. Name it `bench_{new_bench_name}` for consistency.

```bash
mkdir bench_{new_bench_name}
```

**2. Folder Structure**

Inside the new benchmark folder, include the following structure

```
bench_{new_bench_name}
├── bench.sh           # Benchmark script for setup and execution
├── requirements.txt   # Dependencies required for the benchmark
└── ...                # Any additional files needed for the benchmark
```

**3. Benchmark Script (`bench.sh`):**

The `bench.sh` script should handle setup, environment configuration, and the actual execution of the benchmark. Ensure it supports the parameters mentioned in the [Benchmark Script Parameters](#benchmark-script-parameters) section.

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality and consistency.

**1. Install Pre-commit:** Ensure you have `pre-commit` installed

```bash
pip install pre-commit
```

**2. Install Hooks:** Run the following command to install the pre-commit hooks

```bash
pre-commit install
```

The existing pre-commit configuration will be used for automatic checks before each commit, ensuring code quality and adherence to defined standards.
