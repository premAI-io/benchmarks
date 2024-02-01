<div align="center">

  <h1 align="center">🕹️ Benchmarks</h1>
    <p align="center">A fully reproducible Performance Comparison of MLOps Engines, Frameworks, and Languages on Mainstream AI Models.</p>
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
    <li><a href="#-roadmap">Roadmap</a></li>
    <li><a href="#-introducing-prem-grant-program">Introducing Prem Grant Program</a></li>
  </ol>
</details>

</br>

## 📊 Quick glance towards performance metrics for Llama-2-7B

Take a first glance of Llama-2-7B Model Performance Metrics Across Different Precision and Inference Engines. Metric used: `tokens/sec`


| Engine                                      | float32      | float16        | int8          | int4          |
|---------------------------------------------|--------------|----------------|---------------|---------------|
| [burn](/bench_burn/)                        | 10.04 ± 0.64 |      -         |      -        |      -        |
| [candle](/bench_candle/)                    |      -       | 36.78 ± 2.17   |      -        |      -        |
| [llama.cpp](/bench_llamacpp/)               |      -       |      -         | 79.15 ± 1.20  | 100.90 ± 1.46 |
| [ctranslate](/bench_ctranslate/)            | 35.23 ± 4.01 | 55.72 ± 16.66  | 35.73 ± 10.87 |      -        |
| [tinygrad](/bench_tinygrad/)                |      -       | 20.32 ± 0.06   |      -        |      -        |
| [onnx](/bench_onnxruntime/)                 |      -       | 54.16 ± 3.15   |      -        |      -        |
| [transformers (pytorch)](/bench_pytorch/)   | 43.79 ± 0.61 | 46.39 ± 0.28   | 6.98 ± 0.05   | 21.72 ± 0.11  |
| [vllm](/bench_vllm/)                        | 90.78 ± 1.60 | 90.54 ± 2.22   |      -        |      -        |
| [exllamav2](/bench_exllamav2/)              |      -       |      -         | 121.63 ± 0.74 | 130.16 ± 0.35 |
| [ctransformers](/bench_ctransformers/)      |      -       |      -         | 76.75 ± 10.36 | 84.26 ± 5.79  |
| [AutoGPTQ](/bench_autogptq/)                | 42.01 ± 1.03 | 30.24 ± 0.41   |      -        |      -        |
| [AutoAWQ](/bench_autoawq/)                  |      -       |      -         |      -        | 109.20 ± 3.28 |
| [DeepSpeed](/bench_deepspeed/)              |      -       | 81.44 ± 8.13   |      -        |               |
| [PyTorch Lightning](/bench_lightning/)      | 24.85 ± 0.07 | 44.56 ± 2.89   | 10.50 ± 0.12  | 24.83 ± 0.05  |
| [Optimum Nvidia](/bench_optimum_nvidia/)    | 110.36 ± 0.52| 109.09 ± 4.26  |      -        |      -        |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/)  | 55.19 ± 1.03 | 85.03 ± 0.62   | 167.66 ± 2.05 | 235.18 ± 3.20 |

*(Data updated: `31th January 2024`)



- The above benchmarking is done on A100-80GB GPU. You can find more details for other devices like CPU/Metal under [docs](docs/llama2.md) folder.

- Also if you want to see more detailed information about each of the benchmark, you can find those details the respective benchmark folders.

- If you want to compare side by side which inference engines supports which precision and device, you can check out the [ml_engines.md](/docs/ml_engines.md) file. Please note that this file is incomplete and a better comparision of engines will be added in the later versions.

Benchmarks can also be considered as a repository of hackable scripts, that contains the code and all the knowledge base to run the popular inference engines.

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

Replace `<value>` with the specific values for your benchmark, and `<file_path>` and `<path_to_models>` with the appropriate file and directory paths.

### Run All Benchmarks Collectively

For a comprehensive execution of all benchmarks, use the overarching `benchmark.sh` script:

```bash
./bench.sh --prompt <value> --max_tokens <value> --repetitions <value> --log_file <file_path> --device <cpu/cuda/metal> --models_dir <path_to_models>
```

Again, customize the parameters according to your preferences, ensuring that <file_path> and <path_to_models> point to the correct locations.

Feel free to adjust the parameters as needed for your specific benchmarking requirements. Please note that, running all the benchmarks collectively can requires lot of storage (around 500 GB). Please make sure that you have enough storage to run all of them at once.

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


## 🗾 Roadmap

In our upcoming versions, we will be adding support for the following:

1. Add more metrics on memory consumption. This includes how much RAM/GPU memory is consumed when we run the benchmarks.
2. Add support for more models. Upcoming versions will support popular LLMs like [Mamba](https://huggingface.co/state-spaces/mamba-2.8b), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [Phi2](https://huggingface.co/microsoft/phi-2) etc.
3. Add ways to understand and articulate on change of generation quality with the change of frameworks and precision. We will try to add ways to understand how the generation quality of an LLM changes when we change the precision of the models or use a different inference engine framework.
4. Add support for batching. Since batching is very important while deploying LLMs. So coming versions will benchmark LLMs on batched inputs.

If you feel like there is something more to add, feel free to open an issue or a PR. We would be super happy to take contributions from the community.


## 🏆 Introducing Prem Grant Program

![Alt Text](https://blog.premai.io/content/images/size/w1200/2024/01/IMG.jpg)

🌟 Exciting news, AI enthusiasts! Prem is thrilled to launch the Prem Grant Program, exclusively designed for forward-thinking AI startups ready to reshape the future. With this program, you get six months of free access to OpenAI, Anthropic, Cohere, Llama2, Mistral (or any other open-source model) APIs, opening doors to endless AI possibilities at zero cost. Enjoy free fine-tuning, seamless model deployment, and expert ML support. This is more than a grant; it's an invite to lead the AI revolution. Don't miss out – apply now and let's build the future of AI together with Prem! 🌟

Read more about the Prem Startup grant program [here](https://blog.premai.io/announcing-our-startup-grants-program/). You can directly apply to the program from [here](https://docs.google.com/forms/d/e/1FAIpQLSdv1WuZ5aC7raefnupMTla5z_-7p1XD9D28HK0nZ7JkKkQwRQ/viewform).
