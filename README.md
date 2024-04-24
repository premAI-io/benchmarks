<div align="center">

  <h1 align="center">🕹️ Benchmarks</h1>
    <p align="center">A fully reproducible Performance Comparison of MLOps Engines, Frameworks, and Languages on Mainstream AI Models</p>
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
    <li><a href="#-quick-glance">Quick glance towards performance metrics for Mistral</a></li>
    <li><a href="#-getting-started">Getting started</a></li>
    <li><a href="#-usage">Usage</a></li>
    <li><a href="#-contribute">Contribute</a></li>
    <li><a href="#-roadmap">Roadmap</a></li>
    <li><a href="#-introducing-prem-grant-program">Introducing Prem Grant Program</a></li>
  </ol>
</details>


## 🥽 Quick glance towards performance metrics for Mistral

Take a first glance of Mistral 7B v0.1 Instruct Model Performance Metrics Across Different Precision and Inference Engines.

**Performance Metrics:** (unit: Tokens / second)

| Engine                                     | float32       | float16       | int8          | int4          |
| ------------------------------------------ | ------------- | ------------- | ------------- | ------------- |
| [transformers (pytorch)](/bench_pytorch/)  | 39.61 ± 0.65  | 37.05 ± 0.49  | 5.08 ± 0.01   | 19.58 ± 0.38  |
| [AutoAWQ](/bench_autoawq/)                 | -             | -             | -             | 63.12 ± 2.19  |
| [AutoGPTQ](/bench_autogptq/)               | 39.11 ± 0.42  | 42.94 ± 0.80  |               |               |
| [DeepSpeed](/bench_deepspeed/)             |               | 79.88 ± 0.32  |               |               |
| [ctransformers](/bench_ctransformers/)     | -             | -             | 86.14 ± 1.40  | 87.22 ± 1.54  |
| [llama.cpp](/bench_llamacpp/)              | -             | -             | 88.27 ± 0.72  | 95.33 ± 5.54  |
| [ctranslate](/bench_ctranslate/)           | 43.17 ± 2.97  | 68.03 ± 0.27  | 45.14 ± 0.24  | -             |
| [PyTorch Lightning](/bench_lightning/)     | 32.79 ± 2.74  | 43.01 ± 2.90  | 7.75 ± 0.12   | -             |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) | 117.04 ± 2.16 | 206.59 ± 6.93 | 390.49 ± 4.86 | 427.40 ± 4.84 |
| [vllm](/bench_vllm/)                       | 84.91 ± 0.27  | 84.89 ± 0.28  | -             | 106.03 ± 0.53 |
| [exllamav2](/bench_exllamav2/)             | -             | -             | 114.81 ± 1.47 | 126.29 ± 3.05 |
| [onnx](/bench_onnxruntime/)                | 15.75 ± 0.15  | 22.39 ± 0.14  | -             | -             |
| [Optimum Nvidia](/bench_optimum_nvidia/)   | 50.77 ± 0.85  | 50.91 ± 0.19  | -             | -             |

**Performance Metrics:** GPU Memory Consumption (unit: MB)

| Engine                                     | float32  | float16  | int8     | int4     |
| ------------------------------------------ | -------- | -------- | -------- | -------- |
| [transformers (pytorch)](/bench_pytorch/)  | 31071.4  | 15976.1  | 10963.91 | 5681.18  |
| [AutoGPTQ](/bench_autogptq/)               | 13400.80 | 6633.29  |          |          |
| [AutoAWQ](/bench_autoawq/)                 | -        | -        | -        | 6572.47  |
| [DeepSpeed](/bench_deepspeed/)             |          | 80104.34 |          |          |
| [ctransformers](/bench_ctransformers/)     | -        | -        | 10255.07 | 6966.74  |
| [llama.cpp](/bench_llamacpp/)              | -        | -        | 9141.49  | 5880.41  |
| [ctranslate](/bench_ctranslate/)           | 32602.32 | 17523.8  | 10074.72 | -        |
| [PyTorch Lightning](/bench_lightning/)     | 48783.95 | 18738.05 | 10680.32 | -        |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) | 79536.59 | 78341.21 | 77689.0  | 77311.51 |
| [vllm](/bench_vllm/)                       | 73568.09 | 73790.39 | -        | 74016.88 |
| [exllamav2](/bench_exllamav2/)             | -        | -        | 21483.23 | 9460.25  |
| [onnx](/bench_onnxruntime/)                | 33629.93 | 19537.07 | -        | -        |
| [Optimum Nvidia](/bench_optimum_nvidia/)   | 79563.85 | 79496.74 | -        | -        |


*(Data updated: `<LAST_UPDATE>`)

## 📊 More Benchmarks

- The latest version of Benchmarks supports benchmarking on GPU only. You can also checkout similar benchmark result for Llama 2 7B Chat [here](/docs/llama2.md).

- Our previous version of Benchmarks supported benchmarking on Metal and M1/M2 CPUs. So if you are curious, you can check that out [here](/docs/archive.md)

## 🛳 ML Engines

In the current market, there are several ML Engines. Here is a quick glance of all the engines used for the benchmark and a quick summary of their support matrix. You can fine the details about the nuances [here](/docs/ml_engines.md).

| Engine                                     | Float32 | Float16 | Float8 | Int8  | Int4  | CUDA  | ROCM  | Mac M1/M2 | Training |
| ------------------------------------------ | :-----: | :-----: | :----: | :---: | :---: | :---: | :---: | :-------: | :------: |
| [candle](/bench_candle/)                   |    ⚠️    |    ✅    |   ❌    |   ⚠️   |   ⚠️   |   ✅   |   ❌   |     🚧     |    ❌     |
| [llama.cpp](/bench_llamacpp/)              |    ❌    |    ❌    |   ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     🚧     |    ❌     |
| [ctranslate](/bench_ctranslate/)           |    ✅    |    ✅    |   ❌    |   ✅   |   ❌   |   ✅   |   ❌   |     🚧     |    ❌     |
| [onnx](/bench_onnxruntime/)                |    ✅    |    ✅    |   ❌    |   ❌   |   ❌   |   ✅   |   ⚠️   |     ❌     |    ❌     |
| [transformers (pytorch)](/bench_pytorch/)  |    ✅    |    ✅    |   ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     ✅     |    ✅     |
| [vllm](/bench_vllm/)                       |    ✅    |    ✅    |   ❌    |   ❌   |   ✅   |   ✅   |   🚧   |     ❌     |    ❌     |
| [exllamav2](/bench_exllamav2/)             |    ❌    |    ❌    |   ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     ❌     |    ❌     |
| [ctransformers](/bench_ctransformers/)     |    ❌    |    ❌    |   ❌    |   ✅   |   ✅   |   ✅   |   🚧   |     🚧     |    ❌     |
| [AutoGPTQ](/bench_autogptq/)               |    ✅    |    ✅    |   ❌    |   ⚠️   |   ⚠️   |   ✅   |   ❌   |     ❌     |    ❌     |
| [AutoAWQ](/bench_autoawq/)                 |    ❌    |    ❌    |   ❌    |   ❌   |   ✅   |   ✅   |   ❌   |     ❌     |    ❌     |
| [DeepSpeed-MII](/bench_deepspeed/)         |    ❌    |    ✅    |   ❌    |   ❌   |   ❌   |   ✅   |   ❌   |     ❌     |    ⚠️     |
| [PyTorch Lightning](/bench_lightning/)     |    ✅    |    ✅    |   ❌    |   ✅   |   ✅   |   ✅   |   ⚠️   |     ⚠️     |    ✅     |
| [Optimum Nvidia](/bench_optimum_nvidia/)   |    ✅    |    ✅    |   🚧    |   ❌   |   ❌   |   ✅   |   ❌   |     ❌     |    ❌     |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) |    ✅    |    ✅    |   🚧    |   ✅   |   ✅   |   ✅   |   ❌   |     ❌     |    ❌     |


### Legend:
- ✅ Supported
- ❌ Not Supported
- ⚠️ There is a catch related to this
- 🚧 It is supported but not implemented in this current version


## 🤔 Why Benchmarks

This can be a common question. What are the benifits you can expect from this repository? So here are some quick pointers to answer those.

1. Often times, we are confused when given several choices on which engines or precision to use for our LLM inference workflow. Because sometimes we have constraints on compute and sometimes we have other requirements. So this repository helps you to get a quick idea on what to use based on your requirements.

2. Some times there comes quality vs speed tradeoff between engines and precisions. So this repository keeps a track of those and gives you an idea to understand the tradeoffs, so that you can give more importance to your priorities.

3. A fully reproducible and hackable script. Latest benchmarks comes with lot of best practices so that it can be robust enough to run on GPU devices. Also you can reference and extend the implementations to build your own workflows out of it.

## 🚀Usage and workflow

Welcome to our benchmarking repository! This organized structure is designed to simplify benchmark management and execution. Each benchmark is runs an inference engine which provides some sort of optimizations either through just quantization or device specific optimizations like custom cuda kernels.

To get started you need to download the models first. This will download the following models:[Llama2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [Mistral-7B v0.1 Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). You can start download by typing this command:

```bash
./download.sh
```

Please make sure that when you are running [Llama2-7B Chat weights](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.

### A Benchmark workflow

When you run a benchmark, following set of events occurs:

- Automatically setting up the environments and installing the required dependencies.
- Converting the models to some specific format (if required) and saving them.
- Running the benchmarks and store it inside logs folder. Each log folder has this following structure:

  - `performance.log` : This will track the model run performances. You can see the `token/sec` and `memory consumption (MB)` here.
  - `quality.md` : This file is an automatically generated readme file, which contains a qualitative comparisions of different precisions of some engines. We take 5 prompts and run them for the set of supported precisions of that engine. We then put those results side by side. Our ground truth is the output from huggingface PyTorch model with raw float32 weights.
  - `quality.json` Same as the readme file but more in raw format.

Inside each benchmark folder, you will also see a readme.md file which contains all the information and the qualitative comparision about the engine. For example: [bench_tensorrtllm](/bench_tensorrtllm/README.md).

### Running a Benchmark

Here is how we run benchmark for an inference engine.

```bash
./bench_<engine-name>/bench.sh \
  --prompt <value> \               # Enter a prompt string
  --max_tokens <value> \           # Maximum number of tokens to output
  --repetitions <value> \          # Number of repititions to be made for the prompt.
  --device <cpu/cuda/metal> \      # The device in which we want to benchmark.
  --model_name <name-of-the-model> # The name of the model. (options: 'llama' for Llama2 and 'mistral' for Mistral-7B-v0.1)
```

Here is an example. Let's say we want to benchmark Nvidia TensorRT LLM. So here is how the command would look like:

```bash
./bench_tensorrtllm/bench.sh -d cuda -n llama -r 10
```

To know more, here is a more detailed info on each command line argument.

```
  -p, --prompt        Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')
  -r, --repetitions   Number of repetitions for benchmarks (default: 10)
  -m, --max_tokens    Maximum number of tokens for benchmarks (default: 512)
  -d, --device        Device for benchmarks (possible values: 'metal', 'cuda', and 'cpu', default: 'cuda')
  -n, --model_name    The name of the model to benchmark (possible values: 'llama' for using Llama2, 'mistral' for using Mistral 7B v0.1)
  -lf, --log_file     Logging file name.
  -h, --help          Show this help message
```

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
