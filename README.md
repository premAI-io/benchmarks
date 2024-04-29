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
 <li><a href="#-ml-engines">ML Engines</a></li>
 <li><a href="#-why-benchmarks">Why Benchmarks</a></li>
 <li><a href="#-usage-and-workflow">Usage and workflow</a></li>
 <li><a href="#-contribute">Contribute</a></li>
 </ol>
</details>


## 🥽 Quick glance towards performance benchmarks

Take a first glance at [Mistral 7B v0.1 Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and [Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) Performance Metrics Across Different Precision and Inference Engines. Here is our run specification that generated this performance benchmark reports.

**Environment:**
- Model: Mistral 7B v0.1 Instruct / Llama 2 7B Chat
- CUDA Version: 12.1
- Batch size: 1

**Command:**

```
./benchmark.sh --repetitions 10 --max_tokens 512 --device cuda --model mistral/llama --prompt 'Write an essay about the transformer model architecture'
```

### Mistral 7B v0.1 Instruct

**Performance Metrics:** (unit: Tokens/second)

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

*(Data updated: `30th April 2024`)

### Llama 2 7B Chat

**Performance Metrics:** (unit: Tokens / second)

| Engine                                     | float32       | float16       | int8          | int4          |
| ------------------------------------------ | ------------- | ------------- | ------------- | ------------- |
| [transformers (pytorch)](/bench_pytorch/)  | 36.65 ± 0.61  | 34.20 ± 0.51  | 6.91 ± 0.14   | 17.83 ± 0.40  |
| [AutoAWQ](/bench_autoawq/)                 | -             | -             | -             | 63.59 ± 1.86  |
| [AutoGPTQ](/bench_autogptq/)               | 34.36 ± 0.51  | 36.63 ± 0.61  |               |               |
| [DeepSpeed](/bench_deepspeed/)             |               | 84.60 ± 0.25  |               |               |
| [ctransformers](/bench_ctransformers/)     | -             | -             | 85.50 ± 1.00  | 86.66 ± 1.06  |
| [llama.cpp](/bench_llamacpp/)              | -             | -             | 89.90 ± 2.26  | 97.35 ± 4.71  |
| [ctranslate](/bench_ctranslate/)           | 46.26 ± 1.59  | 79.41 ± 0.37  | 48.20 ± 0.14  | -             |
| [PyTorch Lightning](/bench_lightning/)     | 38.01 ± 0.09  | 48.09 ± 1.12  | 10.68 ± 0.43  | -             |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) | 104.07 ± 1.61 | 191.00 ± 4.60 | 316.77 ± 2.14 | 358.49 ± 2.38 |
| [vllm](/bench_vllm/)                       | 89.40 ± 0.22  | 89.43 ± 0.19  | -             | 115.52 ± 0.49 |
| [exllamav2](/bench_exllamav2/)             | -             | -             | 125.58 ± 1.23 | 159.68 ± 1.85 |
| [onnx](/bench_onnxruntime/)                | 14.28 ± 0.12  | 19.42 ± 0.08  | -             | -             |
| [Optimum Nvidia](/bench_optimum_nvidia/)   | 53.64 ± 0.78  | 53.82 ± 0.11  | -             | -             |


**Performance Metrics:** GPU Memory Consumption (unit: MB)

| Engine                                     | float32  | float16  | int8     | int4     |
| ------------------------------------------ | -------- | -------- | -------- | -------- |
| [transformers (pytorch)](/bench_pytorch/)  | 29114.76 | 14931.72 | 8596.23  | 5643.44  |
| [AutoAWQ](/bench_autoawq/)                 | -        | -        | -        | 7149.19  |
| [AutoGPTQ](/bench_autogptq/)               | 10718.54 | 5706.35  |          |          |
| [DeepSpeed](/bench_deepspeed/)             |          | 83978.35 |          |          |
| [ctransformers](/bench_ctransformers/)     | -        | -        | 9774.83  | 6889.14  |
| [llama.cpp](/bench_llamacpp/)              | -        | -        | 8797.55  | 5783.95  |
| [ctranslate](/bench_ctranslate/)           | 29951.52 | 16282.29 | 9470.74  | -        |
| [PyTorch Lightning](/bench_lightning/)     | 42748.35 | 14736.69 | 8028.16  | -        |
| [Nvidia TensorRT-LLM](/bench_tensorrtllm/) | 79421.24 | 78295.07 | 77642.86 | 77256.98 |
| [vllm](/bench_vllm/)                       | 77928.07 | 77928.07 | -        | 77768.69 |
| [exllamav2](/bench_exllamav2/)             | -        | -        | 16582.18 | 7201.62  |
| [onnx](/bench_onnxruntime/)                | 33072.09 | 19180.55 | -        | -        |
| [Optimum Nvidia](/bench_optimum_nvidia/)   | 79429.63 | 79295.41 | -        | -        |

*(Data updated: `30th April 2024`)

> **Note:** Our previous version of Benchmarks supported benchmarking on Metal and M1/M2 CPUs. We did the benchmarking on similar environments (including mac devices) on Llama 2 7B model. However this version is more focussed on enterprices. But if you are curious, you can check that out [here](/docs/archive.md). Also please not that the numbers might be a bit outdated.

## 🛳 ML Engines

In the current market, there are several ML Engines. Here is a quick glance at all the engines used for the benchmark and a quick summary of their support matrix. You can find the details about the nuances [here](/docs/ml_engines.md).

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

You can check out the nuances related to ⚠️ and 🚧 in details [here](/docs/ml_engines.md)

## 🤔 Why Benchmarks

This can be a common question. What are the benefits you can expect from this repository? So here are some quick pointers to answer those.

1. Oftentimes, we are confused when given several choices on which engines or precision to use for our LLM inference workflow. Because sometimes we have constraints on computing and sometimes we have other requirements. So this repository helps you to get a quick idea of what to use based on your requirements.

2. Sometimes there comes a quality vs speed tradeoff between engines and precisions. So this repository keeps track of those and gives you an idea to understand the tradeoffs so that you can give more importance to your priorities.

3. A fully reproducible and hackable script. The latest benchmarks come with a lot of best practices so that they can be robust enough to run on GPU devices. Also, you can reference and extend the implementations to build your own workflows out of it.

## 🚀 Usage and workflow

Welcome to our benchmarking repository! This organized structure is designed to simplify benchmark management and execution. Each benchmark runs an inference engine that provides some sort of optimizations either through just quantization or device-specific optimizations like custom cuda kernels.

To get started you need to download the models first. This will download the following models:[Llama2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [Mistral-7B v0.1 Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). You can start download by typing this command:

```bash
./download.sh
```

Please make sure that when you are running [Llama2-7B Chat weights](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.

### A Benchmark workflow

When you run a benchmark, the following set of events occurs:

- Automatically setting up the environments and installing the required dependencies.
- Converting the models to some specific format (if required) and saving them.
- Running the benchmarks and storing them inside the logs folder. Each log folder has the following structure:

 - `performance.log`: This will track the model run performances. You can see the `token/sec` and `memory consumption (MB)` here.
 - `quality.md`: This file is an automatically generated readme file, which contains qualitative comparisons of different precisions of some engines. We take 5 prompts and run them for the set of supported precisions of that engine. We then put those results side by side. Our ground truth is the output from huggingface PyTorch model with raw float32 weights.
 - `quality.json` Same as the readme file but more in raw format.

Inside each benchmark folder, you will also see a readme.md file which contains all the information and the qualitative comparison of the engine. For example: [bench_tensorrtllm](/bench_tensorrtllm/README.md).

### Running a Benchmark

Here is how we run benchmarks for an inference engine.

```bash
./bench_<engine-name>/bench.sh \
 --prompt <value> \ # Enter a prompt string
 --max_tokens <value> \  # Maximum number of tokens to output
 --repetitions <value> \  # Number of repetitions to be made for the prompt.
 --device <cpu/cuda/metal> \  # The device in which we want to benchmark.
 --model_name <name-of-the-model> # The name of the model. (options: 'llama' for Llama2 and 'mistral' for Mistral-7B-v0.1)
```

Here is an example. Let's say we want to benchmark Nvidia TensorRT LLM. So here is how the command would look like:

```bash
./bench_tensorrtllm/bench.sh -d cuda -n llama -r 10
```

To know more, here is more detailed info on each command line argument.

```
 -p, --prompt Prompt for benchmarks (default: 'Write an essay about the transformer model architecture')
 -r, --repetitions Number of repetitions for benchmarks (default: 10)
 -m, --max_tokens Maximum number of tokens for benchmarks (default: 512)
 -d, --device Device for benchmarks (possible values: 'metal', 'cuda', and 'CPU', default: 'cuda')
 -n, --model_name The name of the model to benchmark (possible values: 'llama' for using Llama2, 'mistral' for using Mistral 7B v0.1)
 -lf, --log_file Logging file name.
 -h, --help Show this help message
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
├── bench.sh # Benchmark script for setup and execution
├── requirements.txt # Dependencies required for the benchmark
└── ... # Any additional files needed for the benchmark
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
