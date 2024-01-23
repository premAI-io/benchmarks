# TensorRT-LLM

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NVIDIA/TensorRT-LLM) &nbsp;

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is a Python library that facilitates the creation and optimization of Large Language Models (LLMs) for efficient inference on NVIDIA GPUs. TensorRT-LLM supports various quantization modes, including INT4 and INT8 weights, along with FP16 activations, allowing users to maximize performance and minimize memory usage. It also provides pre-defined models that can be easily customized and extended to meet specific requirements, and it integrates with the [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) for production deployment.

### 🚀 Running the TensorRT-LLM Benchmark.

Running TensorRT-LLM requires Docker. So make sure you have installed Docker. You can run the TensorRT-LLM  benchmark using the following command:

```bash
./bench_tensorrt_llm/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_tensorrt_llm/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_tensorrt_llm/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for TensorRT-LLM [here](/docs/llama2.md).


### 👀 Some points to note:

1. Running this benchmark requires HuggingFace Llama2-7B weights. So running this benchmark would assume that you already agree to the required terms and conditions and verified to download the weights.
2. TensorRT LLM only works with CUDA. So it does not support Metal/CPU.
3. Although TensorRT supports INT4/8 quantization, but it uses AutoGPTQ under the hood. Since we wanted to keep our implementation independent, so we skipped it.
