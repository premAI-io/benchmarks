# vLLM

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vllm-project/vllm) &nbsp;

[vLLM](https://github.com/vllm-project/vllm) is a high-performance library designed for efficient language model (LLM) inference and serving. With a focus on speed, it incorporates state-of-the-art features such as PagedAttention for memory management, continuous batching, and optimized CUDA kernels. It supports various models like LLama, Falcon etc. It is very much flexible and also supports different decoding methods, tensor, distributed inference etc.


### 🚀 Running the vLLM Benchmark.

You can run the vLLM  benchmark using the following command:

```bash
./bench_vllm/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_vllm/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_vllm/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for vLLM [here](/docs/llama2.md).

**Note:** vLLM seems to very dependent on cuda versions. So it is recommended to either have cuda version `>= 12.0`. We also do have support for cuda version `< 12.0`.

### 👀 Some points to note:

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. vLLM Does not support CPU (check [this](https://github.com/vllm-project/vllm/issues/176) issue) and Metal devices (check [this](https://github.com/vllm-project/vllm/issues/1441) issue).
3. Current implementation of vLLM uses [AWQ quantizatized model](https://github.com/vllm-project/vllm?tab=readme-ov-file#about) to benchmark Llama2 on INT-4 precision under vLLM.
4. vLLM does not support INT8 currently. See this [issue](https://github.com/vllm-project/vllm/pull/1508).
