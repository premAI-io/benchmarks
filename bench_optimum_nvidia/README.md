# Optimum-Nvidia

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/huggingface/optimum-nvidia) &nbsp;

[Optimum-Nvidia](https://github.com/huggingface/optimum-nvidia) is a Large Language Model inference library developed by HuggingFace. It leverages the advanced compilation capabilities of [Nvidia's TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) to enhance performance. The library specifically takes advantage of the Float8 format supported on Nvidia's Ada Lovelace and Hopper architectures. It's worth noting that benchmarking for Float8 is not currently included in this implementation, as it is not widely supported in other inference engines or providers.

### 🚀 Running the Optimum-Nvidia Benchmark.

Before running this benchmark, make sure you have Docker installed. You can run the Optimum-Nvidia  benchmark using the following command:

```bash
./bench_optimum_nvidia/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_optimum_nvidia/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_optimum_nvidia/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for Optimum-Nvidia [here](/docs/llama2.md).


### 👀 Some points to note:

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. Optimum Nvidia uses Docker to convert the models into a specific engine format. You can find the weight conversion logic under [setup.sh](/bench_optimum_nvidia/setup.sh) file.
3. Optimum Nvidia only supports CUDA.
4. Current implementation readily supports Float16/32 and FP-8 precision. We do not benchmark FP-8 precision, because that it can not be compared with other frameworks. And, INT8/4 seems not to be supported.
