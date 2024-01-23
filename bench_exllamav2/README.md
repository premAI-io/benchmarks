# ExLlamaV2

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/turboderp/exllamav2) &nbsp;

[ExLlamaV2](https://github.com/turboderp/exllamav2) uses custom Kernels to speed up LLM inference under different quantizations. ExLlamaV2 supports a new "EXL2" format. EXL2 is based on the same optimization method as GPTQ and supports 2, 3, 4, 5, 6 and 8-bit quantization. For this benchmark implementation, we use 4-bit and 8-bit quantization version of Llama2.


### 🚀 Running the ExLlamaV2 Benchmark.

You can run the ExLlamaV2 benchmark using the following command:

```bash
./bench_exllamav2/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_exllamav2/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_exllamav2/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for ExLlamaV2 [here](/docs/llama2.md).


### 👀 Some points to note:

1. ExLlamaV2 supports quantized LLMs. So Float32/16 is not supported here.
2. ExLlamaV2 currently [does not have support](https://github.com/turboderp/exllamav2/issues/184) for Mac/Metal.
3. Although it supports CPU, but it is too slow to offload and run. So we did not include in our benchmarks.
