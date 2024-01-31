# Burn

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gadersd/llama2-burn) &nbsp;

[Burn](https://github.com/tracel-ai/burn) is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals. For this benchmark implementation, we used a [forked version](https://github.com/premAI-io/llama2-burn) of the [Llama2-Burn project](https://github.com/Gadersd/llama2-burn)


### 🚀 Running the Burn Benchmark.

For running this benchmark, make sure you have [Rust installed](https://www.rust-lang.org/tools/install). You can run the Burn benchmark using the following command:

```bash
./bench_burn/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_burn/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_burn/bench.sh) file) and do the benchmarks. You can find all the benchmarks results for Burn [here](/docs/llama2.md). The HuggingFace Llama 2 weights through a conversion process before benchmarking. See [setup.sh](/bench_burn/setup.sh) to know more.


### 👀 Some points to note:

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. The current implementation of Llama2-Burn only supports Float32 precision for CUDA and CPU.
3. The current implementation of Llama2-Burn does not support Metal.
4. The current implementation of Llama2-Burn does not support INT-4/8 precision quantized models.
