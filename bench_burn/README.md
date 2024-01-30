# Burn

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gadersd/llama2-burn) &nbsp;

We use [Llama2-Burn project](https://github.com/Gadersd/llama2-burn), which provides a port of the Llama2 model to [Burn](https://github.com/tracel-ai/burn). Burn is the DeepLearning Framework for Rust, which provides similar concepts and interfaces like PyTorch.


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
This will take all the default values (see in the [bench.sh](/bench_burn/bench.sh) file) and do the benchmarks. You can find all the benchmarks results for Burn [here](/docs/llama2.md).


### 👀 Some points to note:

1. For CUDA and Metal, Burn runs for only Float32 precision.
2. You need to download weights of LLama-2 7B from HuggingFace. This repo already does it. However it assumes that you already have accepted the [terms and condition](https://huggingface.co/meta-llama/Llama-2-7b-hf) before running or downloading the model and runnning this benchmark.
