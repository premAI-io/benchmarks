# TinyGrad

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tinygrad/tinygrad) &nbsp;

TinyGrad is a minimalistic deep learning framework, very similar to [PyTorch](https://github.com/pytorch/pytorch). It's simplicity is inspired from the [micrograd](https://github.com/karpathy/micrograd) implementation by [Andrej Karpathy](https://karpathy.ai/). TinyGrad leverages uses different methods like lazy computation and kernel fusion techniques to run different operations. It supports various accelerators out of the box, including CPU, GPU etc. This benchmark implementation uses the [Llama 2 example](https://github.com/tinygrad/tinygrad/blob/master/examples/llama.py) written inside tinygrad/examples.


### 🚀 Running the TinyGrad Benchmark.

You can run the TinyGrad  benchmark using the following command:

```bash
./bench_tinygrad/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_tinygrad/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_tinygrad/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for TinyGrad [here](/docs/llama2.md).


### 👀 Some points to note:

1. The current implementation of TinyGrad only supports Float16 for CUDA, CPU and Metal.
2. This benchmark implementation expects the Raw Llama 2 weights from Meta AI to run LLama2 Model. So it assumes that you already accepted all the [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) before running it.
