# PyTorch

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/huggingface/transformers) &nbsp;

The implementation of benchmarking [PyTorch](https://github.com/pytorch/pytorch) uses the [Transformers Library by Huggingface](https://github.com/huggingface/transformers) under the hood. The reason being, Transformers provides an easy to use interface for the Llama-2-7B model in PyTorch backend.


### 🚀 Running the PyTorch Benchmark.

You can run the PyTorch  benchmark using the following command:

```bash
./bench_pytorch/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_pytorch/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_pytorch/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for PyTorch [here](/docs/llama2.md).


### 👀 Some points to note:

1. Running this benchmark requires HuggingFace Llama2-7B weights. So running this benchmark would assume that you already agree to the required terms and conditions and verified to download the weights.
2. Running Llama 2 with PyTorch requires lot of memory to run inside CPU/Metal devices. So those are skipped.
