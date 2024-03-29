# Candle

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/huggingface/candle) &nbsp;

[Candle](https://github.com/huggingface/candle) is a minimalistic Machine/Deep Learning framework written on Rust by [huggingface](https://github.com/huggingface). It tries to provide a simpler interface to implement models along with GPU support. This is a modified implementation of [Llama2-Candle example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs) to analyse the benchmark performance across different devices and precision.


### 🚀 Running the Candle Benchmark.

For running this benchmark, make sure you have [Rust installed](https://www.rust-lang.org/tools/install). You can run the Candle benchmark using the following command:

```bash
./bench_candle/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_candle/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_candle/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for Candle [here](/docs/llama2.md).


### 👀 Some points to note:

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. Candle does not have support for Metal devices.
3. Candles does support [quantized models](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs). The benchmarks for quantized candles model will be available in the next versions.
