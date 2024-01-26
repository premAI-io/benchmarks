# CTranslate2

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OpenNMT/CTranslate2) &nbsp;

[CTranslate2](https://github.com/OpenNMT/CTranslate2) tries to provide efficient and faster inference for Transformer Models by implementing custom runtime and performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc. Written in C++, but it also provides a nice and simple Python API to get started with it easily.

### 🚀 Running the ctranslate2 Benchmark.

Running this code requires Docker. So make sure you have [Docker installed](https://docs.docker.com/engine/install/). You can run the ctranslate2 benchmark using the following command:

```bash
./bench_ctranslate/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_ctranslate/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_ctranslate/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for ctranslate2 [here](/docs/llama2.md).


### 👀 Some points to note:

1. CTranslate2 does not support INT-4 precision. See this [issue](https://github.com/OpenNMT/CTranslate2/issues/1104)
2. This implementation uses Llama2 weights from HuggingFace. So running this benchmark will assume that all the [terms and conditions](https://huggingface.co/meta-llama/Llama-2-7b) are met from user's side.
3. CTranslate2 does not support FP16/32 for CPU and it does not support any precision for Metal device.
