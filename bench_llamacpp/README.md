# Llama.cpp

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ggerganov/llama.cpp) &nbsp;

[Llama.cpp](https://github.com/ggerganov/llama.cpp) is a port of Llama 2 model written in C++. Right now it supports [different popular](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#description) LLMs. Llama.cpp supports LLMs under various quantizations. For this benchmark implementation, we are only running it under 4 and 8 bit quantized versions. Please note: This benchmark implementation uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), which is the python binding for LLama.cpp library.

### 🚀 Running the Llama.cpp Benchmark.

You can run the Llama.cpp benchmark using the following command:

```bash
./bench_llamacpp/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_llamacpp/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_llamacpp/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for Llama.cpp [here](/docs/llama2.md).


### 👀 Some points to note:

1. This implementation uses [4-bit](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) and [8-bit](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q8_0.gguf) quantized versions of the Llama2 7B model.
