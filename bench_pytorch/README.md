# PyTorch

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/huggingface/transformers) &nbsp;

The implementation of benchmarking [PyTorch](https://github.com/pytorch/pytorch) uses the [Transformers Library by Huggingface](https://github.com/huggingface/transformers) under the hood. The reason being, Transformers provides an easy to use interface for Llama-2-7B model in PyTorch backend.


### 🚀 Running the PyTorch Benchmark.

You can run the PyTorch benchmark using the following command:

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

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. Running Llama 2 with PyTorch on CPU/Metal devices are super slow and goes out of memory with increase in context size. So those are skipped. 
3. The PyTorch Benchmark uses [BitsAndBytes library](https://github.com/TimDettmers/bitsandbytes/tree/main) to run INT8/4 quantization and running the Llama-2 models.  
4. Running LLama 2 on INT-8 precision is [not supported](https://github.com/TimDettmers/bitsandbytes/blob/1e642109dc7bb668c1e80f53ef80803d4ff11701/bitsandbytes/autograd/_functions.py#L225) for CPU/Metal, since it requires custom CUDA Kernels. 
5. Running LLama 2 on INT-4 precision is [not supported](https://github.com/TimDettmers/bitsandbytes/blob/1e642109dc7bb668c1e80f53ef80803d4ff11701/bitsandbytes/autograd/_functions.py#L565) for CPU/Metal devices, since it requires custom CUDA Kernels. 
