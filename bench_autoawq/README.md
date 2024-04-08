# AutoAWQ

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/casper-hansen/AutoAWQ) &nbsp;
[![ArXiv](https://img.shields.io/badge/arXiv-%230170FE.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.00978)


[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) is a package that is a polished implemementation of the original work [llm-awq](https://github.com/mit-han-lab/llm-awq) from MIT. AWQ or Activation Aware Quantization is a quantization method which supports 4-bit quantization. It massively increases the inference throughput and decreases the memory requirement of the model at the same time. (For example, according to this [reference](https://huggingface.co/TheBloke/Llama-2-70B-Chat-AWQ), Llama2 70B requires 2 x 80 GB but with AutoAWQ it can be run on 1 x 48 GB GPU). You can learn more about AWQ on the research paper and the github implementations.

### 🚀 Running the AutoAWQ Benchmark.

We can run the AutoAWQ benchmark for two models: [Llama2](https://huggingface.co/meta-llama/Llama-2-7b) and [Mistral-7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) Here is how we run benchmark for AutoAWQ.

```bash
./bench_autoawq/bench.sh \
  --prompt <value> \               # Enter a prompt string
  --max_tokens <value> \           # Maximum number of tokens to output
  --repetitions <value> \          # Number of repititions to be made for the prompt.
  --device <cpu/cuda/metal> \      # The device in which we want to benchmark.
  --model_name <name-of-the-model> # The name of the model. (options: 'llama' for Llama2 and 'mistral' for Mistral-7B-v0.1)
```

To get started quickly you can simply run:

```bash
./bench_autoawq/bench.sh -d cuda
```

This will take all the default values (see in the [bench.sh](/bench_autoawq/bench.sh) file) and do the benchmarks for Llama 2 model. You can find all the benchmarks results for AutoAWQ [here](/docs/llama2.md). You can also do a minimal benchmarking for Mistral 7


### 👀 Some points to note:

1. AutoAWQ is not supported devices other than GPU (only supports when CUDA is available).
2. We are independently benchmarking AutoAWQ (i.e. the actual AWQ quantization method here). We are not benchmarking with combinations like: AutoAWQ + VLLM or AutoAWQ + TensorRT.
3. For doing this benchmark, the default model that was choosen was: [Llama2-AutoAWQ by The Bloke](https://huggingface.co/TheBloke/Llama-2-7B-AWQ)
4. AutoAWQ does not support INT8 quantization properly yet. See [this issue](https://github.com/casper-hansen/AutoAWQ/issues/45).
