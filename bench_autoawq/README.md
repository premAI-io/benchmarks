# AutoAWQ

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/casper-hansen/AutoAWQ) &nbsp;
[![ArXiv](https://img.shields.io/badge/arXiv-%230170FE.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.00978)


AutoAWQ is a package that is a polished implemementation of the original work [llm-awq](https://github.com/mit-han-lab/llm-awq) from MIT. AWQ or Activation Aware Quantization is a quantization method which supports 4-bit quantization. It massively increases the inference throughput and decreases the memory requirement of the model at the same time. (For example a Llama2 70B requires 2 x 80 GB but with AutoAWQ it can be run on 1 x 48 GB GPU). You can learn more about AWQ on the research paper and the github implementations.

### 🚀 Running the AutoAWQ Benchmark.

You can run the AutoAWQ benchmark using the following command:

```bash
./bench_{bench_name}/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which AWQ model weights are present
```

To get started quickly you can simply run:

```bash
./bench_autoawq/bench.sh -d cuda
```

This will take all the default values (see in the `./bench_autoawq/bench.sh` file) and do the benchmarks.

### 📊 Results of AutoAWQ

Here are the results of AutoAWQ. The metric used to benchmark is `tokens/sec` and the results are gathered by running the following command:

```
./bench_autoawq/bench.sh --repetitions 10 --max_tokens 100 --device cuda --prompt 'Explain what is a transformer'
```
#### On GPU

| Engine                       | float32      | float16       | int8          | int4          |
|------------------------------|--------------|---------------|---------------|---------------|
| AutoAWQ                      |      -       |      -        |      -        | 116.94 ± 13.14|


### 👀 Some points to note:

1. AutoAWQ is not supported devices other than GPU (only supports when CUDA is available).
2. We are independently benchmarking AutoAWQ (i.e. the actual AWQ quantization method here). We are not benchmarking with combinations like: AutoAWQ + VLLM or AutoAWQ + TensorRT.
3. For doing this benchmark, the default model that was choosen was: [Llama2-AutoAWQ by The Bloke](https://huggingface.co/TheBloke/Llama-2-7B-AWQ)
