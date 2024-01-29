# Lightning

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Lightning-AI/lit-gpt) &nbsp;

[Lit-GPT](https://github.com/Lightning-AI/lit-gpt) is a hackable implementation of [different Open Source LLMs](https://github.com/Lightning-AI/lit-gpt?tab=readme-ov-file#-lit-gpt-1). Lit-GPT is written using the [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) framework. Lightning Fabric is a fast and lightweight way to scale PyTorch models. It comes with features that enables to do distributed training and inference with ease. Lightning Fabric is based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).


### 🚀 Running the Lightning Benchmark.

You can run the Lightning benchmark using the following command:

```bash
./bench_lightning/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_lightning/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_lightning/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for Lightning [here](/docs/llama2.md).


### 👀 Some points to note:

1. This implementation runs Llama-2-7B models. Lit-GPT model implementation requires converting HuggingFace models to lit-gpt formats. The model conversion can be found in the [setup.sh](/bench_lightning/setup.sh) file.
2. Since, running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So we would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
3. When running it on Metal devices, it runs out of memory. Therefore we were not able to do inference on Metal.
4. For CPU, it is just too much slow (> 1 minute). Hence we skipped the benchmarking for CPUs.
