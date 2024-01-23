# AutoGPTQ

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AutoGPTQ/AutoGPTQ) &nbsp;
[![ArXiv](https://img.shields.io/badge/arXiv-%230170FE.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2210.17323)


[AutoGPTQ Library](https://github.com/AutoGPTQ/AutoGPTQ) implements the [GPTQ quantization method](https://arxiv.org/abs/2210.17323). This is a point-wise layerwise quantization algorithm, where it tries to approximate the floating point parameters of each weight matrix into a quantized integers such that the error between the output from the actual float weights and the quantized weight is minimum. This quantization process relies heavily on some input samples to evaluate and enhance the quality of the quantization, hence it comes under the one-shot weight quantization method.

### 🚀 Running the AutoGPTQ Benchmark.

You can run the AutoGPTQ benchmark using the following command:

```bash
./bench_autogptq/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which GPTQ model weights are present
```

To get started quickly you can simply run:

```bash
./bench_autogptq/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_autogptq/bench.sh) file) and do the benchmarks. You can find the results for AutoGPTQ [here](/docs/llama2.md)


### 👀 Some points to note:

1. Technically, GPTQ can run on CPUs, but it is super slow. So we did not go for benchmarking that. To understand more, you can reference to this [issue](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/4)
2. The model that was used in this benchmarking process is [LLama2-GPTQ by The Bloke](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ).
3. You might wander that, although the quantization is int-4, then why do we have float32/16 benchmarking. GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16. You can checkout this [reference](https://huggingface.co/blog/gptq-integration#a-gentle-summary-of-the-gptq-paper) to learn more. So here the memory requirement will depend on 4/8 bit quantization, but since the main operations are done on fp16/32 in dequantization, so we included under float-16/32 columns.
