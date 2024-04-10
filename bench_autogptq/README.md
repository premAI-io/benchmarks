# AutoGPTQ

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AutoGPTQ/AutoGPTQ) &nbsp;
[![ArXiv](https://img.shields.io/badge/arXiv-%230170FE.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2210.17323)


[AutoGPTQ Library](https://github.com/AutoGPTQ/AutoGPTQ) implements the [GPTQ quantization method](https://arxiv.org/abs/2210.17323). This is a layerwise post-training quantization algorithm, where it tries to approximate the floating point parameters of each weight matrix into a quantized integers such that the error between the output from the actual float weights and the quantized weight is minimum. The layer-wise weight quantization process uses the [Optimal Brain Quantization framework](https://arxiv.org/abs/2208.11580) and relies heavily on some input samples to evaluate and enhance the quality of the quantization, hence it comes under the one-shot weight quantization method.

> GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16. [source](https://huggingface.co/blog/gptq-integration)

### 🚀 Running the AutoGPTQ Benchmark.

We can run the AutoAWQ benchmark for two models: [Llama2-chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) and [Mistral-7B v0.1-instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GPTQ) Here is how we run benchmark for AutoAWQ.

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
./bench_autogptq/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_autogptq/bench.sh) file) and do the benchmarks. You can find the results for AutoGPTQ [here](/docs/llama2.md)


### 👀 Some points to note:

1. AutoGPTQ adopts a mised int-4/float16 quantization scheme. It can also do int-4/float32 scheme. Where weights will be in INT-4 and activation will be in float16/32. So we have kept benchmarks numbers in float16 and float32, although quantization is done for INT-4.
2. Technically, GPTQ can run on CPUs, but it is super slow. So we did not go for benchmarking that. To understand more, you can reference to this [issue](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/4)
3. The model that was used in this benchmarking process is [LLama2-GPTQ by The Bloke](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ).
4. INT-8 is not available right now because AutoGPTQ [integrates](https://huggingface.co/blog/gptq-integration#room-for-improvement) with the most performant W4A16 kernel (weights as int4, activations as fp16). Although quantizing to INT-8 is possible but is likely to be super slow, see [this](https://github.com/AutoGPTQ/AutoGPTQ/issues/452) and [this](https://github.com/AutoGPTQ/AutoGPTQ/issues/499) issue.
5. AutoGPTQ [does not support](https://github.com/AutoGPTQ/AutoGPTQ/issues/366) Metal till now.
6. AutoGPTQ [also supports ExllamaV2](https://huggingface.co/blog/gptq-integration#autogptq-library--the-one-stop-library-for-efficiently-leveraging-gptq-for-llms) and other quantization methods, but we did not used it, so that we can benchmark each methods and framework independently without any mutual intersections.
7. Tokens/sec for INT4/FP-32 is greater than INT4/FP-16, which is not an expected behaviour, probably due to some [downcasting](https://github.com/huggingface/transformers/issues/28647) overhead.
