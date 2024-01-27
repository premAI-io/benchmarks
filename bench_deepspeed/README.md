# DeepSpeed

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/microsoft/DeepSpeed-MII) &nbsp;

[DeepSpeed](https://github.com/microsoft/DeepSpeed) by Microsoft is a library that helps us to do scalable training and inference across multiple GPUs and nodes. In this implementation we are using a library called [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII), written on top of DeepSpeed. It uses four key technologies to speed up the inference, viz Blocked KV Caching, Continuous Batching, Dynamic SplitFuse and High Performance CUDA Kernels. To learn more about DeepSpeed-MII, check out there detailed [blogpost](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen).


### 🚀 Running the DeepSpeed Benchmark.

You can run the DeepSpeed benchmark using the following command:

```bash
./bench_deepspeed/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_deepspeed/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_deepspeed/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for DeepSpeed [here](/docs/llama2.md).


### 👀 Some points to note:

1. Running this benchmark requires [HuggingFace Llama2-7B weights](https://huggingface.co/meta-llama/Llama-2-7b). So running this benchmark would assume that you already agreed to the required [terms and conditions](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and got verified to download the weights.
2. DeepSpeed-MII is designed to run on CUDA. So by default it does not support for Metal or CPU devices.
3. Since we run the benchmark for only LLama2-7B model. And for this implementation, DeepSpeed-MII [only supports](https://github.com/microsoft/DeepSpeed/blob/b81bed69a8db3c1e3263c27f48dcecf12b354931/deepspeed/inference/v2/model_implementations/llama_v2/model.py#L83) Float16 precision.
4. Current implementation of DeepSpeed-MII [does not support](https://github.com/microsoft/DeepSpeed-MII/issues/255) Quantized models. So INT4/8 benchmarking is not available.
