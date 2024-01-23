# ONNX Runtime

[![GitHub Repo](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ggerganov/llama.cpp) &nbsp;


[ONNX (Open Neural Network Exchange) Runtime](https://github.com/microsoft/onnxruntime) is an open-source, cross-platform runtime that enables efficient execution of neural network models trained in various frameworks, promoting interoperability and flexibility in deploying machine learning models. This benchmark implementation uses [HuggingFace Optimum](https://github.com/huggingface/optimum) which supports models running under ONNX Runtime.

### 🚀 Running the ONNX Runtime Benchmark.

You can run the ONNX Runtime  benchmark using the following command:

```bash
./bench_onnxruntime/bench.sh \
  --prompt <value> \            # Enter a prompt string
  --max_tokens <value> \        # Maximum number of tokens to output
  --repetitions <value> \       # Number of repititions to be made for the prompt.
  --log_file <file_path> \      # A .log file underwhich we want to write the results.
  --device <cpu/cuda/metal> \   # The device in which we want to benchmark.
  --models_dir <path_to_models> # The directory in which model weights are present
```

To get started quickly you can simply run:

```bash
./bench_onnxruntime/bench.sh -d cuda
```
This will take all the default values (see in the [bench.sh](/bench_onnxruntime/bench.sh) file) and perform the benchmarks. You can find all the benchmarks results for ONNX Runtime [here](/docs/llama2.md).


### 👀 Some points to note:

1. ONNX Runtime requires HuggingFace Llama2-7B weights. And it converts those weights into ONNX format using this [setup.sh](/bench_onnxruntime/setup.sh) script. So running this benchmark would assume that you already agree to the required terms and conditions and verified to download the weights.
2. ONNX Runtime GPU only support Float16 precision format.
3. Running LLama 2 using ONNX Runtime in CPU/Metal is too memory intensive, so benchmarking is skipped for those.
