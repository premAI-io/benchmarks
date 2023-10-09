# benchmarking and understanding performance of ML frameworks and libraries.

## Motivation

Notes on early benching passes, working on proper benching plans to get concrete 1:1 data between different projects.
Currently working on requirement and understanding different project constraints and general performance characteristics.

## Goals

- Figure out constraints of current frameworks "for our needs".
- Note the possible optimization directions.
- Understand requirements for optimal cross platform ML performance and support.

## Early Investigation

The overall investigation assumes PyTorch as the performance base and all relevant understand should be built in context of that,
the specific benchmark might not be directly comparable to each other but it should provide a rough picture of the state of 
open source ML framework performance.

This generally is to port, add and support new things into burn and other platforms.

### Burn vs PyTorch

#### Performance for Torch backend:

Noticeably better or comparable to using torch backend directly due to optimization such as In-place Operations, and stronger typing which allows static resolution of const known parameters(like dims and dtype). [^1]

#### Other Backends

- Candle: experimental, not properly supported.
- WebGPU: beta, but not well implemented or mature, requires some work. Do note lots of optimization & fixes already have WIP PRs.
- nd-array: CPU only, BLAS support, but no cuBLAS or similar partial GPU support planned yet!

#### Overall performance is the same as PyTorch generally, with torch backend.

#### References
[1] https://burn-rs.github.io/blog/burn-rusty-approach-to-tensor-handling

### vLLM

vLLM is an early research project, which highly lacks interms of capabilities.

It provides means of improving performance by using their own technique called
PagedAttention, [read more here](https://arxiv.org/abs/2309.06180).
Notes on the paper below, in the papers section.

#### Platform: x86 & GPU

No CPU support.
No GPU platform supported, except Nvidia.

CUDA performance is hard to judge with no competing implementations.
But it provides significant CUDA .

#### Platform: M1 & Apple

No CPU support.
No ANE or Accelerate support.

#### Pros
We can use it on top of any backend conceptually for it only implements token tensor indexing
for more efficient memory usage. And it aligns with our model as an API service philosophy.

#### Cons
- Extremely limited platform support.
- Would require a lot of effort to port to non-CUDA systems.
- It's largest gains come from, efficient memory sharing, which requires parallel workloads specifically.

### FasterTransformer

This is a model library for encoder and decoder components for specific models optimized for CUDA platforms,
and it supports running on PyTorch, Tensorflow or Triton backend.

This directly works with PyTorch backend, hence any library support torch backend should also be indirectly
compatible(like burn).

To simplify it provides a custom implementation for torch, tensorflow, and triton ops for specific models with custom
CUDA kernels for faster performance. Overall the general perf improvement is around 2-5x, on high perf CUDA platforms.
Torch cuda kernels aren't nearly as well optimized it seems.

#### Platform: x86 & GPU
No CPU or non CUDA support.

#### Platform: M1 & Apple
No support.

#### Pros
We can consider borrowing CUDA kernel optimizations, especially for high performance platforms.

#### Cons
Extremely limited support.

### tensor-rt
Developed by Nvidia.

#### Platform: x86 & GPU
No CPU or non CUDA support.

#### Platform: M1 & Apple
NA.

#### Pros
Fairly straight forward library with very wide model format and library support.
Possible to use as a tensor backend for other projects as well.

#### Cons
Extremely limited platform support.

### Tinygrad

Fairly simple framework/library for AI inference.
Training seems to largely be a non-priority.

Focuses on providing layered primitive abstraction model to allow for platforms to implement
support in an easier fashion.

It abstracts with llops(low level ops) that need to be implemented, mlops(middle ops, that are derivatives of llops but can be customized for greater performance), and hlops(completely optional, useful for exposing additional backend features).
[read more](https://github.com/tinygrad/tinygrad/blob/e1f2c2cc190246c90a1b1713c4631cafe84ca629/docs/adding_new_accelerators.md)

#### Platform Support
CPU Performance in general is quite lack lustre, it's not highly optimized and being a python framework and geohot's disinterest in it doesn't seem like a priority by a long shot.

Most GPU platforms are directly supported using WebGPU for general fallback, CUDA for Nvidia, ROCM for AMD, SYCL/OneAPI for Intel and Metal & also CoreML for Mac.

Generally a bit slower than torch on most platforms other than M1, where it's quite decent.

#### Pros
Simple and cross platform support.

#### Cons
Not so performant for most targets.
And especially lacks scalability for popular large high compute platforms(aka Nvidia, it's slower on Nvidia than alternatives).

TODO(add references)

### GGML

Extremely portable and simple library written purely in C & C++.
Works with weights directly(requires hand-rolled model definition in C/C++).
Also works with GGUF format to simplify loading and unloading models.

Supports trivial training. But lacks features and support for a large number of models,
and requires good understanding of code and low level programs to use efficiently.

Also as it avoids repeated runtime allocation by building contexts/arenas for tensors,
it can be hard to figure out preallocation for larger models. Can require a lot of tinkering,
or setup. For adding support for new models for inference.

#### Platform: x86 & GPU
CPUs are supported by default both through BLAS and hand-rolled implementation. Fairly decent performance over all.

CUDA, ROCM(via HIP) & OpenVINO support is already present. But the kernels aren't the most optimized.
Generally comparable performance to standard torch(torch without any custom optimization frameworks).

#### Platform: M1 & Apple

Supports M1 (ARM-NEON), aka CPU.

Supports MPS(Accelerate) aka GPU.

Also supports CoreML, but not ANE(Apple Neural Engine), ANE seems to be hard to use.
This can be upto 3x faster compared to CPU for inference.

#### Pros
- Support for non standard formats like int4 and int5, and portable to other quantization formats as well.
- Also fairly simple and easy to use library, much less complexity when compared to PyTorch or Tensorflow.

#### Cons
- Very sparse documentation and tutorials for beginners to programming or people who are used to Python land.
- Have to rewrite the model network for every new model and figuring out the optimal setup for allocations can be hard.

### Tensorflow Lite

It is a good solution for executing ML models on mobile and embedded devices. Also support web platforms.
In terms of performance and flexibility it's generally behind pytorch.

### onnx

IR Format for ML models, it stores both weights and network, allowing for single bundle models to be executed in runtime agnostic fashion.
Several engines above support this format.

This simplifies deployment for inference as, if the runtime supports onnx using onnx model format make it trivial to port models to the runtime.

#### wonnx (webgpu + onnx + rust)
- Supports a decent subset of onnx IR.
- Works on all platforms, linux, mac, windows. And all architectures.
- Performance-wise it's behind libraries with platform specific GEMM ops.

### DirectML

Open Source windows only platform for ML ops on top of DirectX 12 Compute Dispatch, relatively decent performance.
Supports ONNX format.

#### Platform: x86 & GPU
Supports GPUs only, supports all IHV GPUs.

#### Platform: M1 & Apple
None.

#### Pros
Well optimized solution with cross platform IHV GPU support.

#### Cons
Windows only. DirectX Compute dispatch is not well supported via d3dvk, atleast not in a performant way. So translating this for use on Vulkan is not very feasible atm.

### openvino
Optimized kernels and library replacements for Intel to compete with CUDA ecosystem.

### Modular Inference Engine (NOT open-source)
Their self reported performance metrics put them on par with optimized CUDA kernel performance.
They use custom DAG and IR for platform specific code optimizations via the compiler on the code side.
And the inference engine seems to be quite interested in scalability as well.

Useless for us though. Completely closed source, and only licensed to businesses.


## Some Thoughts

- Work on adding platform specific GEMMs that are highly optimized and generalize the rest of the inference engine.
- Use cross-platform GEMMs as fallback for platforms and devices which we don't provide optimizations for yet, or we don't plan on supporting explicitly.


## Notes on Distributed Training

Things to look into.

### TORCH.DISTRIBUTED
### [HiveMind by Learning @ Home](https://github.com/learning-at-home/hivemind)
This uses MoE from DeepSpeed.

## Research Papers

### [PagedAttention](https://arxiv.org/abs/2309.06180)
Blogpost: https://vllm.ai/

### [8-bit Optimizer with Block-wise quantization(CUDA only)](https://arxiv.org/abs/2110.02861)
One of the authors's blog post, https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/

### [DeepSpeed MoE & Model Parallelism](https://arxiv.org/abs/2201.05596)
Blogpost: https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/
