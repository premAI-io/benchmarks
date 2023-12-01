# 🔧 ML Engines

## Features

| Features                    | pytorch | burn | llama.cpp | candle | tinygrad | onnxruntime | CTranslate2 |
| --------------------------- | ------- | ---- | --------- | ------ | -------- | ----------- | ----------- |
| Inference support           | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 16-bit quantization support | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 8-bit quantization support  | ✅      | ❌   | ✅        | ✅     | ✅       | ✅          | ✅          |
| 4-bit quantization support  | ✅      | ❌   | ✅        | ✅     | ❌       | ❌          | ❌          |
| 2/3bit quantization support | ✅      | ❌   | ✅        | ✅     | ❌       | ❌          | ❌          |
| CUDA support                | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ✅          |
| ROCM support                | ✅      | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Intel OneAPI/SYCL support   | ✅**    | ✅   | ✅        | ✅     | ✅       | ❌          | ❌          |
| Mac M1/M2 support           | ✅      | ✅   | ✅        | ⭐     | ✅       | ✅          | ⭐          |
| BLAS support(CPU)           | ✅      | ✅   | ✅        | ✅     | ❌       | ✅          | ✅          |
| Model Parallel support      | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Tensor Parallel support     | ✅      | ❌   | ❌        | ✅     | ❌       | ❌          | ✅          |
| Onnx Format support         | ✅      | ✅   | ✅        | ✅     | ✅       | ✅          | ❌          |
| Training support            | ✅      | 🌟   | ❌        | 🌟     | ❌       | ❌          | ❌          |

⭐ = No Metal Support
🌟 = Partial Support for Training (Finetuning already works, but training from scratch may not work)

# Comparative Analysis of AI Frameworks

## Burn vs PyTorch

### Performance for Torch Backend:

Burn's performance is noticeably better or comparable to using the torch backend directly. This is attributed to optimizations such as In-place Operations and stronger typing, enabling static resolution of constant known parameters (e.g., dims and dtype) [1](https://burn-rs.github.io/blog/burn-rusty-approach-to-tensor-handling).

### Other Backends

- **Candle:** Experimental, not properly supported.
- **WebGPU:** Beta, but not well-implemented or mature, requires further work. Notable optimizations and fixes are in progress.
- **nd-array:** CPU only, with BLAS support. No cuBLAS or similar partial GPU support planned yet.

#### Overall performance is on par with PyTorch, especially with the torch backend.

## vLLM

- Early research project with limitations.
- Utilizes PagedAttention for performance improvement [read more](https://arxiv.org/abs/2309.06180).

### Platform: x86 & GPU

- No CPU support.
- GPU support limited to Nvidia.

### Platform: M1 & Apple

- No CPU support.
- No ANE or Accelerate support.

### Pros

- Can be used on top of any backend for efficient memory usage.

### Cons

- Extremely limited platform support.
- Requires substantial effort to port to non-CUDA systems.
- Largest gains come from efficient memory sharing, tailored for parallel workloads.

## FasterTransformer

- Model library for encoder and decoder components optimized for CUDA platforms.
- Supports PyTorch, Tensorflow, or Triton backend.

### Platform: x86 & GPU

- No CPU or non-CUDA support.

### Platform: M1 & Apple

- No support.

### Pros

- Potential to borrow CUDA kernel optimizations for high-performance platforms.

### Cons

- Extremely limited support.

## tensor-rt

- Developed by Nvidia.

### Platform: x86 & GPU

- No CPU or non-CUDA support.

### Platform: M1 & Apple

- Not applicable.

### Pros

- Straightforward library with wide model format and library support.
- Can be used as a tensor backend for other projects.

### Cons

- Extremely limited platform support.

## Tinygrad

- Simple AI inference framework/library.
- Focus on layered primitive abstraction model.

### Platform Support

- CPU performance is lackluster.
- Most GPU platforms supported using WebGPU, CUDA, ROCM, SYCL/OneAPI, and Metal/CoreML.

### Pros

- Simple and cross-platform support.

### Cons

- Not performant for most targets.
- Lacks scalability for large high-compute platforms.

## GGML

- Portable library in C & C++.
- Works with weights directly and GGUF format for model loading/unloading.

### Platform: x86 & GPU

- CPUs supported with BLAS and hand-rolled implementation.
- CUDA, ROCM, OpenVINO present with comparable performance to standard torch.

### Platform: M1 & Apple

- Supports M1 (ARM-NEON) CPU.
- Supports MPS (Accelerate) GPU.
- Supports CoreML, but not ANE (Apple Neural Engine).

### Pros

- Support for non-standard formats like int4 and int5.
- Fairly simple and easy-to-use library.

### Cons

- Sparse documentation and tutorials.
- Model network must be rewritten for each new model.

## Tensorflow Lite

- Suitable for executing ML models on mobile, embedded devices, and web platforms.
- Generally behind PyTorch in performance and flexibility.

## ONNX

- Intermediate Representation (IR) format for ML models.
- Simplifies deployment for inference across runtimes.

### wonnx (webgpu + onnx + rust)

- Supports a decent subset of ONNX IR.
- Works on all platforms and architectures.

## DirectML

- Open-source Windows-only platform for ML ops on DirectX 12 Compute Dispatch.
- Supports GPUs with ONNX format.

### Platform: x86 & GPU

- Supports all IHV GPUs.

### Platform: M1 & Apple

- Not supported.

### Pros

- Well-optimized solution with cross-platform GPU support.

### Cons

- Windows-only.
- DirectX Compute dispatch not well supported via d3dvk.

## OpenVINO

- Optimized kernels and library replacements for Intel to compete with CUDA ecosystem.

## Modular Inference Engine (NOT open-source)

- Closed-source, licensed only to businesses.
- Self-reported performance metrics on par with optimized CUDA kernel performance.
- Custom DAG and IR for platform-specific code optimizations.

Useless for general use due to being closed source.
