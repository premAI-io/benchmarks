import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ONNXBenchmark:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.provider = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        self.results = []

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = ORTModelForCausalLM.from_pretrained(
            self.model_path,
            use_cache=False,
            use_io_binding=False,
            provider=self.provider,
        )
        return self

    def run_model(self, prompt, max_tokens) -> float:
        device_str = "cuda" if self.device == "cuda" else "cpu"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_str)
        start = time.time()
        gen_tokens = self.model.generate(**inputs, max_length=max_tokens)
        tokens_per_second = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / (
            time.time() - start
        )
        return tokens_per_second

    def benchmark(self, prompt, max_tokens, repetitions):
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONXX Runtime Benchmark for Llama model."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt for the model.",
    )
    parser.add_argument("--max_tokens", type=int, help="The maximum number of tokens.")
    parser.add_argument(
        "--repetitions",
        type=int,
        help="The number of repetitions for the benchmark.",
    )
    parser.add_argument(
        "--device",
        help="Device to use for the benchmark.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to the log file for writing logs (in append mode).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        help="Path to the models directory.",
    )
    args = parser.parse_args()
    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))
    onnx_bench = ONNXBenchmark(
        f"{args.models_dir}/llama-2-7b-onnx",
        device=args.device,
    ).load_model()
    onnx_bench.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )
    report["onnx"]["float16"] = {
        "mean": np.mean(onnx_bench.results),
        "std": np.std(onnx_bench.results),
    }

    logging.info("Benchmark report")
    with open(args.log_file, "a") as file:
        for framework, quantizations in report.items():
            for quantization, stats in quantizations.items():
                logging.info(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
                print(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}",
                    file=file,
                )
