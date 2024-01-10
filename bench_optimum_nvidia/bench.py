import argparse
import logging
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=print,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Optimum-Nvidia is meant for Nvidia GPU usage. Not any other platform is supported.


class LlamaOptimumNvidiaBenchmark:
    def __init__(
        self, model_path: str, precision: str, device: Optional[str] = "cuda"
    ) -> None:
        self.model_path = model_path
        self.precision = precision
        self.results = []
        self.precision_to_dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        # some of the conditions where things can not be supported
        assert precision in ["fp16", "fp32"], ValueError(
            "Supported precisions are: fp16', 'fp32'"
        )
        assert device in ["cuda"], ValueError("Supported devices are: 'cuda'")

        self.model_args = {
            "torch_dtype": self.precision_to_dtype_map[self.precision],
        }
        self.device = device

    def load_model(self):
        """Loads the model into various formats and device"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **self.model_args
        )

        # Hardcoding this for now.
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/models/llama-2-7b-hf")
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        tokenized_input = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        start = time.time()
        generated = self.model.generate(
            input_ids=tokenized_input, max_new_tokens=max_tokens
        )[0]
        delta = time.time() - start

        output = generated.detach().cpu().numpy()
        decoded = self.tokenizer.decode(output[0][0], skip_special_tokens=True)
        return len(self.tokenizer.encode(decoded)) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            print(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)
        del self.model
        if self.device == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTransformers Benchmark.")
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
    print(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))

    for precision in ("fp16", "fp32"):
        print(f"Running Optimum-Nvidia on Llama with precision: {precision}")
        llama_transformers_pytorch_benchmark = LlamaOptimumNvidiaBenchmark(
            model_path=args.models_dir,
            device=args.device,
            precision=precision,
        ).load_model()
        llama_transformers_pytorch_benchmark.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )

        report["llama_optimum_nvidia"][precision] = {
            "mean": np.mean(llama_transformers_pytorch_benchmark.results),
            "std": np.mean(llama_transformers_pytorch_benchmark.results),
        }
    print("Benchmark Report")
    with open(args.log_file, "a") as file:
        for framework, quantizations in report.items():
            for quantization, stats in quantizations.items():
                print(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
                print(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}",
                    file=file,
                )
