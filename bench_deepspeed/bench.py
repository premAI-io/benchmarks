import argparse
import logging
import sys
import time
from collections import defaultdict
from typing import Optional

import mii
import numpy as np
import torch
from transformers import AutoTokenizer

logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaDeepSpeedBenchmark:
    def __init__(
        self,
        model_path: str,
        precision: Optional[str] = "fp16",
        device: Optional[str] = "cuda",
    ) -> None:
        assert precision == "fp16" or precision == "bf16", ValueError(
            "fp32 support is not implemented in DeepSpeed"
        )
        assert device == "cuda", ValueError(
            "Supported device is only cuda for DeepSpeed"
        )
        self.model_path, self.results = model_path, []
        self.device = device

    def load_model(self):
        self.pipeline = mii.pipeline(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        start = time.time()
        output = self.pipeline([prompt], max_new_tokens=max_tokens)
        delta = time.time() - start
        tokens = self.tokenizer(str(output[0]))["input_ids"]
        return len(tokens) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)
        del self.pipeline
        if self.device == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed Benchmark.")
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

    logging.info(
        "Running Transformer benchmark (pytorch backend) on Llama with precision: fp16"
    )

    llama_deepspeed_benchmark = LlamaDeepSpeedBenchmark(
        model_path=f"{args.models_dir}/llama-2-7b-hf", device=args.device
    ).load_model()

    llama_deepspeed_benchmark.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )

    report["llama_deepspeed"]["fp16"] = {
        "mean": np.mean(llama_deepspeed_benchmark.results),
        "std": np.std(llama_deepspeed_benchmark.results),
    }

    logging.info("Benchmark Report")
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
