import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.getLogger("auto-gptq").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaAutoAWQBenchmark:
    def __init__(self, model_path: str, precision: int, device: str) -> None:
        assert device == "cuda", "Device other than CUDA is not supported for autoawq."
        assert precision == "int4", "Precison other than INT4 is not supported."

        self.model_path, self.precision, self.device = (
            model_path,
            precision,
            "cuda:0" if device == "cuda" else device,
        )
        self.results = []

    def load_model(self):
        """Loads the model in the required precision."""
        self.model = AutoAWQForCausalLM.from_quantized(
            self.model_path, fuse_layers=True, safetensors=True, strict=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        tokenized_input = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        start = time.time()
        output = (
            self.model.generate(input_ids=tokenized_input, max_new_tokens=max_tokens)
            .detach()
            .cpu()
            .numpy()
        )
        delta = time.time() - start
        return len(output[0]) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)
        del self.model
        if self.device == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoAWQ Benchmark.")
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

    precision = 4

    if args.device == "cpu":
        logging.info("Skipping running model on int4 on CPU, not implemented for Half")
        pass
    else:
        logging.info(
            f"Running AutoGPT benchmark on Llama with {precision} bit precision"
        )
        llama_autogptq_benchmark = LlamaAutoAWQBenchmark(
            model_path=f"{args.models_dir}/llama-2-7b-autoawq",
            device=args.device,
            precision=f"int{precision}",
        ).load_model()
        llama_autogptq_benchmark.benchmark(
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            repetitions=args.repetitions,
        )

        report["Llama AutoAWQ"][f"INT-{precision}"] = {
            "mean": np.mean(llama_autogptq_benchmark.results),
            "std": np.std(llama_autogptq_benchmark.results),
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
