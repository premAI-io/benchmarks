import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
from llama_cpp import Llama

logging.getLogger("llama_cpp").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaCPPBenchmark:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.results = []

    def load_model(self):
        self.model = Llama(model_path=self.model_path, n_gpu_layers=-1, verbose=True)
        return self

    def run_model(self, prompt, max_tokens):
        start = time.time()
        output = self.model.create_completion(prompt, max_tokens=max_tokens)
        tokens = output["usage"]["completion_tokens"]
        return tokens / (time.time() - start)

    def benchmark(self, prompt, max_tokens, repetitions):
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama.cpp Benchmark Llama model.")
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
    for quantize in ("Q8_0", "Q4_0"):
        logging.info(f"Running llama-cpp benchmark with {quantize}")
        llamacpp_bench = LlamaCPPBenchmark(
            f"{args.models_dir}/llama-2-7b-gguf/llama-2-7b.{quantize}.gguf",
            device=args.device,
        ).load_model()
        llamacpp_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        q = "int8" if quantize == "Q8_0" else "int4"
        report["llama.cpp"][q] = {
            "mean": np.mean(llamacpp_bench.results),
            "std": np.std(llamacpp_bench.results),
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
