import argparse
import logging
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np
from ctransformers import AutoModelForCausalLM

logging.getLogger("ctransformers").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaCTransformersBenchmark:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = "cpu",
    ) -> None:
        self.model_path, self.device = model_path, device
        self.results = []
        self.device = device

    def load_model(self):
        # FIXME: Not sure how to get num layers for each model to know how many to fit into VRAM.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_type="llama",
            gpu_layers=50 if self.device in ["cuda", "metal"] else 0,
        )
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        start = time.time()
        output = self.model(prompt, max_new_tokens=max_tokens)
        tokens = len(self.model.tokenize(output))
        return tokens / (time.time() - start)

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)


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
    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))
    for quantize in ("Q8_0", "Q4_0"):
        logging.info(f"Running CTransformer benchmark on Llama with {quantize}")
        llama_ctransformers_bench = LlamaCTransformersBenchmark(
            f"{args.models_dir}/llama-2-7b-gguf/llama-2-7b.{quantize}.gguf",
            device=args.device,
        ).load_model()
        llama_ctransformers_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        q = "int8" if quantize == "Q8_0" else "int4"
        report["llama_ctransformers"][q] = {
            "mean": np.mean(llama_ctransformers_bench.results),
            "std": np.std(llama_ctransformers_bench.results),
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
