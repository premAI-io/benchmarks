import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache
from exllamav2.config import ExLlamaV2Config
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from exllamav2.tokenizer import ExLlamaV2Tokenizer

logging.getLogger("llama_cpp").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ExllamaV2Benchmark:
    def __init__(self, model_path: str) -> None:
        self.model_path, self.results = model_path, []

    def load_model(self):
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.model_path
        self.config.prepare()

        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.85
        self.settings.top_k = 50
        self.settings.top_p = 0.8
        self.settings.token_repetition_penalty = 1.05
        self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        self.generator.warmup()
        return self

    @torch.inference_mode()
    def run_model(self, prompt: str, max_tokens: int) -> float:
        start = time.time()
        _ = self.generator.generate_simple(prompt, self.settings, max_tokens, seed=1234)
        delta = time.time() - start
        return len(self.generator.sequence_ids[0]) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
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
        + f"repetitions={args.repetitions} device=cuda"
    )
    report = defaultdict(lambda: defaultdict(float))
    for quantize in ("q8", "q4"):
        logging.info(f"Running ExllamaV2 benchmark with {quantize}")
        llamacpp_bench = ExllamaV2Benchmark(
            f"{args.models_dir}/llama-2-7b-exllamav2-{quantize}"
        ).load_model()
        llamacpp_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        q = "int8" if quantize == "q8" else "int4"
        report["exllamav2"][q] = {
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
