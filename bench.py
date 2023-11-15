import argparse
from collections import defaultdict
import logging
import sys

import numpy as np

from python_bench.llama_cpp import LlamaCPPBenchmark
from python_bench.ctranslate import CTranslateBenchmark, get_compute_types
from python_bench.tinygrad import TinyGradBenchmark

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Llama model.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt for the model.",
        default="Explain what is a transformer",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="The maximum number of tokens."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="The number of repetitions for the benchmark.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Flag to indicate whether to use GPU for the benchmark.",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        default=False,
        help="Flag to indicate whether the system uses NVIDIA",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to the log file for writing logs (in append mode).",
        default="benchmark.log",
    )
    args = parser.parse_args()

    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} repetitions={args.repetitions} gpu={args.gpu} nvidia={args.gpu}"
    )
    report = defaultdict(lambda: defaultdict(float))
    for quantize in ("Q8_0", "Q4_0"):
        logging.info(f"Running llama-cpp benchmark with {quantize}")
        llamacpp_bench = LlamaCPPBenchmark(
            f"./models/llama-2-7b-gguf/llama-2-7b.{quantize}.gguf", gpu=args.gpu
        ).load_model()
        llamacpp_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        q = "int8" if quantize == "Q8_0" else "int4"
        report["llama.cpp"][q] = {
            "mean": np.mean(llamacpp_bench.results),
            "std": np.std(llamacpp_bench.results),
        }

    compute_types = get_compute_types(gpu=args.gpu, nvidia=args.nvidia)
    for compute_type in compute_types.intersection({"float16", "int8"}):
        logging.info(f"Running ctranslate benchmark with {compute_type}")
        ctranslate_bench = CTranslateBenchmark(
            f"./models/llama-2-7b-hf-float16",
            gpu=args.gpu,
            compute_type=compute_type,
        ).load_model()
        ctranslate_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        report["ctranslate"][compute_type] = {
            "mean": np.mean(ctranslate_bench.results),
            "std": np.std(ctranslate_bench.results),
        }

    logging.info(f"Running tinygrad benchmark")
    tinygrad_bench = TinyGradBenchmark(
        "./models/llama-2-7b-hf",
        quantize=False,
        device="CPU" if not args.gpu else "GPU" if args.nvidia else "METAL",
    ).load_model()
    tinygrad_bench.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )
    report["tinygrad"]["float16"] = {
        "mean": np.mean(tinygrad_bench.results),
        "std": np.std(tinygrad_bench.results),
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
