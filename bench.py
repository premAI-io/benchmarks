import argparse
import logging

from python_bench.llama_cpp import LlamaCPPBenchmark
from python_bench.ctranslate import CTranslateBenchmark
from python_bench.tinygrad import TinyGradBenchmark

LOGGER = logging.getLogger("bench")
LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

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
        default=1,
        help="The number of repetitions for the benchmark.",
    )
    args = parser.parse_args()

    LOGGER.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} repetitions={args.repetitions}"
    )
    LOGGER.info(f"Running llama-cpp benchmark")
    llamacpp_bench = LlamaCPPBenchmark(
        "./models/Llama-2-7B-GGUF/llama-2-7b.Q8_0.gguf"
    ).load_model()
    llamacpp_result = llamacpp_bench.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )

    LOGGER.info(f"Running ctranslate benchmark")
    ctranslate_bench = CTranslateBenchmark(
        "./models/Llama-2-7b-chat-hf-ct2-int8"
    ).load_model()
    ctranslate_result = ctranslate_bench.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )

    LOGGER.info(f"Running tinygrad benchmark")
    tinygrad_bench = TinyGradBenchmark(
        "./models/llama-2-7b-hf", quantize=False
    ).load_model()
    tinygrad_result = tinygrad_bench.benchmark(
        max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
    )

    LOGGER.info("Benchmark report")
    LOGGER.info(f"llama-cpp ended with {llamacpp_result:.2f} token/s")
    LOGGER.info(f"ctranslate ended with {ctranslate_result:.2f} token/s")
    LOGGER.info(f"tinygrad ended with {tinygrad_result:.2f} token/s")
