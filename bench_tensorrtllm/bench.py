import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import tensorrt_llm
from tensorrt_llm.model import TRTModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class TRTLLMBenchmark:
    def __init__(self, model_path, device="gpu"):
        self.model_path = model_path
        self.device = device

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = TRTModelForCausalLM.from_pretrained(self.model_path)
        return self

    def run_model(self, prompt, max_tokens) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        start = time.time()
        gen_tokens = self.model.generate(**inputs, max_length=max_tokens)
        tokens_per_second = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / (
            time.time() - start
        )
        return tokens_per_second

    def benchmark(self, prompt, max_tokens, repetitions):
        results = []
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            results.append(tokens_per_second)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRT-LLM Benchmark.")
    parser.add_argument("--prompt", type=str, help="The prompt for the model.")
    parser.add_argument("--max_tokens", type=int, help="The maximum number of tokens.")
    parser.add_argument("--repetitions", type=int, help="The number of repetitions for the benchmark.")
    parser.add_argument("--device", help="Device to use for the benchmark.")
    parser.add_argument("--log_file", type=str, help="Path to the log file for writing logs (in append mode).")
    parser.add_argument("--models_dir", type=str, help="Path to the models directory.")

    args = parser.parse_args()
    logging.info(f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} repetitions={args.repetitions} device={args.device}")

    report = defaultdict(lambda: defaultdict(float))
    trtllm_bench = TRTLLMBenchmark(f"{args.models_dir}/model_name", device=args.device).load_model()
    results = trtllm_bench.benchmark(max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions)

    report["trtllm"]["default"] = {
        "mean": np.mean(results),
        "std": np.std(results),
    }

    logging.info("Benchmark report")
    with open(args.log_file, "a") as file:
        for framework, quantizations in report.items():
            for quantization, stats in quantizations.items():
                logging.info(f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}", file=file)
