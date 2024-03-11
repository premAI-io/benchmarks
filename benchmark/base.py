import argparse
import logging
import sys
import torch 
import time
from collections import defaultdict
from benchmark.utils import profile_latency, profile_usage


class LlamaBenchmarkBase:
    def __init__(self, model_dir_path: str, device: str, *args, **kwargs) -> None:
        self.model_dir_path, self.device = model_dir_path, device 
        self.results = []
    
    def load_model(self):
        return self 

    @profile_usage
    @profile_latency
    def run_model(self, prompt: str, max_tokens: int, *args, **kwargs):
        raise NotImplementedError
    
    def benchmark(self, prompt: str, max_tokens: int, repetitions: int, *args, **kwargs) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            (latency, memory_usage), results = self.run_model(
                prompt=prompt, max_tokens=max_tokens, *args, **kwargs
            )
            
            print(latency, memory_usage)
            
            self.results.append((latency, memory_usage))
            
        del self.model
        if self.device == "cuda":
            torch.cuda.synchronize()

def benchmark_arg_parser(name: str, benchmark_class):
    parser = argparse.ArgumentParser(description=f"{name} Benchmark.")
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

    for precision in ("fp32", "fp16", "int4"):
        logging.info(f"Running VLLM benchmark on Llama on {precision} precision.")

        llama_vllm_bench = benchmark_class(
            f"{args.models_dir}/llama-2-7b-hf"
            if precision != "int4"
            else f"{args.models_dir}/llama-2-7b-autoawq",
            device=args.device,
            precision=precision,
        ).load_model()

        llama_vllm_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )