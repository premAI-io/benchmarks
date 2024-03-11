import argparse
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from vllm import LLM
from vllm.model_executor.parallel_utils import parallel_state

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaVLLMBenchmark:
    def __init__(self, model_path: str, device: str, precision: str):
        # VLLM is not supported for CPU issue: https://github.com/vllm-project/vllm/issues/176
        # VLLM also not supports Metal, issue: https://github.com/vllm-project/vllm/issues/1441

        assert device == "cuda", ValueError("Supported device is cuda only.")
        assert precision in ["fp16", "fp32", "int4"], ValueError(
            "supported precision are: fp16, fp32 and int4"
        )

        self.model_path, self.precision, self.device = model_path, precision, device
        self.results = []
        self.precision_map = {"fp16": "float16", "fp32": "float32"}

    def load_model(self):
        if self.precision != "int4":
            self.model = LLM(model=self.model_path)
            self.model.dtype = self.precision_map[precision]
        else:
            self.model = LLM(model=self.model_path, quantization="AWQ")
        return self

    def run_model(self, prompt: str, max_tokens: int) -> float:
        self.model.max_num_seqs = max_tokens
        start = time.time()
        output = self.model.generate(prompts=[prompt])
        delta = time.time() - start
        return len(output[0].outputs[0].token_ids) / delta

    def benchmark(
        self,
        prompt: str,
        max_tokens: int,
        repetitions: int,
    ) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)

        del self.model
        if self.device == "cuda":
            parallel_state.destroy_model_parallel()
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vllm Benchmark.")
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

        llama_vllm_bench = LlamaVLLMBenchmark(
            f"{args.models_dir}/llama-2-7b-hf"
            if precision != "int4"
            else f"{args.models_dir}/llama-2-7b-autoawq",
            device=args.device,
            precision=precision,
        ).load_model()

        llama_vllm_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )

        report["llama_vllm"][precision] = {
            "mean": np.mean(llama_vllm_bench.results),
            "std": np.std(llama_vllm_bench.results),
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
