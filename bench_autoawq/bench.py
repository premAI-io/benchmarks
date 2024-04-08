import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.getcwd())


from common.base import BaseBenchmarkClass  # noqa
from common.cli import get_common_cli_arguments, get_logger  # noqa

logger = get_logger(benchmark_name="auto-gptq", logging_level=logging.INFO)


class AutoAWQBenchmark(BaseBenchmarkClass):
    def __init__(self, model_path: str, precision: int, device: str) -> None:
        assert device == "cuda", "Device other than CUDA is not supported for autoawq."
        assert precision == "int4", "Precision other than INT4 is not supported."

        super().__init__(model_path=model_path, precision=precision, device=device)
        self.logger = logger

    def load_model(self):
        self.model = AutoAWQForCausalLM.from_quantized(
            self.model_path, fuse_layers=True, safetensors=True, strict=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self

    def run_model(self, input_token_or_prompt: str, max_tokens: int):
        output = (
            self.model.generate(
                input_ids=input_token_or_prompt, max_new_tokens=max_tokens
            )
            .detach()
            .tolist()
        )
        return output[0]

    def on_exit(self):
        del self.model
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = get_common_cli_arguments(description="AWQ Benchmark.")
    args = parser.parse_args()

    logger.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )

    report = defaultdict(lambda: defaultdict(float))
    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-autoawq"
        if args.model_name == "llama"
        else f"{args.model_name}-v0.1-7b-autoawq"
    )

    precision = 4

    if args.device == "cpu":
        logger.info("Skipping running model on int4 on CPU, not implemented for Half")
        pass
    else:
        logger.info(
            f"Running AutoGPT benchmark on {args.model_name} with {precision} bit precision"
        )
        autogptq_benchmark = AutoAWQBenchmark(
            model_path=f"{model_folder}/{model_name}",
            device=args.device,
            precision=f"int{precision}",
        ).load_model()
        autogptq_benchmark.benchmark_cuda(
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            repetitions=args.repetitions,
        )

        report[f"{args.model_name} AutoAWQ (token/sec)"][f"INT-{precision}"] = {
            "mean": np.mean(autogptq_benchmark.tps_results),
            "std": np.std(autogptq_benchmark.tps_results),
        }

        report[f"{args.model_name} AutoAWQ memory-usage in MB"][f"INT-{precision}"] = {
            "mean": np.mean(autogptq_benchmark.memory_usage_results),
            "std": np.std(autogptq_benchmark.memory_usage_results),
        }

        logger.info("Benchmark Report")

        for framework, quantizations in report.items():
            for quantization, stats in quantizations.items():
                logger.info(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
