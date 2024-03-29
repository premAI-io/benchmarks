import argparse
import logging
import os
import sys
from collections import defaultdict
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LlamaPyTorchBenchmark:
    def __init__(
        self, model_path: str, precision: str, device: Optional[str] = "cpu"
    ) -> None:
        self.model_path = model_path
        self.precision = precision
        self.results = []
        self.precision_to_dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        # some of the conditions where things can not be supported
        assert precision in ["fp16", "fp32", "int4", "int8"], ValueError(
            "Supported precisions for this benchmarks are: 'fp16', 'fp32', 'int4', 'int8'"
        )
        assert device in ["cpu", "cuda", "metal"], ValueError(
            "Supported devices are: 'cpu', 'cuda', 'metal'"
        )

        if device == "cpu" and precision != "fp32":
            raise ValueError(
                "When device is set to CPU, fp32 is the only supported precision."
            )

        self.device = "cuda:0" if device == "cuda" else device

    @torch.inference_mode()
    def load_model(self):
        """Loads the model into various formats and device."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.precision in ["fp16", "fp32"]:
            model_args = {
                "device_map": self.device,
                "torch_dtype": self.precision_to_dtype_map[self.precision],
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_args
            )
        elif self.precision in ["int4", "int8"] and self.device == "cuda:0":
            from transformers import BitsAndBytesConfig

            bnb_config = (
                BitsAndBytesConfig(load_in_8bit=True)
                if self.precision == "int8"
                else BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            )

            if self.precision == "int8":
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map=self.device, quantization_config=bnb_config
            )
        else:
            raise ValueError(
                f"Invalid configuration: {self.device}, {self.precision}"
                "INT4/8 requires CUDA to execute."
            )

        return self

    @torch.inference_mode()
    def run_model(self, prompt: str, max_tokens: int) -> float:
        tokenized_input = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        start = perf_counter()
        output = self.model.generate(
            input_ids=tokenized_input, max_new_tokens=max_tokens
        )
        delta = perf_counter() - start
        return len(output[0]) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)
        del self.model
        if self.device == "cuda:0":
            torch.cuda.synchronize()


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

    precisions_mapping = {
        "cpu": ("fp32",),
        "cuda": ("fp32", "fp16", "int8", "int4"),
        "metal": ("fp32", "fp16"),
    }
    for precision in precisions_mapping[args.device]:
        logging.info(
            f"Running Transformer benchmark (pytorch backend) on Llama with precision: {precision}"
        )
        llama_transformers_pytorch_benchmark = LlamaPyTorchBenchmark(
            model_path=f"{args.models_dir}/llama-2-7b-hf",
            device=args.device,
            precision=precision,
        ).load_model()
        llama_transformers_pytorch_benchmark.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )

        report["llama_transformers_pytorch"][precision] = {
            "mean": np.mean(llama_transformers_pytorch_benchmark.results),
            "std": np.std(llama_transformers_pytorch_benchmark.results),
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
