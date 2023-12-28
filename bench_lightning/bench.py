import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import lightning as L
import numpy as np
import torch
from inference import generate
from lightning.fabric.plugins import BitsandbytesPrecision
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import load_checkpoint

logging.getLogger("lightning_ai").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Assumption: "Although quantization in Lightning AI supports both gptq and normal with bitsandbytes."
# For benchmarking purposes we are doing with only bitsandbytes, otherwise we need to have two seperate rows
# for LightninAI which can be a bit in-consistent.


class LlamaPyTorchLightningBenchmark:
    def __init__(self, model_path: str, precision: str, device: str) -> None:
        assert precision in [
            "fp16",
            "fp32",
            "int8",
            "int4",
        ], "Supported precision: 'fp16', 'fp32', 'int8', 'int4'"
        assert device in [
            "cuda",
            "cpu",
            "metal",
        ], f"Device {device} is not supported. Supported devices are: 'cuda' or 'cpu' or 'metal'"

        self.model_path, self.precision, self.device = model_path, precision, device
        if self.device == "metal":
            self.device = "mps"

        dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.plugins = None
        self.results = []

        if self.precision == "int4":
            self.precision = "nf4"

        if self.precision in ["int8", "int4"]:
            self.weight_dtype = dtype[
                "fp16"
            ]  # using default fp16 since fp32 not supported.
            self.quant_precision = self.precision
            self.plugins = BitsandbytesPrecision(
                self.quant_precision, self.weight_dtype
            )
            self.precision = None

        else:
            self.precision = "16-true" if precision == "fp16" else "32-true"
            if device == "cpu" and self.precision == "16-true":
                raise ValueError(
                    "When precision is set to 32, then only CPU is supported"
                )

        self.fabric = L.Fabric(
            accelerator=self.device, precision=self.precision, plugins=self.plugins
        )
        self.config = Config.from_json(os.path.join(self.model_path, "lit_config.json"))

    def load_model(self):
        self.tokenizer = Tokenizer(self.model_path)
        with self.fabric.init_module(empty_init=True):
            self.model = GPT(self.config)

        with self.fabric.init_tensor():
            self.model.set_kv_cache(batch_size=1)

        self.model.eval()
        model_file_path = os.path.join(self.model_path, "lit_model.pth")
        load_checkpoint(self.fabric, self.model, model_file_path)
        return self

    @torch.inference_mode()
    def run_model(self, prompt: str, max_tokens: int) -> float:
        encoded = self.tokenizer.encode(prompt, device=self.fabric.device)
        prompt_length = encoded.size(0)

        start_time = time.perf_counter()
        generated = generate(self.model, encoded, max_tokens)
        delta = time.perf_counter() - start_time

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        tokens_generated = generated.size(0) - prompt_length
        return tokens_generated / delta

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
    parser.add_argument(
        "--device", type=str, help="Device on which benchmarking will run."
    )
    args = parser.parse_args()
    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))
    for precision in ("fp16", "fp32", "int8", "int4"):
        logging.info(f"Running Lightning AI Llama benchmark with {precision}")
        try:
            lightning_bench = LlamaPyTorchLightningBenchmark(
                model_path=f"{args.models_dir}/llama-2-7b-lit-gpt",
                precision=precision,
                device=args.device,
            ).load_model()

            lightning_bench.benchmark(
                max_tokens=args.max_tokens,
                prompt=args.prompt,
                repetitions=args.repetitions,
            )

            report["lightningai"][precision] = {
                "mean": np.mean(lightning_bench.results),
                "std": np.std(lightning_bench.results),
            }
        except Exception as e:
            logging.info(f"Error: {e}")
            continue

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
