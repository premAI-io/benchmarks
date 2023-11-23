import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import ctranslate2
import numpy as np
import sentencepiece as spm

logging.getLogger("ctranslate2").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_compute_types(device):
    compute_types = set()
    if device in ("cuda", "cpu"):
        return set(ctranslate2.get_supported_compute_types(device))
    else:
        return compute_types


class CTranslateBenchmark:
    def __init__(self, model_path, device, compute_type):
        self.model_path = model_path
        self.results = []
        self.device = device
        self.compute_type = compute_type

    def load_model(self):
        self.generator = ctranslate2.Generator(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.sp = spm.SentencePieceProcessor(
            os.path.join(self.model_path, "tokenizer.model")
        )
        return self

    def run_model(self, prompt, max_tokens):
        prompt_tokens = ["<s>"] + self.sp.encode_as_pieces(
            f"{B_INST} {prompt.strip()} {E_INST}"
        )
        start = time.time()
        step_results = self.generator.generate_tokens(
            prompt_tokens,
            max_length=max_tokens,
            sampling_temperature=0.6,
            sampling_topk=20,
            sampling_topp=1,
        )
        count = 0
        for _ in self.generate_words(step_results):
            count += 1
        return count / (time.time() - start)

    def benchmark(self, prompt, max_tokens, repetitions):
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(prompt, max_tokens)
            self.results.append(tokens_per_second)

    def generate_words(self, step_results):
        tokens_buffer = []

        for step_result in step_results:
            is_new_word = step_result.token.startswith("▁")

            if is_new_word and tokens_buffer:
                word = self.sp.decode(tokens_buffer)
                if word:
                    yield word
                tokens_buffer = []

            tokens_buffer.append(step_result.token_id)

        if tokens_buffer:
            word = self.sp.decode(tokens_buffer)
            if word:
                yield word


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTranslate Benchmark Llama model.")
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
    if args.device == "metal":
        logging.info(f"Skipping benchmark with device={args.device}")
        sys.exit(0)

    logging.info(
        f"Running benchmark with: max_tokens={args.max_tokens} prompt={args.prompt} "
        + f"repetitions={args.repetitions} device={args.device}"
    )
    report = defaultdict(lambda: defaultdict(float))
    compute_types = get_compute_types(args.device)
    for compute_type in compute_types.intersection({"float16", "int8"}):
        logging.info(f"Running ctranslate benchmark with {compute_type}")
        ctranslate_bench = CTranslateBenchmark(
            f"{args.models_dir}/llama-2-7b-hf-float16",
            device=args.device,
            compute_type=compute_type,
        ).load_model()
        ctranslate_bench.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )
        report["ctranslate"][compute_type] = {
            "mean": np.mean(ctranslate_bench.results),
            "std": np.std(ctranslate_bench.results),
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
