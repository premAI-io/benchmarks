import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


def get_logger(
    benchmark_name: str, log_file_path: str = None, logging_level=logging.INFO
):
    logger = logging.getLogger(benchmark_name)
    logger.setLevel(logging_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file_path is None:
        logfile_name = (
            f"benchmark_{benchmark_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
        )
        log_file_path = os.path.join(os.getcwd(), "logs", logfile_name)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def launch_cli(description: str):
    parser = argparse.ArgumentParser(description=description)
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
        "--model_name",
        type=str,
        help="Path to the models directory.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature to use.",
    )

    return parser


def make_report(args, benchmark_class, runner_dict, benchmark_name):
    experiment_name = f"{benchmark_name}-{str(datetime.now())}"
    report = defaultdict(lambda: defaultdict(float))
    all_answers = {}

    for instance in runner_dict[args.device]:
        model_path, precision = instance["model_path"], instance["precision"]
        benchmark = benchmark_class(
            model_path=model_path,
            model_name=args.model_name,
            benchmark_name=benchmark_name,
            precision=precision,
            device=args.device,
            experiment_name=experiment_name,
        ).load_model_and_tokenizer()

        logger = benchmark.logger

        # First we do benchmarking
        benchmark.benchmark(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            repetitions=args.repetitions,
            temperature=args.temperature,
        )

        # Make report for benchmarks
        # Memory seems to be stay the same, so we can take the max of it

        report[f"{args.model_name}-{benchmark_name} (token/sec)"][precision] = {
            "mean": np.mean(benchmark.tps_results),
            "std": np.std(benchmark.tps_results),
        }

        report[f"{args.model_name}-{benchmark_name} (memory usage)"][precision] = {
            "usage": max(benchmark.memory_usage_results)
        }

        # Second we get the answers
        benchmark.get_answers()
        all_answers[precision] = benchmark.answers

    # Make the final report

    for framework, quantizations in report.items():
        for quantization, stats in quantizations.items():
            if framework == f"{args.model_name}-{benchmark_name} (memory usage)":
                logger.info(f"{framework}, {quantization}: {stats['usage']} MB")
            else:
                logger.info(
                    f"{framework}, {quantization}: {stats['mean']:.2f} ± {stats['std']:.2f}"
                )
    # Finally write the quality checks results
    logger.info("Writing the model completion for empirical tests")
    with open(benchmark.answers_json_path, "w") as json_file:
        json.dump(all_answers, json_file)

        logger.info("Benchmarking Fininshed")
