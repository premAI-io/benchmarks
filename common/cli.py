import argparse
import logging
import os
import sys
from datetime import datetime

# Assumption: all the benchmarks are run inside /benchmarks folder
# example: bench_autoawq/bench.sh -d cuda
# not inside bench_autoawq folder


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
        log_file_path = os.path.join(os.getcwd(), "Logs", logfile_name)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_common_cli_arguments(description: str):
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
    return parser
