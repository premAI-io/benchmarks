import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import AutoTokenizer

logging.getLogger("tensorrt_llm").setLevel(logging.ERROR)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# todo: investigate for int-4/8 precision


class LlamaTensorRTMBenchmark:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        precision: str,
        device: Optional[str] = "cuda",
    ) -> None:
        assert precision in ["fp32", "fp16"], ValueError(
            "Supported Precision: 'fp32' or 'fp16'"
        )
        assert device == "cuda", ValueError("Supported device: 'cuda'")

        self.engine_dir_path = Path(model_path)
        engine_files = list(self.engine_dir_path.glob("*.engine"))

        if len(engine_files) == 0:
            raise ValueError(".engine file does not exist. Try to build the engine.")

        self.engine_path = engine_files[0]
        self.config_path = self.engine_dir_path / "config.json"

        self.precision, self.device = precision, device
        self.result = []
        self.tokenizer_path = tokenizer_path

    def load_model(self):
        with open(self.config_path) as f:
            config = json.load(f)

        # set the precision here

        use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
        remove_input_padding = config["plugin_config"]["remove_input_padding"]
        tp_size = config["builder_config"]["tensor_parallel"]
        pp_size = config["builder_config"]["pipeline_parallel"]
        world_size = tp_size * pp_size

        num_heads = config["builder_config"]["num_heads"] // tp_size
        hidden_size = config["builder_config"]["hidden_size"] // tp_size
        vocab_size = config["builder_config"]["vocab_size"]
        num_layers = config["builder_config"]["num_layers"]
        num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
        paged_kv_cache = config["plugin_config"]["paged_kv_cache"]

        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

        model_config = ModelConfig(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            paged_kv_cache=paged_kv_cache,
            remove_input_padding=remove_input_padding,
        )

        world_size = tp_size * pp_size
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(
            world_size, runtime_rank, tp_size=tp_size, pp_size=pp_size
        )

        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        with open(self.engine_path, "rb") as f:
            engine_buffer = f.read()

        self.model = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping
        )
        return self

    def run_model(self, input_ids, input_lengths, sampling_config):
        start = time.time()
        output_ids = self.model.decode(input_ids, input_lengths, sampling_config)
        delta = time.time() - start
        return len(output_ids.detach().cpu().numpy()[0][0]) / delta

    def benchmark(self, prompt: str, max_tokens: int, repetitions: int) -> None:
        input_tokens = []
        input_tokens.append(self.tokenizer.encode(prompt, add_special_tokens=False))

        input_lengths = torch.tensor(
            [len(x) for x in input_tokens], dtype=torch.int32, device="cuda"
        )
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda").unsqueeze(
            0
        )

        max_input_length = torch.max(input_lengths).item()
        self.model.setup(input_lengths.size(0), max_input_length, max_tokens, 1)

        sampling_config = SamplingConfig(
            end_id=2, pad_id=2, num_beams=1, temperature=0.1
        )
        for i in range(repetitions):
            logging.info(
                f"Running repetition [{str(i+1).zfill(len(str(repetitions)))}/{repetitions}]"
            )
            tokens_per_second = self.run_model(
                input_ids, input_lengths, sampling_config
            )
            self.results.append(tokens_per_second)
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

    for precision in ("fp16", "fp32"):
        logging.info(
            f"Running TensorRT LLM benchmark (pytorch backend) on Llama with precision: {precision}"
        )
        llama_tensorrt_benchmark = LlamaTensorRTMBenchmark(
            model_path=f"{args.models_dir}/llama-2-7b-nvidia_tensorrt_build_{precision[2:]}",
            device=args.device,
            precision=precision,
        ).load_model()

        llama_tensorrt_benchmark.benchmark(
            max_tokens=args.max_tokens, prompt=args.prompt, repetitions=args.repetitions
        )

        report["llama_transformers_pytorch"][precision] = {
            "mean": np.mean(llama_tensorrt_benchmark.results),
            "std": np.std(llama_tensorrt_benchmark.results),
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
