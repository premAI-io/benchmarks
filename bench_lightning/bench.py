import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from bench_lightning.inference import generate, load_model  # noqa
from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class PyTorchLightningBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            precision=precision,
            device=device,
        )

        self.quantization_precision_mapping = {
            "float16": {"precision": "16-true", "quantize": None},
            "float32": {"precision": "32-true", "quantize": None},
            "int8": {"precision": "16-true", "quantize": "bnb.int8"},
        }

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        self.model, self.lit_tokenizer, self.prompt_style, self.fabric = load_model(
            checkpoint_dir=self.model_path,
            quantize=self.quantization_precision_mapping[self.precision]["quantize"],
            precision=self.quantization_precision_mapping[self.precision]["precision"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
        return self

    def preprocess(
        self, prompt: str, chat_mode: bool = True, for_benchmarks: bool = True
    ):
        return {"prompt": prompt}

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        prompt = inputs["prompt"]
        output = generate(
            model=self.model,
            tokenizer=self.lit_tokenizer,
            prompt_style=self.prompt_style,
            fabric=self.fabric,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        output_prompt = self.tokenizer.decode(
            output["output_tokens"], skip_special_tokens=True
        )
        return {**output, "output_prompt": output_prompt}

    def postprocess(self, output: dict) -> str:
        return output["output_prompt"]


if __name__ == "__main__":
    parser = launch_cli(description="PyTorch Lightning")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-litgpt"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-litgpt"
    )

    model_path = Path(os.path.join(model_folder, model_name))

    runner_dict = {
        "cuda": [
            {"precision": "float16", "model_path": model_path},
            {"precision": "float32", "model_path": model_path},
            {"precision": "int8", "model_path": model_path},
        ]
    }

    make_report(
        args=args,
        benchmark_class=PyTorchLightningBenchmark,
        runner_dict=runner_dict,
        benchmark_name="PyTorch Lightning",
        is_bench_pytorch=False,
    )
