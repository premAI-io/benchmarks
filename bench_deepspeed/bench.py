import os
import sys

import mii
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class DeepSpeedBenchmark(BaseBenchmarkClass):
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
            model_path=model_path,
            model_name=model_name,
            benchmark_name=benchmark_name,
            precision=precision,
            device=device,
            experiment_name=experiment_name,
        )

        assert precision == "float16", ValueError(
            "Precision other than 'float16' is not supported in DeepSpeed"
        )
        assert device == "cuda", ValueError(
            "Supported device is only cuda for DeepSpeed"
        )

    def load_model_and_tokenizer(self):
        self.model = mii.pipeline(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self

    def preprocess(
        self, prompt: str, chat_mode: bool = True, for_benchmarks: bool = True
    ):
        if chat_mode:
            template = self.get_chat_template_with_instruction(
                prompt=prompt, for_benchmarks=for_benchmarks
            )
            prompt = self.tokenizer.apply_chat_template(template, tokenize=False)

        tokenized_input = self.tokenizer.encode(text=prompt)

        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": None,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        prompt = inputs["prompt"]
        output = self.model(
            [prompt], max_new_tokens=max_tokens, temperature=temperature
        )[0].generated_text

        output_tokens = self.tokenizer.encode(text=output)
        return {
            "output_prompt": output,
            "output_tokens": output_tokens,
            "num_output_tokens": len(output_tokens),
        }

    def postprocess(self, output: dict) -> str:
        return output["output_prompt"]


if __name__ == "__main__":
    parser = launch_cli(description="DeepSpeed Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-hf"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-hf"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name),
            }
        ]
    }

    make_report(
        args=args,
        benchmark_class=DeepSpeedBenchmark,
        runner_dict=runner_dict,
        benchmark_name="DeepSpeed",
        is_bench_pytorch=False,
    )
