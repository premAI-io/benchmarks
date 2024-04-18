import os
import sys

import ctranslate2
from transformers import AutoTokenizer

# have to hard code this thing
sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class CTranslateBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert precision in ["float32", "float16", "int8"], ValueError(
            "Precision other than: 'float32', 'float16', 'int8' are not supported"
        )
        super().__init__(
            model_path=model_path,
            model_name=model_name,
            benchmark_name=benchmark_name,
            precision=precision,
            device=device,
            experiment_name=experiment_name,
        )

    def load_model_and_tokenizer(self):
        self.model = ctranslate2.Generator(self.model_path, device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self

    def preprocess(self, prompt: str, chat_mode: bool = True, for_benchmarks=True):
        if chat_mode:
            template = self.get_chat_template_with_instruction(
                prompt=prompt, for_benchmarks=for_benchmarks
            )
            prompt = self.tokenizer.apply_chat_template(template, tokenize=False)

        tokenized_input = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt)
        )
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": None,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(
        self, inputs: dict, max_tokens: int, temperature: float = 0.1
    ) -> dict:
        tokenized_input = inputs["input_tokens"]
        num_input_tokens = inputs["num_input_tokens"] - 1

        output = self.model.generate_batch(
            [tokenized_input], max_length=max_tokens, sampling_temperature=0.1
        )

        output_tokens = output[0].sequences_ids[0][num_input_tokens:]
        output_prompt = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return {
            "output_prompt": output_prompt,
            "output_tokens": output_tokens,
            "num_output_tokens": len(output_tokens),
        }

    def postprocess(self, output: dict) -> str:
        return output["output_prompt"]


if __name__ == "__main__":
    parser = launch_cli(description="CTransformers Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-ctranslate2-"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-ctranslate2-"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float32",
                "model_path": os.path.join(model_folder, model_name + "float32"),
            },
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name + "float16"),
            },
            {
                "precision": "int8",
                "model_path": os.path.join(model_folder, model_name + "int8"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=CTranslateBenchmark,
        runner_dict=runner_dict,
        benchmark_name="CTranslate2",
        is_bench_pytorch=False,
    )
