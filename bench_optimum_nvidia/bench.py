import os
import sys

import torch
from optimum.nvidia import AutoModelForCausalLM
from transformers import AutoTokenizer

sys.path.append("/mnt")
sys.path.append("/mnt/benchmarks/")

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class OptimumBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert precision in ["float32", "float16"], ValueError(
            "Supported precision: 'float32' and 'float16'"
        )
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            benchmark_name=benchmark_name,
            experiment_name=experiment_name,
            precision=precision,
            device=device,
            root_folder="/mnt/benchmarks",
        )

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        dtype_mapper = {"float16": torch.float16, "float32": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=dtype_mapper[self.precision],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
        return self

    def preprocess(self, prompt: str, chat_mode: bool = True, for_benchmarks=True):
        if chat_mode:
            template = self.get_chat_template_with_instruction(
                prompt=prompt, for_benchmarks=for_benchmarks
            )
            prompt = self.tokenizer.apply_chat_template(template, tokenize=False)

        tokenized_input = self.tokenizer.encode(text=prompt)
        tensor = self.tokenizer(prompt, return_tensors="pt")
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": tensor,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        tensor = inputs["tensor"]
        num_input_tokens = inputs["num_input_tokens"]

        generated, _ = self.model.generate(
            **tensor,
            top_k=40,
            top_p=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        output_tokens = generated[0].detach().tolist()[num_input_tokens:]
        return {"output_tokens": output_tokens, "num_output_tokens": len(output_tokens)}

    def postprocess(self, output: dict) -> str:
        output_tokens = output["output_tokens"]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return output_text

    def on_exit(self):
        if self.device == "cuda:0":
            del self.model
            torch.cuda.synchronize()
        else:
            del self.model


if __name__ == "__main__":
    parser = launch_cli(description="HF-Optimum Nvidia Benchmark.")
    args = parser.parse_args()

    model_folder = "/mnt/benchmarks/models"
    model_name = (
        f"{args.model_name}-2-7b-chat-optimum"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-optimum"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float32",
                "model_path": os.path.join(model_folder, model_name + "-float32"),
            },
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name + "-float16"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=OptimumBenchmark,
        runner_dict=runner_dict,
        benchmark_name="HF-Optimum Nvidia",
        is_bench_pytorch=False,
    )
