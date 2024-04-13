import logging
import os
import sys

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class AutoAWQBenchmark(BaseBenchmarkClass):
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

        # Have to do this step
        # since tokenizer in autoawq is not the instruction tuned one for the instruction tuned model

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        self.model = AutoAWQForCausalLM.from_quantized(
            self.model_path, fuse_layers=True, safetensors=True, strict=False
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
        tensor = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": tensor,
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        tensor = inputs["tensor"]
        num_input_tokens = inputs["num_input_tokens"]

        output = (
            self.model.generate(
                input_ids=tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            .detach()
            .tolist()[0]
        )

        output_tokens = (
            output[num_input_tokens:] if len(output) > num_input_tokens else output
        )

        return {"output_tokens": output_tokens, "num_output_tokens": len(output_tokens)}

    def postprocess(self, output: dict) -> str:
        output_tokens = output["output_tokens"]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)

    def on_exit(self):
        if self.device == "cuda:0":
            del self.model
            torch.cuda.synchronize()
        else:
            del self.model


if __name__ == "__main__":
    parser = launch_cli(description="AWQ Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-autoawq"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-autoawq"
    )

    runner_dict = {
        "cuda": [
            {"precision": "int4", "model_path": os.path.join(model_folder, model_name)}
        ]
    }

    if args.device == "cpu":
        logging.info("Skipping running model on int4 on CPU, not implemented for Half")
        pass
    else:
        make_report(
            args=args,
            benchmark_class=AutoAWQBenchmark,
            runner_dict=runner_dict,
            benchmark_name="AutoAWQ",
            is_bench_pytorch=False,
        )
