import logging
import os
import sys

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa

_MESSAGE = """
GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain
in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16.
"""


class AutoGPTQBenchmark(BaseBenchmarkClass):
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

        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                os.getcwd(), "models", "mistral-7b-v0.1-instruct-hf"
            )

        self.precision_map = {"float16": torch.float16, "float32": torch.float32}

    def load_model_and_tokenizer(self):
        device = "cuda:0" if self.device == "cuda" else self.device

        if self.model_name == "llama":
            if self.precision == "float16":
                use_marlin = True
            else:
                use_marlin = False
        else:
            use_marlin = False

        self.model = AutoGPTQForCausalLM.from_quantized(
            self.model_path,
            device=device,
            use_marlin=use_marlin,
            torch_dtype=self.precision_map[self.precision],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
        return self

    def preprocess(self, prompt: str, chat_mode: bool = True):
        if chat_mode:
            prompt = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

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


if __name__ == "__main__":
    parser = launch_cli(description="AutoGPTQ Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-autogptq"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-autogptq"
    )
    logging.info(_MESSAGE)

    runner_dict = {
        "cuda": [
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name),
            },
            {
                "precision": "float32",
                "model_path": os.path.join(model_folder, model_name),
            },
        ]
    }

    if args.device == "cpu":
        logging.info("Skipping running model on int4 on CPU, not implemented for Half")
        pass
    else:
        make_report(
            args=args,
            benchmark_class=AutoGPTQBenchmark,
            runner_dict=runner_dict,
            benchmark_name="AutoGPTQ",
        )
