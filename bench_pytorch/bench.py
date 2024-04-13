import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class PyTorchBenchmark(BaseBenchmarkClass):
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

    @torch.inference_mode()
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        precision_dtype_mapping = {"float16": torch.float16, "float32": torch.float32}

        if self.precision in ["float16", "float32"]:
            device = "cuda:0" if self.device == "cuda" else self.device
            model_args = {
                "device_map": device,
                "torch_dtype": precision_dtype_mapping[self.precision],
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_args
            )
        elif self.precision in ["int4", "int8"] and self.device in ["cuda:0", "cuda"]:
            from transformers import BitsAndBytesConfig

            bnb_config = (
                BitsAndBytesConfig(load_in_8bit=True)
                if self.precision == "int8"
                else BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            )

            if self.precision == "int8":
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map=self.device, quantization_config=bnb_config
            )
        else:
            raise ValueError(
                f"Invalid configuration: {self.device}, {self.precision}"
                "INT4/8 requires CUDA to execute."
            )
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
                pad_token_id=self.tokenizer.eos_token_id,
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
    parser = launch_cli(
        description="HuggingFace Transformers Benchmark (PyTorch backend)"
    )
    args = parser.parse_args()
    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-hf"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-hf"
    )
    model_path = os.path.join(model_folder, model_name)
    precisions_mapping = {
        "cpu": ("float32",),
        "cuda": ("float32", "float16", "int8", "int4"),
        "metal": ("float32", "float16"),
    }
    runner_dict = {}
    for device, precisions in precisions_mapping.items():
        runner_dict[device] = [
            {"precision": precision, "model_path": model_path}
            for precision in precisions
        ]
    make_report(
        args=args,
        benchmark_class=PyTorchBenchmark,
        runner_dict=runner_dict,
        benchmark_name="HF-Transformers (PyTorch Backend)",
    )
