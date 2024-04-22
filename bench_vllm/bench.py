import gc
import os
import sys

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils import parallel_state

sys.path.append(os.getcwd())

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class VLLMBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert device == "cuda", ValueError("Only supported device is 'cuda'")
        assert precision in ["float16", "float32", "int4"], ValueError(
            "supported precision are: 'float16', 'float32' and 'int4'"
        )

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

    def load_model_and_tokenizer(self):
        if self.precision == "int4":
            self.model = LLM(
                model=self.model_path, quantization="AWQ", tensor_parallel_size=1
            )
        else:
            self.model = LLM(model=self.model_path)
            self.model.dtype = self.precision
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
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
        prompt = [inputs["prompt"]]

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        output = self.model.generate(prompt, sampling_params)

        generated_text = output[0].outputs[0].text
        generated_tokens = output[0].outputs[0].token_ids

        return {
            "output_tokens": generated_tokens,
            "num_output_tokens": len(generated_tokens),
            "output_prompt": generated_text,
        }

    def postprocess(self, output: dict) -> str:
        return output["output_prompt"]

    def on_exit(self):
        if self.device == "cuda":
            parallel_state.destroy_model_parallel()
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            torch.cuda.synchronize()
        else:
            del self.model


if __name__ == "__main__":
    parser = launch_cli(description="vLLM Benchmark.")
    args = parser.parse_args()

    model_folder = os.path.join(os.getcwd(), "models")
    model_name = (
        f"{args.model_name}-2-7b-chat-"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float32",
                "model_path": os.path.join(model_folder, model_name + "hf"),
            },
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name + "hf"),
            },
            {
                "precision": "int4",
                "model_path": os.path.join(model_folder, model_name + "autoawq"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=VLLMBenchmark,
        runner_dict=runner_dict,
        benchmark_name="vLLM",
        is_bench_pytorch=False,
    )
