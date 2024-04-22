import os
import sys

import tensorrt_llm
import torch
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer

sys.path.append("/mnt")
sys.path.append("/mnt/benchmarks/")

from common.base import BaseBenchmarkClass  # noqa
from common.utils import launch_cli, make_report  # noqa


class TensorRTLLMBenchmark(BaseBenchmarkClass):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        benchmark_name: str,
        precision: str,
        device: str,
        experiment_name: str,
    ) -> None:
        assert precision in ["float16", "int8", "int4"], ValueError(
            "Supported precision: 'float16', 'int8' and 'int4'"
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
        self.runtime_rank = tensorrt_llm.mpi_rank()
        if model_name == "llama":
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "llama-2-7b-chat-hf"
            )
        else:
            self.tokenizer_folder = os.path.join(
                self.root_folder, "models", "mistral-7b-v0.1-instruct-hf"
            )

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_folder)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        # load the runner kawargs
        runner_kwargs = dict(
            engine_dir=self.model_path,
            rank=self.runtime_rank,
            max_batch_size=1,
            max_input_len=512,
            max_output_len=512,
            max_beam_width=1,
            max_attention_window_size=None,
            sink_token_length=None,
        )
        self.model = ModelRunnerCpp.from_dir(**runner_kwargs)
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
        tensor = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True
        ).squeeze(0)
        return {
            "prompt": prompt,
            "input_tokens": tokenized_input,
            "tensor": [tensor],
            "num_input_tokens": len(tokenized_input),
        }

    def run_model(self, inputs: dict, max_tokens: int, temperature: float) -> dict:
        tensor = inputs["tensor"]
        num_input_tokens = inputs["num_input_tokens"]

        with torch.no_grad():
            output = self.model.generate(
                tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                pad_id=self.pad_id,
                end_id=self.end_id,
                return_dict=True,
            )

        output_ids = output["output_ids"]
        output_tokens = output_ids[0][0].detach().cpu().tolist()[num_input_tokens:]

        return {
            "output_tokens": output_tokens,
            "num_output_tokens": len(output_tokens),
        }

    def postprocess(self, output: dict) -> str:
        output_tokens = output["output_tokens"]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return output_text

    def on_exit(self):
        del self.model
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = launch_cli(description="Nvidia TRT-LLM Benchmark.")
    args = parser.parse_args()

    model_folder = "/mnt/benchmarks/models"
    model_name = (
        f"{args.model_name}-2-7b-chat-trt"
        if args.model_name == "llama"
        else f"{args.model_name}-7b-v0.1-instruct-trt"
    )

    runner_dict = {
        "cuda": [
            {
                "precision": "float16",
                "model_path": os.path.join(model_folder, model_name + "-float16"),
            },
            {
                "precision": "int8",
                "model_path": os.path.join(model_folder, model_name + "-int8"),
            },
            {
                "precision": "int4",
                "model_path": os.path.join(model_folder, model_name + "-int4"),
            },
        ]
    }

    make_report(
        args=args,
        benchmark_class=TensorRTLLMBenchmark,
        runner_dict=runner_dict,
        benchmark_name="Nvidia-TRT-LLM",
        is_bench_pytorch=False,
    )
